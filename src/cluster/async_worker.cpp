#include <thread>
#include "cluster/timer.hpp"
#include "cluster/async_worker.hpp"

template <typename Dtype>
AsyncWorker<Dtype>::AsyncWorker(const SyncCommConfig<Dtype>& sync_comm_config, 
	const AsyncCommConfig<Dtype>& async_comm_config) : 
	Worker<Dtype> (sync_comm_config),
	async_mem_(2, this->sync_comm_.mpi_sync_buf_size_),
	async_comm_(async_comm_config) {
	async_comm_.AttachAsyncMem(&async_mem_);
}


template <typename Dtype>
void AsyncWorker<Dtype>::CommitDiffToAsyncMem(Dtype* diff_buf) {
	Dtype scale = N_PROC_PER_GROUP * N_DEVICE_PER_PROC;
	// TODO Jian: replace this with more efficient blas implementation
	for (int i = 0; i < async_mem_.buf_size_; i++)
		async_mem_.buf_[i] += diff_buf[i] / scale;
}


template <typename Dtype>
void AsyncWorker<Dtype>::AsyncComputeLoop() {

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in AsyncComputeLoop function");
#endif

	// prevent trigger send overlaps with recv thread in wait on the same buf. 
	this->async_comm_.ThreadBarrierWait();

	// int test_rank;
	// MPI_Comm_rank(MPI_COMM_WORLD, &test_rank);

#ifdef TEST
	std::vector<Dtype> test_res;
#endif

#ifdef DEBUG
	Timer timer;
#endif 

	for (int i = 0; i < this->solver_.n_iter_; i++) {

#ifdef DEBUG
	timer.start();
#endif 

		// b1: wait until comm thread finish receive
		this->async_comm_.ThreadBarrierWait();

#ifdef DEBUG
	timer.stop();
	// DEBUG_PRINT_RANK(MPI_COMM_WORLD, "")
	DEBUG_PRINT_TIME(timer.getElapsedTimeInMilliSec(), "Receive wait ");
	timer.start();
#endif

		// commit delta to mem_
		this->CommitDiffToAsyncMem(this->sync_comm_.GetMpiSyncBuffer() );
		// this->solver_.CommitModelDiff(this->async_mem_.buf_, this->async_mem_.buf_size_);

		// b2: wait until finish update delta to mem_
		this->async_comm_.ThreadBarrierWait();

		// read mem_ for compute
		this->solver_.RecvModel(this->async_mem_.buf_, this->async_mem_.buf_size_);

#ifdef TEST
		Dtype* host_buf = new Dtype[this->solver_.buf_size_];
		CUDA_CHECK(cudaMemcpy(host_buf, this->solver_.model_, 
			sizeof(Dtype) * this->solver_.buf_size_, cudaMemcpyDeviceToHost) );
		test_res.push_back(host_buf[0] );
		for (int i = 0; i < this->solver_.buf_size_; i++)
			assert(host_buf[0] == host_buf[i] );
		delete[] host_buf;
#endif

		// b3 : wait until finish read mem_ out before recv
		this->async_comm_.ThreadBarrierWait();

		// b_data: wait until data loading is done 
		pthread_barrier_wait(&(this->data_ready_) );

		// do computation
		this->solver_.Compute();

		/**
		 * do intra-group synchronization, we do not do braodcast after inter-machine
		 * all reduce. As in async worker the new model will be copied from asyn_mem
		 * directly to gpu memory.
		 */
		this->sync_comm_.SyncGroup(false);

		std::cout << "rank " << async_comm_.config_.mpi_rank_ << " round " 
			<< i << " done " << std::endl;
		
#ifdef DEBUG
	timer.stop();
	// DEBUG_PRINT_RANK(MPI_COMM_WORLD, "")
	DEBUG_PRINT_TIME(timer.getElapsedTimeInMilliSec(), "Computing in ");
#endif

	}

#ifdef TEST
	// verify the pattern of the result
	int n_proc;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
	std::cout << "rank " << rank << " value " << test_res[0] << std::endl;
	assert(test_res[0] == 0);
	for (int i = 1; i < test_res.size(); i++) {
		std::cout << "rank " << rank << " value " << test_res[i] << std::endl;
		assert(test_res[i] == n_proc / N_PROC_PER_GROUP * (i - 1) + rank / N_PROC_PER_GROUP + 1);
	}
#endif

}


template <typename Dtype>
void AsyncWorker<Dtype>::AsyncCommLoop() {

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in AsyncCommLoop function");
#endif

	// last group need to initilize the ring based async computation
	if (async_comm_.config_.mpi_rank_ / N_PROC_PER_GROUP 
		== async_comm_.config_.n_group_ - 1) {
		// TODO Jian: adapt to solver buffer
		MPI_Datatype type = DtypeToMPIDtype<Dtype>::type;
		int64_t buf_size = async_comm_.mem_->GetBufSize();
		Dtype* init_buf = new Dtype [buf_size];
		memset(init_buf, 0, sizeof(Dtype) * buf_size);
		MPI_Comm* comm = async_comm_.GetMPIComm();
		MPI_Send(init_buf, buf_size, type, 0, ASYNC_MSG, *comm);
		delete init_buf;
	}

	// prevent trigger send overlaps with recv thread in wait on the same buf.
	this->async_comm_.ThreadBarrierWait();

	this->async_comm_.SendRecvLoop(this->solver_.n_iter_);

	// first group need to receive the final model from ring-based computation
	if (async_comm_.config_.mpi_rank_ / N_PROC_PER_GROUP == 0) {
		// TODO Jian get the final result our for output
		MPI_Status recv_status;
		MPI_Datatype type = DtypeToMPIDtype<Dtype>::type;
		MPI_Comm* comm = async_comm_.GetMPIComm();
		int64_t buf_size = async_comm_.mem_->GetBufSize();
		Dtype* finalize_buf = new Dtype[buf_size]; 
		MPI_Recv(finalize_buf, buf_size, type,
			async_comm_.config_.n_group_ - 1, ASYNC_MSG, *comm, &recv_status);
		delete finalize_buf;
	}
}


template <typename Dtype>
void AsyncWorker<Dtype>::Run() {

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in run function");
#endif

	// spawn data loading thread 
	std::thread data_load_thread(&AsyncWorker<Dtype>::LoadDataLoop, this);

	// spawn computing and group-sync thread
	std::thread compute_sync_thread(&AsyncWorker<Dtype>::AsyncComputeLoop, this);

	// clique root will handle communication among async groups 
	if (this->sync_comm_.IsCliqueRoot() ) {
		// spawn async communication thread 
		std::thread async_comm_thread(&AsyncWorker<Dtype>::AsyncCommLoop, this);
		async_comm_thread.join();
	}
	compute_sync_thread.join();		
	data_load_thread.join();
}


template class AsyncWorker<float>;
template class AsyncWorker<double>;