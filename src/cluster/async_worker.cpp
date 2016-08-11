#include <thread>
#include "cluster/timer.hpp"
#include "cluster/async_worker.hpp"


template <typename Dtype>
void AsyncWorker<Dtype>::Init() {
	Worker<Dtype>::Init();
	// only the clique root deal with the asyn communication
	async_comm_.Init(this->sync_comm_.IsCliqueRoot() );
	async_comm_.AttachAsyncMem(async_mem_);
	if (this->sync_comm_.IsCliqueRoot() ) {
		// there will be 1 communication thread and computing thread around the mem
		async_mem_->Init(this->sync_comm_.mpi_sync_buf_size_, 2);
		// wait for MPI async group finish intialization
		MPI_Barrier(*(async_comm_.mpi_async_comm_) );
		
		// TODO Jian remove
		this->sync_comm_.mpi_mutex_ = &(this->debug_mutex_);
		this->async_comm_.mpi_mutex_ = &(this->debug_mutex_);
	}
}


template <typename Dtype>
void AsyncWorker<Dtype>::CommitDiffToAsyncMem(Dtype* diff_buf) {
	Dtype scale = N_PROC_PER_GROUP * N_DEVICE_PER_PROC;
	// TODO Jian: replace this with more efficient blas implementation
	for (int i = 0; i < async_mem_->buf_size_; i++)
		async_mem_->buf_[i] += diff_buf[i] / scale;
}


template <typename Dtype>
void AsyncWorker<Dtype>::AsyncComputeLoop() {

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in AsyncComputeLoop function");
#endif

	// set device id as this is a function called in a new thread
	CUDA_CHECK(cudaSetDevice(this->sync_comm_.config_.GetDeviceId() ) );

	// prevent trigger send overlaps with recv thread in wait on the same buf. 
	if (this->sync_comm_.IsCliqueRoot() )
		this->async_comm_.ThreadBarrierWait();

#ifdef TEST
	std::vector<Dtype> test_res;
#endif

#ifdef DEBUG
	Timer timer;
#endif 

	for (int i = 0; i < this->solver_->n_iter_; i++) {

#ifdef DEBUG
	timer.start();
#endif 

		if (this->sync_comm_.IsCliqueRoot() ) {
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

			// DEBUG
			DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, i, " COMP: finish commit diff to async mem");

			// b2: wait until finish update delta to mem_ before send
			this->async_comm_.ThreadBarrierWait();
		}

		// DEBUG
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, i, " COMP: start recv model buf size ");

		// read mem_ for compute after delta is committed on clique root worker
		// pthread_barrier_wait(this->sync_comm_.process_barrier_);
		this->sync_comm_.ProcessBarrierWait();

		// DEBUG
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, i, " COMP: start recv model after process barrier ");

		this->solver_->RecvModel(this->async_mem_->buf_, this->async_mem_->buf_size_);

// // DEBUG
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, i, " COMP: finish recv model");
		// std::cout << "finish recv model" << std::endl;


#ifdef TEST
		Dtype* host_buf = new Dtype[this->solver_->buf_size_];
		CUDA_CHECK(cudaMemcpy(host_buf, this->solver_->model_, 
			sizeof(Dtype) * this->solver_->buf_size_, cudaMemcpyDeviceToHost) );
		test_res.push_back(host_buf[0] );
		for (int i = 0; i < this->solver_->buf_size_; i++)
			assert(host_buf[0] == host_buf[i] );
		delete[] host_buf;
#endif


		// DEBUG
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, i, " COMP: after test");


		if (this->sync_comm_.IsCliqueRoot() ) {
			// DEBUG
			DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, i, " COMP: b3 wait before");			

			// b3 : wait until finish read mem_ out before recv
			this->async_comm_.ThreadBarrierWait();

			// DEBUG
			DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, i, " COMP: b3 wait after");
			usleep(1000000);

		}

			// DEBUG
			DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, i, " COMP: start compute model");

		usleep(1000000);

		// b_data: wait until data loading is done 
		pthread_barrier_wait(&(this->data_ready_) );

		// do computation
		this->solver_->Compute();

		// DEBUG
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, i, " COMP:  finish compute model");
				usleep(1000000);
		/**
		 * do intra-group synchronization, we do not do braodcast after inter-machine
		 * all reduce. As in async worker the new model will be copied from asyn_mem
		 * directly to gpu memory.
		 */
		this->sync_comm_.SyncGroup(false);

		// DEBUG
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, i, " COMP:  finish intragroup sync model");


		std::cout << "rank " << async_comm_.config_.mpi_rank_ << " round " 
			<< i << " done " << std::endl;
		
#ifdef DEBUG
	timer.stop();
	// DEBUG_PRINT_RANK(MPI_COMM_WORLD, "")
	DEBUG_PRINT_TIME(timer.getElapsedTimeInMilliSec(), " COMP:  Computing in ");
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

	// DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, "compute loop done ");

}


template <typename Dtype>
void AsyncWorker<Dtype>::AsyncCommLoop() {

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in AsyncCommLoop function");
#endif

	// set device id as this is a function called in a new thread
	CUDA_CHECK(cudaSetDevice(this->sync_comm_.config_.GetDeviceId() ) );

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

	this->async_comm_.SendRecvLoop(this->solver_->n_iter_);

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


	// DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, "comm loop done ");

}


template <typename Dtype>
void AsyncWorker<Dtype>::Run() {

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in run function");
#endif

	// // set device id as this is a function called in a new thread
	// CUDA_CHECK(cudaSetDevice(this->sync_comm_.config_.GetDeviceId() ) );

	// inti first before spawn threads
	Init();

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

	// 	// DEBUG
	// DEBUG_PRINT_RANK(MPI_COMM_WORLD, "worker thread 0 done");


	compute_sync_thread.join();		

	// // DEBUG
	// DEBUG_PRINT_RANK(MPI_COMM_WORLD, "worker thread 1 done");


	data_load_thread.join();

	// 	// DEBUG
	// DEBUG_PRINT_RANK(MPI_COMM_WORLD, "worker thread 2 done");
}


template class AsyncWorker<float>;
template class AsyncWorker<double>;