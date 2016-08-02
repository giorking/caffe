#include <thread>
#include "cluster/worker.hpp"
#include "cluster/debug_utils.hpp"


template <typename Dtype>
Worker<Dtype>::Worker(const SyncCommConfig<Dtype>& sync_comm_config) : 
	solver_(20000000, 5), 
	sync_comm_(sync_comm_config, this->solver_.buf_size_) {
	// TODO Jian : get buffer size from solver, combining everything of the solver
	solver_.SetDiffBuf(sync_comm_.gpu_buf_);	
	pthread_barrier_init(&data_ready_, NULL, 2);
}

template <typename Dtype>
Worker<Dtype>::Worker(const Worker<Dtype>& worker) : 
	solver_(worker.solver_),
	sync_comm_(worker.sync_comm_) {
	pthread_barrier_init(&data_ready_, NULL, 2);
}


template <typename Dtype>
void Worker<Dtype>::SyncComputeLoop() {
	// b_data: wait until data loading is done
	for (int i = 0; i < this->solver_.n_iter_; i++) 
		pthread_barrier_wait(&(this->data_ready_) );
}


template <typename Dtype>
void Worker<Dtype>::LoadDataLoop() {
	// b_data: wait until data loading is done 
	// TODO Jian perform data loading 
	
#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in LoadDataLoop function");
#endif

	for (int i = 0; i < solver_.n_iter_; i++) {

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " dataload done");
#endif
		
		pthread_barrier_wait(&data_ready_);

	}
}


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
	for (int i = 0; i < async_mem_.buf_size_; i++)
		async_mem_.buf_[i] += diff_buf[i] / scale;
}


template <typename Dtype>
void AsyncWorker<Dtype>::AsyncComputeLoop() {

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in AsyncComputeLoop function\n");
#endif

	// prevent trigger send overlaps with recv thread in wait on the same buf. 
	this->async_comm_.ThreadBarrierWait();

	int test_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &test_rank);

#ifdef TEST
	std::vector<Dtype> test_res;
#endif

	for (int i = 0; i < this->solver_.n_iter_; i++) {

		// if (test_rank == 0)
		// 	DEBUG_PRINT("rank 0 compute step 1\n");
		// else
		// 	DEBUG_PRINT("rank 1 compute step 1\n");

		// b1: wait until comm thread finish receive
		this->async_comm_.ThreadBarrierWait();

		// if (test_rank == 0)
		// 	DEBUG_PRINT("rank 0 compute step 2\n");
		// else
		// 	DEBUG_PRINT("rank 1 compute step 2\n");

		// commit delta to mem_
		this->CommitDiffToAsyncMem(this->sync_comm_.GetMpiSyncBuffer() );
		// this->solver_.CommitModelDiff(this->async_mem_.buf_, this->async_mem_.buf_size_);


		// if (test_rank == 0)
		// 	DEBUG_PRINT("rank 0 compute step 3\n");
		// else
		// 	DEBUG_PRINT("rank 1 compute step 3\n");

		// b2: wait until finish update delta to mem_
		this->async_comm_.ThreadBarrierWait();

		// if (test_rank == 0)
		// 	DEBUG_PRINT("rank 0 compute step 4\n");
		// else
		// 	DEBUG_PRINT("rank 1 compute step 4\n");

		// read mem_ for compute
		this->solver_.RecvModel(this->async_mem_.buf_, this->async_mem_.buf_size_);

		// if (test_rank == 0)
		// 	DEBUG_PRINT("rank 0 compute step 5\n");
		// else
		// 	DEBUG_PRINT("rank 1 compute step 5\n");

#ifdef TEST
		test_res.push_back(this->solver_.model_[0] );
		for (int i = 0; i < this->solver_.buf_size_; i++)
			assert(this->solver_.model_[0] == this->solver_.model_[i] );
#endif

		// b3 : wait until finish read mem_ out before recv
		this->async_comm_.ThreadBarrierWait();

		// b_data: wait until data loading is done 
		pthread_barrier_wait(&(this->data_ready_) );

		// do computation
		this->solver_.Compute();

		// do intra-group synchronization
		this->sync_comm_.SyncGroup();


		std::cout << "rank " << async_comm_.config_.mpi_rank_ << " round " 
			<< i << " done " << std::endl;
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
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in AsyncCommLoop function\n");
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
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in run function\n");
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



template class Worker<float>;
template class Worker<double>;
template class AsyncWorker<float>;
template class AsyncWorker<double>;

