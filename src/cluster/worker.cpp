#include "cluster/worker.hpp"


template <typename Dtype>
Worker<Dtype>::Worker(SyncCommConfig<Dtype>& sync_comm_config) : 
	sync_comm_(NULL) {

	// TODO Jian : get buffer size from solver
	int64_t buf_size = 20000000;
	solver_ = new Solver(buf_size);

	sync_comm_ = new SyncCommunicator(sync_comm_config, buf_size);
	/**
	 * two thread involved. provider: LoadDataLoop thread
	 * consumer: SyncComputeLoop thread
	 */
	pthread_barrier_init(&data_ready_, NULL, 2);
}


template <typename Dtype>
void Worker<Dtype>::SyncComputeLoop() {
	// b_data: wait until data loading is done
	for (int i = 0; i < this->solver_.n_iter_; i++) 
		pthread_barrier_wait(this->data_ready_);

}


template <typename Dtype>
void Worker<Dtype>::LoadDataLoop() {
	// b_data: wait until data loading is done 
	// TODO Jian perform data loading 
	for (int i = 0; i < solver_.n_iter_; i++)
		pthread_barrier_wait(this->data_ready_);
}


template <typename Dtype>
AsyncWorker<Dtype>::AsyncWorker(SyncCommConfig<Dtype>& sync_comm_config, 
	AsyncCommConfig<Dtype>& async_comm_config) : 
	async_comm_(NULL) {

	// TODO Jian : get buffer size from solver
	int64_t buf_size = 20000000;	

	async_mem_ = new AsyncMem<Dtype>(2, buf_size);
	async_comm_ = new AsyncCommunicator(async_comm_config);
	async_comm_.AttachAsyncMem(async_mem_);
}


template <typename Dtype>
void AsyncWorker<Dtype>::AsyncComputeLoop() {
	// wait for communicator setup
	this->async_comm_.ThreadBarrierWait();
	
	// prevent trigger send overlaps with recv thread in wait on the same buf. 
	this->async_comm_.ThreadBarrierWait();

#ifdef TEST
	std::vector<Dtype> test_res;
#endif

	for (int i = 0; i < this->solver_.n_iter_; i++) {
		// b1: wait until comm thread finish receive
		this->async_comm_.ThreadBarrierWait();

		// commit delta to mem_
		this->solver_.RecvModel();

		// b2: wait until finish update delta to mem_
		this->async_comm_.ThreadBarrierWait();

		// read mem_ for compute
		this->solver_.CommitModelDiff();

#ifdef TEST
		test_res.push_back(this->solver_.mem_[0] );
		for (int i = 0; i < this->solver_.buf_size_; i++)
			assert(this->solver_.mem_[0] == this->solver_.mem_[i] );
#endif

		// b3 : wait until finish read mem_ out before recv
		this->async_comm_.ThreadBarrierWait();

		// b_data: wait until data loading is done 
		pthread_barrier_wait(this->data_ready_);

		// do computation
		this->Solver->Compute();

		std::cout << "rank " << rank << " round " << i << " done " << std::endl;
	}

#ifdef TEST
	// verify the pattern of the result
	int n_proc;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
	assert(test_res[0] = rank / N_PROC_PER_GROUP + 1);
	std::cout << "rank " << rank << " value " << test_res[0] << std::endl;
	for (int i = 1; i < test_res.size(); i++) {
		std::cout << "rank " << rank << " value " << test_res[i] << std::endl;
		assert(test_res[i] - (test_res[i - 1] ) == n_proc / N_PROC_PER_GROUP);
	}
#endif

}


template <typename Dtype>
void AsyncWorker<Dtype>::AsyncCommLoop() {
	// last group need to initilize the ring based async computation
	if (async_comm_.config_.mpi_rank_ / N_PROC_PER_GROUP 
		== async_comm_.config_.n_group_ - 1) {
		// TODO Jian: adapt to solver buffer
		int64_t buf_size = async_comm_.mem_->GetBufSize();
		Dtype* init_buf = new Dtype [buf_size];
		memset(init_buf, 0, sizeof(Dtype) * buf_size);
		MPI_Comm comm = async_comm_->GetMPIComm();
		MPI_Send(init_buf, buf_size, type, 0, 0, *comm);
	}

	this->async_comm_.SendRecvLoop(this->solver_.n_iter_);

	// first group need to receive the final model from ring-based computation
	if (async_comm_.config_.mpi_rank_ / N_PROC_PER_GROUP == 0) {
		MPI_Status recv_status;
		MPI_Comm comm = async_comm_->GetMPIComm(); 
		MPI_Recv(init_buf, mem->GetBufSize(), type,
			async_comm_.config_.n_group - 1, ASYNC_MSG, *comm, &recv_status);
	}
}


template <typename Dtype>
void AsyncWorker<Dtype>::Run() {
	// spawn data loading thread 
	std::thread data_load_thread(&AsyncWorker<Dtype>::LoadDataLoop);

	// spawn computing and group-sync thread
	std::thread compute_sync_thread(&AsyncWorker<Dtype>::SyncComputeLoop);

	// clique root will handle communication among async groups 
	if (sync_comm_.config_.is_clique_root_)
		// spawn async communication thread 
		std::thread async_comm_thread(&AsyncWorker<Dtype>::AsyncCommLoop);

	data_load_thread.join();
	compute_sync_thread.join();
	if (sync_comm_.config_.is_clique_root_)
		async_comm_thread.join();
}



template class Worker<float>;
template class Worker<double>;
template class AsyncWorker<float>;
template class AsyncWorker<double>;

