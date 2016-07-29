#include "cluster/worker.hpp"




template <typename Dtype>
AsyncWorker<Dtype>::AsyncWorker(SyncCommConfig<Dtype>& sync_comm_config, 
	AsyncCommConfig<Dtype>& async_comm_config) : Worker(sync_comm_config),
	async_comm_(async_comm_config) {
	
	/* TODO Jian : Get the number after net / solver initialized */

	async_mem_ = new AsyncMem<Dtype>(2, 20000000);
	async_comm_.AttachAsyncMem(async_mem_);
}


template <typename Dtype>
void AsyncWorker<Dtype>::Run() {

}


template <typename Dtype>
void AsyncWorker<Dtype>::Run() {
	/* spawn data loading thread */
	std::thread data_load_thread(AsyncWorker<Dtype>::LoadDataLoop);

	/* spawn computing and group-sync thread*/
	std::thread compute_sync_thread(AsyncWorker<Dtype>::SyncComputeLoop, &sync_comm_);

	/* spawn async communication thread */
	std::thread async_comm_thread(AsyncWorker<Dtype>::AsyncCommLoop, 
		&sync_comm_, &async_comm_);
}

