#include "cluster/sync_communicator.hpp"
#include "cluster/async_communicator.hpp"
#include "cluster/async_mem.hpp"
#include "cluster/worker.hpp"

template <typename Dtype>
class AsyncWorker : public Worker<Dtype> {
public:
	AsyncWorker(const SyncCommConfig<Dtype>& sync_comm_config,
		pthread_barrier_t* process_barrier, 
		const AsyncCommConfig<Dtype>& async_comm_config,
		AsyncMem<Dtype>* async_mem) : 
		Worker<Dtype> (sync_comm_config, process_barrier),
		async_mem_(async_mem),
		async_comm_(async_comm_config) {}
	AsyncWorker(const AsyncWorker<Dtype>& worker) :
		AsyncWorker<Dtype> (worker.sync_comm_.config_, 
		worker.sync_comm_.process_barrier_, 
		worker.async_comm_.config_,
		worker.async_mem_) {}
	void Init();
	// commit model diff to async memory
	void CommitDiffToAsyncMem(Dtype* diff_buf);
	virtual void AsyncComputeLoop();
	/**
	 * handle async communication in the ring fashion.
	 * for detail, refer to AsyncCommunicator.hpp
	 */
	virtual void AsyncCommLoop();
	/** 
	 * run one step. involve the following:
	 * 1. gradient computation
	 * 2. gradient all reduce communication
	 * 3. asynchronously update groups in a ring-based fashion.
	 * The ring-based design helps keep workers computing 
	 * with no inter-group waiting theoretically. 
	 * Our design hide the asynchronized inter-group communication
	 * to computing while the centralized asynchronized training
	 */
	virtual void Run();

private:
	/* async mpi communicator in addition to the synchronized one */
	AsyncMem<Dtype>* async_mem_;
	AsyncCommunicator<Dtype> async_comm_;	

};