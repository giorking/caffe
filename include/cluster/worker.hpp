#ifndef WORKER_H_
#define WORKER_H_

/**
 * class worker is the base class.
 * Derived class list:
 * 
 * Worker: 
 * a single synchronized group for gradient computing
 * 
 * AsyncWorker: 
 * asynchonized training with multiple training groups
 * 
 * QueueWorkerSeqModel: 
 * Centralized FIFO queue based worker for FC and other layers
 * 
 * AsyncWorkerSeqModel: 
 * Derived from AsyncWorker. only work synchronously on conv layers. 
 *
 * The general working protocol:
 *
 * 1. (TODO Jian) Create a solver / net config from outside. 
 * Initilize it in new thread.
 * 
 * 2. Create communicator config from outside. 
 * Initialize it in the new thread.
 *
 * 3. using the configs to initialize solver, communicators and etc.
 * 
 */
template <typename Dtype>
class Worker {
public:
	Worker(SyncCommConfig<Dtype>& sync_comm_config) :
		sync_comm_(sync_comm_config) {};
	// ~Worker();
	/** 
	 * SyncComputeLoop takes care of the local computation,
	 * single-node multi-GPU communication and and multi-node
	 * single-sync-group communication. More specifically,
	 * except the local computation, gradient aggeragation
	 * is carried out by SyncComputeLoop.
	 */
	virtual void SyncComputeLoop(SyncCommunicator<Dtype>* sync_comm_,
		AsyncCommunicator<Dtype>* async_comm_) {};
	/**
	 * handle async communication in the ring fashion.
	 * for detail, refer to AsyncCommunicator.hpp
	 */
	virtual void AsyncCommLoop(AsyncCommunicator<Dtype>* async_comm_) {};
	/**
	 * We load data in background, the loading time is hidden
	 * in the computation time for last iteration.
	 */
	void LoadDataLoop() {};
	virtual void Run() {};

private:
	/* TODO add a net solver */
	SyncCommunicator<Dtype>* sync_comm_;

};


template <typename Dtype>
class AsyncWorker : public Worker {
public:
	AsyncWorker(SyncCommConfig<Dtype>& sync_comm_config_,
		AsyncCommConfig<Dtype>& async_comm_config_);
	~AsyncWorker() {
		if (async_mem_ != NULL) {
			delete async_mem_;
			async_mem_ = NULL;
		}
	}
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


#endif // end of WORKER_H