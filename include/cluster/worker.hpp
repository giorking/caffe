#ifndef WORKER_H_
#define WORKER_H_

#include "cluster/async_mem.hpp"
#include "cluster/sync_communicator.hpp"
#include "cluster/async_communicator.hpp"


// a solver for debugging before connect to caffe
template <typename Dtype>
class Solver {
public:
	Solver(int64_t buf_size, int n_iter) {
		n_iter_ = n_iter;
		buf_size_ = buf_size;
		Dtype* mem_ = new Dtype[buf_size_];
	}
	~Solver() {
		if (mem_ != NULL)
			delete mem_;
	}
	void Compute() {
		usleep(50000000);
	}
	void RecvModel(Dtype* buf) {
		memcpy(mem_, buf, sizeof(Dtype) * buf_size_);
	}
	void CommitModelDiff(Dtype* buf) {
		// commit delta to mem_
		for (int i = 0; i < buf_size_; i++)
			// +1 for test
			buf[i] += 1;
	}
private:
	Dtype* mem_;
	int64_t buf_size_;
	int n_iter_;

friend class Worker<Dtype>;
friend class AsyncWorker<Dtype>;
};


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
	Worker(SyncCommConfig<Dtype>& sync_comm_config);
	~Worker() {
		pthread_barrier_destroy(&data_ready_);
		if (sync_comm_ != NULL)
			delete sync_comm_;
		if (solver_ != NULL)
			delete solver_;
	};
	/** 
	 * SyncComputeLoop takes care of the local computation,
	 * single-node multi-GPU communication and and multi-node
	 * single-sync-group communication. More specifically,
	 * except the local computation, gradient aggeragation
	 * is carried out by SyncComputeLoop.
	 * As we pass this function to new thread, 
	 * we pass ptr this to simulate conventional use in member functions.
	 */
	virtual void ComputeGradient();
	virtual void SyncComputeLoop();
	/**
	 * We load data in background, the loading time is hidden
	 * in the computation time for last iteration.
	 */
	void LoadDataLoop();
	virtual void Run() {}

private:
	/* TODO Jian: add a real net solver */
	Solver<Dtype>* solver_;

	SyncCommunicator<Dtype>* sync_comm_;

	/* replace this barrier with a barrier from solver*/
	pthread_barrier_t data_ready_;
};


template <typename Dtype>
class AsyncWorker : public Worker<Dtype> {
public:
	AsyncWorker(SyncCommConfig<Dtype>& sync_comm_config_,
		AsyncCommConfig<Dtype>& async_comm_config_);
	~AsyncWorker() {
		if (async_mem_ != NULL) {
			delete async_mem_;
			async_mem_ = NULL;
		}
		if (async_comm_ != NULL)
			delete async_comm_;
	}
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
	AsyncCommunicator<Dtype>* async_comm_;	

};



// TODO Jian remove these instantiation
template class Solver<float>;
template class Solver<double>;


#endif // end of WORKER_H