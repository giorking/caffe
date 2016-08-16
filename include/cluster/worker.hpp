#ifndef WORKER_H_
#define WORKER_H_

#include "cluster/async_mem.hpp"
#include "cluster/sync_communicator.hpp"
#include "cluster/async_communicator.hpp"
// #include "cluster/solver.hpp"
#include "caffe/solver.hpp"


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
	Worker(const SyncCommConfig<Dtype>& sync_comm_config, 
		pthread_barrier_t* process_barrier) : 
		solver_(NULL), 
		sync_comm_(sync_comm_config, process_barrier),
		model_(NULL),
		diff_(NULL), 
		buf_size_(0) {}
	Worker(const Worker<Dtype>& worker) :
		Worker<Dtype> (worker.sync_comm_.config_, 
		worker.sync_comm_.process_barrier_) {}
	virtual ~Worker() {
		delete[] model_; 
		delete[] diff_;
		if (solver_ != NULL)
			delete solver_;
		pthread_barrier_destroy(&data_ready_); 
	}
	void Init(caffe::Solver<Dtype>* solver_template);
	void ReplaceSolverBuffer(const std::vector<caffe::Blob<Dtype>*>& blobs);
	void CopySolverBuffer(const std::vector<caffe::Blob<Dtype>*>& blobs_src);
	int64_t GetModelSize(const std::vector<caffe::Blob<Dtype>*>& blobs);
	/** 
	 * SyncComputeLoop takes care of the local computation,
	 * single-node multi-GPU communication and and multi-node
	 * single-sync-group communication. More specifically,
	 * except the local computation, gradient aggeragation
	 * is carried out by SyncComputeLoop.
	 * As we pass this function to new thread, 
	 * we pass ptr this to simulate conventional use in member functions.
	 */
	virtual void SyncComputeLoop();
	/**
	 * We load data in background, the loading time is hidden
	 * in the computation time for last iteration.
	 */
	void LoadDataLoop();
	virtual void Run(caffe::Solver<Dtype>* solver_template);

protected:
	// TODO Jian: add a real net solver 
	caffe::Solver<Dtype>* solver_;

	SyncCommunicator<Dtype> sync_comm_;

	// replace this barrier with a barrier from solver
	pthread_barrier_t data_ready_;

	// GPU buffers
	Dtype* model_;
	Dtype* diff_;
	int64_t buf_size_;

};



// // TODO Jian remove these instantiation
// template class Solver<float>;
// template class Solver<double>;


#endif // end of WORKER_H