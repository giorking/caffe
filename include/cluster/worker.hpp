#ifndef WORKER_H_
#define WORKER_H_

#include "cluster/async_mem.hpp"
#include "cluster/sync_communicator.hpp"
#include "cluster/async_communicator.hpp"
// #include "cluster/solver.hpp"
#include "cluster/comm_utils.hpp"
#include "caffe/caffe.hpp"

namespace caffe {

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

#ifdef GPU_DIRECT_MPI
	Worker(const SyncCommConfig<Dtype>& sync_comm_config, 
		pthread_barrier_t* process_barrier) : 
		solver_(NULL), 
		sync_comm_(sync_comm_config, process_barrier),
		model_(NULL),
		diff_(NULL), 
		buf_size_(0) {}
#else
	Worker(const SyncCommConfig<Dtype>& sync_comm_config, 
		pthread_barrier_t* process_barrier) : 
		solver_(NULL), 
		sync_comm_(sync_comm_config, process_barrier),
		model_(NULL),
		diff_(NULL), 
		buf_size_(0),
		cpu_mpi_buf_(NULL) {}
#endif

	Worker(const Worker<Dtype>& worker) :
		Worker<Dtype> (worker.sync_comm_.config_, 
		worker.sync_comm_.process_barrier_) {}
	virtual ~Worker() {
		CUDA_CHECK(cudaFree(model_) ); 
		CUDA_CHECK(cudaFree(diff_) );

#ifndef GPU_DIRECT_MPI
		delete[] cpu_mpi_buf_;		
#endif

		// if clique root the solver is attached and it is create from outside
		if (sync_comm_.config_.clique_rank_ == 
			sync_comm_.config_.clique_root_rank_ && solver_ != NULL)
			delete solver_;
		pthread_barrier_destroy(&data_ready_); 
	}
	// void Init(const caffe::SolverParameter& solver_param);
	// void Init(caffe::Solver<Dtype>* root_solver);
	void Init(caffe::shared_ptr<caffe::Solver<Dtype> > root_solver);
	void ReplaceSolverBuffer(const std::vector<caffe::Blob<Dtype>*>& blobs);
	void CopyPretrainLayers(const std::string& model_list);
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
	// virtual void Run(const caffe::SolverParameter& solver_param);
	// virtual void Run(caffe::Solver<Dtype>* root_solver);
	virtual void Run(caffe::shared_ptr<caffe::Solver<Dtype> > root_solver);


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

#ifndef GPU_DIRECT_MPI
	Dtype* cpu_mpi_buf_;
#endif

};

// setup sync workers 
template <typename Dtype>
void RunSyncWorkers(caffe::shared_ptr<caffe::Solver<Dtype> > root_solver);

}

#endif // end of WORKER_H