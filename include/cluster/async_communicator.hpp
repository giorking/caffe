#ifndef ASYNC_COMMUNICATOR_H_
#define ASYNC_COMMUNICATOR_H_

#include <mpi.h>
#include <pthread.h>
#include <cassert>
#include "cluster/comm_utils.hpp"
#include "cluster/async_mem.hpp"


// forward declaration for communicator config
template <typename Dtype>
class AsyncCommunicator;

template <typename Dtype>
class Worker;

template <typename Dtype>
class AsyncWorker;


template <typename Dtype>
class AsyncCommConfig {
public:
	AsyncCommConfig() {
		int n_proc;
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
		MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
		assert(n_proc % N_PROC_PER_GROUP == 0);
		n_group_ = n_proc / N_PROC_PER_GROUP;
		group_id_ = mpi_rank_ / N_PROC_PER_GROUP;
	}
	AsyncCommConfig(const AsyncCommConfig<Dtype>& config) :
		mpi_rank_(config.mpi_rank_),
		mpi_async_rank_(config.mpi_async_rank_),
		n_group_(config.n_group_),
		group_id_(config.group_id_) {}

	// TODO Jian remove
	void PrintInfo() {
		std::cout << "check asyn init rank " << mpi_rank_ << std::endl;
	}

private:
	/* MPI global rank */
	int mpi_rank_;
	int mpi_async_rank_;
	int n_group_;
	int group_id_;

friend class AsyncCommunicator<Dtype>;
friend class AsyncWorker<Dtype>;
};


/**
 * async communicator for asynchronized model update in a round-robin fashion.
 * the order of operation goes in the following
 * 1. inter-group thread : receive from last to mpi async mem
 * (barrier)
 * 2. intra-group thread : add delta to mpi async mem
 * (barrier)
 * 3. intra-group thread : take from mpi async mem to compute
 * 		inter-group thread : send out async mem. The next receive 
 * 		operation directly follows the send. We need mutex to prevent 
 * 		the this receive overlaps with the intra-group operation to 
 * 		"take from mpi async mem to compute"
 *
 * For inter-group parameter update, we do in a round-robin fashion. 
 * With the protocal we described above, a machine will compute 
 * gradient on x_i and commit \Delta x_i to x_(i + g) if we have g groups. 
 * We assume each group has the same number of machines. For inter-group
 * communication, machine 0 in each group will communicate while machine 1
 * in each group communicate. This parallels the parameter communication.
 */
template <typename Dtype>
class AsyncCommunicator {
public:
	AsyncCommunicator(const AsyncCommConfig<Dtype>& config) :
	config_(config), mem_(NULL), mpi_async_comm_(NULL) {}
	AsyncCommunicator(const AsyncCommunicator<Dtype>& comm) :
		AsyncCommunicator<Dtype> (comm.config_) {}
	// ~AsyncCommunicator() {
	// 	if (mpi_async_comm_ != NULL)
	// 		MPI_Comm_free(mpi_async_comm_);
	// 	pthread_barrier_destroy(&thread_barrier_);
	// }
	void Init(bool is_clique_root);
	/** 
	 * as we need to free MPI_Comm which has to be before MPI_Finalize.
	 * If we use deconstructor. The MPI_Comm_free will be after MPI_Finalize.
	 */
	void Destroy();
	// // the communicator stops only after a full receive or send operation
	// bool SetStop();
	// bool CheckStop();
	/** we only attach and detach async memory. 
	 *  this class does not need to deal with memory 
	 *  allocation and delete.
	 */
	inline void AttachAsyncMem(AsyncMem<Dtype>* mem) { mem_ = mem; };
	inline void DetachAsyncMem() { mem_ = NULL; }
	/** we send model in a round robin fashion. Send to next and 
	 * then receive from the last group root.
	 */
	void SendRecvLoop(int n_iter);
	MPI_Comm* GetMPIComm() { return mpi_async_comm_; }
	AsyncMem<Dtype>* GetAsyncMem() { return mem_; }
	void ThreadBarrierWait() { pthread_barrier_wait(&thread_barrier_); }
private:
	AsyncCommConfig<Dtype> config_;
	AsyncMem<Dtype>* mem_;
	MPI_Comm* mpi_async_comm_;
	/**
	 * lock for terminating signal from outside thread. We use it
	 * to guarantee the completion of certain operations. currently
	 * not in use.
	 */
	// bool stop_;
	pthread_mutex_t stop_lock_;
	pthread_barrier_t thread_barrier_;

friend class Worker<Dtype>;
friend class AsyncWorker<Dtype>;
};


#endif // end of ASYNC_COMMUNICATOR_H_