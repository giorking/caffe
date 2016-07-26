#ifndef ASYNC_MEM_H_
#define ASYNC_MEM_H_

#include <stdint.h>
#include <pthread.h>
#include <list>
#include "cluster/debug_utils.hpp"
#include "caffe/util/device_alternate.hpp"



class Semaphore {
public:
	Semaphore(int count, int n_thread) : 
		count_(count), n_thread_(n_thread) {
		pthread_mutex_init(&lock_, NULL);
		wait_conds_ = new pthread_cond_t [n_thread_];
		for (int i = 0; i < n_thread_; i++)
			pthread_cond_init(&wait_conds_[i], NULL);
	}
	~Semaphore() {
		pthread_mutex_destroy(&lock_);
		if (wait_conds_ != NULL) {
			for (int i = 0; i < n_thread_; i++)
				pthread_cond_destroy(&wait_conds_[i] );
			delete wait_conds_;
		}
	}
	
	/* request to execute the task on current threads. pure virtual */
	virtual void Request() {};
	/* Release one resource for other threads to work on. pure virtual. */
	virtual void Release() {};

#ifdef DEBUG
	/* for debugging purpose. record the ops posted into job queue */
	std::vector<bool> debug_op_;
#endif

protected:
	/* count_ is the currently number of availble resources */
	int count_;
	/* the total number of threads involving this semaphore */
	int n_thread_;
	pthread_mutex_t lock_;
	pthread_cond_t* wait_conds_;
	std::list<int> job_queue_;

};


/**
 * Semephore for multi-thread access to the CPU memory for MPI async communication.
 * We initialize a condition variable for each thread to enable signaling first
 * to the prior task. We have one send thread and one receive thread. As long as
 * there is a send task waiting, execute it first. It is guaranteed there will be
 * at most one prior send task in the queue at any time. Read operations are just 
 * kept in the queue in the posting order.
 */
class SemAsyncMem : public Semaphore {
public:
	SemAsyncMem(int count, int n_thread) : Semaphore(count, n_thread) {}
	void UpdateJobQueue(bool prior, int thread_id);
	/**
	 * Request to work on resource
	 * @param prior     set to true. if this is a prior request.
	 * prior request is guaranteed to be executed before any non-prior ones.
	 * @param thread_id indicates which thread is requesting.
	 */
	void Request(bool prior, int thread_id);
	void Release();
};


template <typename Dtype>
class AsyncMem {
public:
	AsyncMem(int n_thread, int64_t buf_size) : 
		sem_(1, n_thread),
		buf_(NULL),
		buf_size_(buf_size)  {
		buf_ = new Dtype [buf_size];
	}
	~AsyncMem() {
		if (buf_ != NULL)
			delete buf_;
	}
	inline void Request(bool prior, int thread_id) { sem_.Request(prior, thread_id); }
	inline void Release() { sem_.Release(); }
	int64_t GetBufSize() { return buf_size_; }

#ifdef DEBUG
	vector<bool> GetDebugOp() { return sem_.debug_op_; }
	Dtype* GetBuf() {return buf_; }
#endif

private:
	/* Semaphore designed for AsyncMem */
	SemAsyncMem sem_;
	Dtype* buf_;
	int64_t buf_size_;
};

template class AsyncMem<float>;
template class AsyncMem<double>;


#endif // end of ASYNC_MEM_H_