// #include <pthread.h>
// #include "cluster/debug_utils.hpp"
#include "cluster/async_mem.hpp"

template <typename Dtype>
void AsyncMem<Dtype>::Init(int64_t buf_size, int n_thread) {
	buf_size_ = buf_size;
	buf_ = new Dtype [buf_size_];
	memset(buf_, 0, sizeof(Dtype) * buf_size_);
	pthread_mutex_init(&access_lock_, NULL);
	pthread_barrier_init(&order_ctrl_, NULL, n_thread_);
}
// void SemAsyncMem::UpdateJobQueue(bool prior, int thread_id) {
// 	/**
// 	 * as we only have one thread post prior send task, 
// 	 * there will be at most one prior task at any time point.
// 	 */
// 	if (prior)
// 		job_queue_.push_front(thread_id);
// 	/* read operations are just queued in order */
// 	else
// 		job_queue_.push_back(thread_id);
// }


// void SemAsyncMem::Request(bool prior, int thread_id) {
// 	pthread_mutex_lock(&lock_);

// #ifdef DEBUG
// 		debug_op_.push_back(prior);
// #endif

// 	count_--;
// 	if (count_ < 0) {
// 		UpdateJobQueue(prior, thread_id);
// 		pthread_cond_wait(&wait_conds_[thread_id], &lock_);
// 	}
// 	pthread_mutex_unlock(&lock_);
// }


// void SemAsyncMem::Release() {
// 	pthread_mutex_lock(&lock_);
// 	count_++;
// 	if (count_ <= 0) {
// 		int thread_id = job_queue_.front();
// 		job_queue_.pop_front();
// 		pthread_cond_signal(&wait_conds_[thread_id] );
// 	}
// 	pthread_mutex_unlock(&lock_);
// }

/* explicit instantiation */
template class AsyncMem<float>;
template class AsyncMem<double>;

