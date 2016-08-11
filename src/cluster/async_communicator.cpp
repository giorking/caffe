#include "cluster/async_mem.hpp"
#include "cluster/async_communicator.hpp"
#include "cluster/debug_utils.hpp"
#include "cluster/timer.hpp"


template <typename Dtype>
void AsyncCommunicator<Dtype>::Init(bool is_clique_root) {
	// pthread_mutex_init(&stop_lock_, NULL);
	pthread_barrier_init(&thread_barrier_, NULL, 2);
	if (is_clique_root) {
		mpi_async_comm_ = new MPI_Comm;
		/**
		 * construct asynchronized "group" where each real synchronized group
		 * contributes one proc. 
		 */
		MPI_Comm_split(MPI_COMM_WORLD, config_.mpi_rank_ % N_PROC_PER_GROUP, 
			config_.mpi_rank_, mpi_async_comm_);
		MPI_Comm_rank(*mpi_async_comm_, &(config_.mpi_async_rank_) );
	}
}


template <typename Dtype>
void AsyncCommunicator<Dtype>::Destroy() {
	pthread_barrier_destroy(&thread_barrier_);
	if (mpi_async_comm_ != NULL)	
		MPI_Comm_free(mpi_async_comm_);
	mpi_async_comm_ = NULL;
	pthread_barrier_destroy(&thread_barrier_);
}



template <typename Dtype>
void AsyncCommunicator<Dtype>::SendRecvLoop(int n_iter) {
	MPI_Datatype type = DtypeToMPIDtype<Dtype>::type;
	int async_rank = config_.mpi_async_rank_;
	MPI_Status recv_status;

#ifdef DEBUG
	Timer timer;
#endif 

	for (int iter = 0; iter < n_iter; iter++) {

#ifdef DEBUG
	timer.start();
#endif 

		// Note config_.group_id_ + config_.n_group_ - 1 prevent result being -1 
		// pthread_mutex_lock(mpi_mutex_);
		MPI_Recv( (void*)mem_->buf_, mem_->buf_size_, type, 
			(config_.group_id_ + config_.n_group_ - 1) % config_.n_group_, 
			ASYNC_MSG, *mpi_async_comm_, &recv_status);
		// pthread_mutex_unlock(mpi_mutex_);

#ifdef DEBUG
	timer.stop();
	DEBUG_PRINT_TIME(timer.getElapsedTimeInMilliSec(), " COMM:  Receive in ");
#endif

		// b1: wait until recv finishes.		
		this->ThreadBarrierWait();

		// b2: wait for the other thread to finish update delta to mem_
		this->ThreadBarrierWait();

#ifdef DEBUG
	timer.start();
#endif 

		MPI_Send( (void*)mem_->buf_, mem_->buf_size_, type, 
			(config_.group_id_ + 1) % config_.n_group_, ASYNC_MSG, *mpi_async_comm_);

#ifdef DEBUG
	timer.stop();
	DEBUG_PRINT_TIME(timer.getElapsedTimeInMilliSec(), "Send in ");
#endif

		// b3: prevent MPI recv overlap with reading updated model for compute
		this->ThreadBarrierWait();
	}
}

/* explicit instantiation */
template class AsyncCommunicator<float>;
template class AsyncCommunicator<double>;