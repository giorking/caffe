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

		// DEBUG
		std::string str = "init color " + std::to_string(config_.mpi_rank_ % N_PROC_PER_GROUP);
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, -1, str);

		std::string str1 = "init rank " + std::to_string(config_.mpi_async_rank_);
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, -1, str1);

		std::string str2 = "init rank " + std::to_string(config_.mpi_async_rank_);
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, -1, str1);

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


// template <typename Dtype>
// bool AsyncCommunicator<Dtype>::SetStop() {
// 	pthread_mutex_lock(&stop_lock_);
// 	stop_ = true;
// 	pthread_mutex_unlock(&stop_lock_);
// }


// template <typename Dtype>
// bool AsyncCommunicator<Dtype>::CheckStop() {
// 	pthread_mutex_lock(&stop_lock_);
// 	bool to_stop = stop_;
// 	pthread_mutex_unlock(&stop_lock_);
// 	return to_stop;
// }


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

		// DEBUG
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, iter, " COMM: before receive");

		// std::string str_comm = " start receive from " + std::to_string( (config_.group_id_ + config_.n_group_ - 1) % config_.n_group_ );
		// DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, iter, str_comm);
		// DEBUG_PRINT_RANK_DEVICE_ID_ITER(*mpi_async_comm_, iter, str_comm);
		// 
		std::ostringstream address0;
  	address0 << (void const *)mpi_async_comm_;
		std::string s0 = " async communicator addr recv " + address0.str();
  	DEBUG_PRINT_RANK_DEVICE_ID(MPI_COMM_WORLD, s0);
  	DEBUG_PRINT_RANK_DEVICE_ID(*mpi_async_comm_, s0);

		// Note config_.group_id_ + config_.n_group_ - 1 prevent result being -1 
		pthread_mutex_lock(mpi_mutex_);
		MPI_Recv( (void*)mem_->buf_, mem_->buf_size_, type, 
			(config_.group_id_ + config_.n_group_ - 1) % config_.n_group_, 
			ASYNC_MSG, *mpi_async_comm_, &recv_status);
		pthread_mutex_unlock(mpi_mutex_);

		// DEBUG
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, iter, " COMM: after receive");


#ifdef DEBUG
	timer.stop();
	// DEBUG_PRINT_RANK(MPI_COMM_WORLD, "")
	DEBUG_PRINT_TIME(timer.getElapsedTimeInMilliSec(), " COMM:  Receive in ");
#endif


		// DEBUG
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, iter, " COMM:  before b1 wait");

		// b1: wait until recv finishes.		
		this->ThreadBarrierWait();

				// DEBUG
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, iter, " COMM: after b1 wait");
				// DEBUG
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, iter, " COMM: before b2 wait");

		// b2: wait for the other thread to finish update delta to mem_
		this->ThreadBarrierWait();

				// DEBUG
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, iter, " COMM: after b2 wait");


#ifdef DEBUG
	timer.start();
#endif 

			// DEBUG
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, iter, " COMM: before send");

		// std::string str_comm1 = " start send to " + std::to_string((config_.group_id_ + 1) % config_.n_group_);
		// DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, iter, str_comm1);
		// DEBUG_PRINT_RANK_DEVICE_ID_ITER(*mpi_async_comm_, iter, str_comm1);

		std::ostringstream address1;
  	address1 << (void const *)mpi_async_comm_;
		std::string s1 = " async communicator addr send " + address1.str();
  	DEBUG_PRINT_RANK_DEVICE_ID(MPI_COMM_WORLD, s1);
  	DEBUG_PRINT_RANK_DEVICE_ID(*mpi_async_comm_, s1);

  	pthread_mutex_lock(mpi_mutex_);
		MPI_Send( (void*)mem_->buf_, mem_->buf_size_, type, 
			(config_.group_id_ + 1) % config_.n_group_, ASYNC_MSG, *mpi_async_comm_);
		pthread_mutex_unlock(mpi_mutex_);

					// DEBUG
		DEBUG_PRINT_RANK_DEVICE_ID_ITER(MPI_COMM_WORLD, iter, " COMM: after send");


#ifdef DEBUG
	timer.stop();
	// DEBUG_PRINT_RANK(MPI_COMM_WORLD, "")
	DEBUG_PRINT_TIME(timer.getElapsedTimeInMilliSec(), "Send in ");
#endif


		// b3: prevent MPI recv overlap with reading updated model for compute
		this->ThreadBarrierWait();
	}
}

/* explicit instantiation */
template class AsyncCommunicator<float>;
template class AsyncCommunicator<double>;