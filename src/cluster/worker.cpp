#include <thread>
#include "cluster/worker.hpp"
#include "cluster/debug_utils.hpp"
#include "cluster/timer.hpp"


template <typename Dtype>
Worker<Dtype>::Worker(const SyncCommConfig<Dtype>& sync_comm_config) : 
	solver_(20000000, 1), 
	sync_comm_(sync_comm_config, this->solver_.buf_size_) {
	// TODO Jian : get buffer size from solver, combining everything of the solver
	solver_.Init(sync_comm_.config_.GetDeviceId() );
	solver_.SetDiffBuf(&(sync_comm_.gpu_buf_) );	
	pthread_barrier_init(&data_ready_, NULL, 2);


	std::cout << "check comm ptr address " << sync_comm_.nccl_comm_ << std::endl;

}


template <typename Dtype>
Worker<Dtype>::Worker(const Worker<Dtype>& worker) :
	Worker<Dtype> (worker.sync_comm_.config_) {}


template <typename Dtype>
void Worker<Dtype>::SyncComputeLoop() {

#ifdef DEBUG
	Timer timer;
#endif 

#ifdef TEST
	std::vector<Dtype> test_res;
#endif

	for (int i = 0; i < this->solver_.n_iter_; i++) {

#ifdef DEBUG
	timer.start();
#endif 

		// b_data: wait until data loading is done
		pthread_barrier_wait(&(this->data_ready_) );

		// // do computation
		// solver_.Compute();

// #ifdef DEBUG
// 	timer.stop();
// 	DEBUG_PRINT_TIME(timer.getElapsedTimeInMilliSec(), "Compute in ");
// 	timer.start();
// #endif

		/**
		 * do intra-group synchronization, the un-divided data is 
		 * in sync_comm_.gpu_buf_. solver.diff_ is hooked onto this buffer.
		 */
		
		// TODO recover syncgroup
		// sync_comm_.SyncGroup(true);
		for (int j = 0; j < 100; j++) {
			std::cout << "simulate reduce round " << j << " start " << std::endl;
			usleep(sync_comm_.config_.device_id_ * 1000000);
			sync_comm_.CliqueReduce();
			std::cout << "simulate reduce round " << j << " finish " << std::endl;
		}
		

		std::cout << "check sync group done" << std::endl;

		// // solver combines the diff and model
		// solver_.UpdateModelFromDiff();


		// std::cout << "check update model done " << std::endl;

// #ifdef TEST
// 	Dtype* host_buf = new Dtype[this->solver_.buf_size_];
// 	CUDA_CHECK(cudaMemcpy(host_buf, this->solver_.model_,
// 		sizeof(Dtype) * this->solver_.buf_size_, cudaMemcpyDeviceToHost) );
// 	test_res.push_back(host_buf[0] );
// 	for (int i = 0; i < this->solver_.buf_size_; i++)
// 		assert(host_buf[0] == host_buf[i] );
// 	delete[] host_buf;
// #endif

#ifdef DEBUG
	timer.stop();
	DEBUG_PRINT_TIME(timer.getElapsedTimeInMilliSec(), "Sync in ");
#endif

		std::cout << "rank " << sync_comm_.config_.mpi_rank_ << " round " 
			<< i << " done " << std::endl;
	}

#ifdef TEST
	// verify the pattern of the result
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	for (int i = 0; i < test_res.size(); i++) {
		std::cout << "rank " << rank << " value " << test_res[i] << std::endl;
		assert(test_res[i] == i + 1);
	}
#endif

}


template <typename Dtype>
void Worker<Dtype>::LoadDataLoop() {
	// b_data: wait until data loading is done 
	// TODO Jian perform data loading 
	
#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in LoadDataLoop function");
#endif

	for (int i = 0; i < solver_.n_iter_; i++) {
		// TODO Jian replace with real data loading
		usleep(300000);

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, "Data loading done!");
#endif
		
		pthread_barrier_wait(&data_ready_);

	}
}


template <typename Dtype>
void Worker<Dtype>::Run(ncclComm_t* comm) {

	std::cout << "in run func " << sync_comm_.config_.device_id_ << std::endl;

	int buf_size = 1000;
	CUDA_CHECK(cudaSetDevice(sync_comm_.config_.device_id_) );
	float* device_buf;
	float* host_buf = (float*)malloc(sizeof(float) * buf_size);
	CUDA_CHECK(cudaMalloc(&device_buf, sizeof(float) * buf_size) );
	cudaStream_t* stream_comm = (cudaStream_t*)malloc(sizeof(cudaStream_t) );
  CUDA_CHECK(cudaStreamCreate(stream_comm) );

  std::cout << "create done " << std::endl;

  std::cout << "compare ptr address " << comm << " " << sync_comm_.nccl_comm_ << std::endl;

   for (int i = 0; i < 100; i++) {
  	std::cout << "dev " << sync_comm_.config_.device_id_ << " iter " << i << " start reduce" << std::endl;
  	usleep(sync_comm_.config_.device_id_ * 10000000);
	  NCCL_CHECK(ncclReduce( (const void*)device_buf, (void*)device_buf, 
	  	buf_size, ncclFloat, ncclSum, 0, 
	  	*(sync_comm_.nccl_comm_), *stream_comm) );
	  std::cout << "dev " << sync_comm_.config_.device_id_ << " iter " << i << " start waiting" << std::endl;
	  cudaStreamSynchronize(*stream_comm);
	}




// 			for (int j = 0; j < 100; j++) {
// 			std::cout << "simulate reduce round " << j << " start " << std::endl;
// 			usleep(sync_comm_.config_.device_id_ * 1000000);
// 			sync_comm_.CliqueReduce();
// 			std::cout << "simulate reduce round " << j << " finish " << std::endl;
// 		}

// 	std::cout << "test done before real run" << std::endl;


// #ifdef DEBUG
// 	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in run function");
// #endif
// 	// spawn data loading thread 
// 	std::thread data_load_thread(&Worker<Dtype>::LoadDataLoop, this);

// 	// spawn computing and group-sync thread
// 	std::thread compute_sync_thread(&Worker<Dtype>::SyncComputeLoop, this);

// 	data_load_thread.join();
// 	compute_sync_thread.join();
}


template class Worker<float>;
template class Worker<double>;


