#include <thread>
#include "cluster/worker.hpp"
#include "cluster/debug_utils.hpp"
#include "cluster/timer.hpp"
#include "caffe/parallel.hpp"


template <typename Dtype>
void Worker<Dtype>::Init(caffe::Solver<Dtype>* solver_template) {
	// // TODO Jian : get buffer size from solver, combining everything of the solver
	// int64_t buf_size = 50000000;
	// int64_t n_iter = 10;
	// solver setup for simulating the experiment
	// solver_ = new Solver<Dtype>(buf_size, n_iter);
	// solver_->Init(sync_comm_.config_.GetDeviceId() );
	// solver_->SetDiffBuf(&(sync_comm_.gpu_buf_) );
	
	Caffe::set_root_solver(false);
	solver_ = new caffe::Solver<Dtype>(solver_template->param(), solver_template);
	Caffe::set_root_solver(true);

	const vector<Blob<Dtype>*>& blobs =
      solver_->net()->learnable_params();
	// total_size is a function from caffe:parallel.cpp
	buf_size_ = total_size(blobs);
	CUDA_CHECK(cudaMalloc(&model_, sizeof(Dtype) * buf_size_) );
	CUDA_CHECK(cudaMalloc(&diff_, sizeof(Dtype) * buf_size_) );
	ReplaceSolverBuffer(blobs);
	CopySolverBuffer(blobs, solver_template->net()->learnable_params);

	// init sync_comm and attach to diff_ buffer on GPU
	sync_comm_.Init(buf_size, diff_);	
	pthread_barrier_init(&data_ready_, NULL, 2);
	// wait for initilization of other workers in the same process
	pthread_barrier_wait(sync_comm_.process_barrier_);
	// wait for MPI sync group to set up
	if (sync_comm_.config_.is_clique_root_)
		MPI_Barrier(*(sync_comm_.mpi_sync_comm_) );
}


template <typename Dtype>
void Worker<Dtype>::ReplaceSolverBuffer(vector<Blob<Dtype>*>& blobs) {
	Dtype* model_ptr = model_;
	Dtype* diff_ptr = diff_;
	for (int i = 0; i < blobs.size(); i++) {
		int size = blobs[i]->count();
		blobs[i]->data()->set_gpu_data(model_ptr);
		blobs[i]->diff()->set_gpu_data(diff_ptr);
		model_ptr += size;
		diff_ptr += size;
	}
}


template <typename Dtype>
void Worker<Dtype>::CopySolverBuffer(vector<Blob<Dtype>*> blobs_dest,
	vector<Blob<Dtype>*> blobs_src) {
	for (int i = 0; i < blobs.size(); i++) {
		int size = blobs_dest[i]->count();
		CHECK_EQ(blobs_dest[i]->count(), blobs_src[i]->count() );
		CUDA_CHECK(cudaMemcpy(blobs_dest[i]->data()->gpu_data(),
			blobs_src[i]->data()->gpu_data(), sizeof(Dtype) * size,
			cudaMemcpyDeviceToDevice) );
	}
}


template <typename Dtype>
void Worker<Dtype>::SyncComputeLoop() {
	// set device id as this is a function called in a new thread
	CUDA_CHECK(cudaSetDevice(sync_comm_.config_.device_id_) );

#ifdef DEBUG
	Timer timer;
#endif 

#ifdef TEST
	std::vector<Dtype> test_res;
#endif

	for (int i = 0; i < this->solver_->n_iter_; i++) {
		// b_data: wait until data loading is done
		pthread_barrier_wait(&(this->data_ready_) );

		// do computation
		// solver_->Compute();
		solver->SingleStep();

		/**
		 * do intra-group synchronization, the un-divided data is 
		 * in sync_comm_.gpu_buf_. solver.diff_ is hooked onto this buffer.
		 */
		sync_comm_.SyncGroup(true);
		
		// solver combines the diff and model
		// solver_->UpdateModelFromDiff();
		solver_->ApplyUpdate();

#ifdef TEST
	Dtype* host_buf = new Dtype[this->solver_->buf_size_];
	CUDA_CHECK(cudaMemcpy(host_buf, this->solver_->model_,
		sizeof(Dtype) * this->solver_->buf_size_, cudaMemcpyDeviceToHost) );
	test_res.push_back(host_buf[0] );
	for (int i = 0; i < this->solver_->buf_size_; i++)
		assert(host_buf[0] == host_buf[i] );
	delete[] host_buf;
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
	
	// set device id as this is a function called in a new thread
	CUDA_CHECK(cudaSetDevice(sync_comm_.config_.device_id_) );

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in LoadDataLoop function");
#endif

	for (int i = 0; i < solver_->n_iter_; i++) {
		// TODO Jian replace with real data loading
		usleep(300000);

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, "Data loading done!");
#endif
		
		pthread_barrier_wait(&data_ready_);

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, "Data loading done after!");
#endif

	}
}


template <typename Dtype>
void Worker<Dtype>::Run() {

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in run function");
#endif

	// inti first before spawn threads
	Init();

	// spawn data loading thread 
	std::thread data_load_thread(&Worker<Dtype>::LoadDataLoop, this);

	// spawn computing and group-sync thread
	std::thread compute_sync_thread(&Worker<Dtype>::SyncComputeLoop, this);

	data_load_thread.join();
	compute_sync_thread.join();
}


template class Worker<float>;
template class Worker<double>;


