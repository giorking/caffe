#include <thread>
#include "boost/algorithm/string.hpp"
#include "cluster/worker.hpp"
#include "cluster/comm_utils.hpp"
#include "cluster/debug_utils.hpp"
#include "cluster/timer.hpp"
#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"


namespace caffe {

template <typename Dtype>
void Worker<Dtype>::Init(caffe::shared_ptr<caffe::Solver<Dtype> > root_solver) {
// I0818 10:43:46.997336 23632 solver.cpp:234] Iteration 0, loss = 2.35804
// I0818 10:43:46.997361 23632 solver.cpp:250]     Train net output #0: loss = 2.35804 (* 1 = 2.35804 loss)
// I0818 10:43:46.997380 23632 sgd_solver.cpp:106] Iteration 0, lr = 0.01
// I0818 10:43:49.477687 23632 solver.cpp:234] Iteration 100, loss = 0.22289
// I0818 10:43:49.477715 23632 solver.cpp:250]     Train net output #0: loss = 0.22289 (* 1 = 0.22289 loss)
// I0818 10:43:49.477721 23632 sgd_solver.cpp:106] Iteration 100, lr = 0.00992565
// I0818 10:43:51.907272 23632 solver.cpp:234] Iteration 200, loss = 0.152105
// I0818 10:43:51.907297 23632 solver.cpp:250]     Train net output #0: loss = 0.152105 (* 1 = 0.152105 loss)
// I0818 10:43:51.907301 23632 sgd_solver.cpp:106] Iteration 200, lr = 0.00985258
// I0818 10:43:54.341142 23632 solver.cpp:234] Iteration 300, loss = 0.178918
// I0818 10:43:54.341166 23632 solver.cpp:250]     Train net output #0: loss = 0.178918 (* 1 = 0.178918 loss)
// I0818 10:43:54.341171 23632 sgd_solver.cpp:106] Iteration 300, lr = 0.00978075
// I0818 10:43:56.780573 23632 solver.cpp:234] Iteration 400, loss = 0.0830779
// I0818 10:43:56.780596 23632 solver.cpp:250]     Train net output #0: loss = 0.0830779 (* 1 = 0.0830779 loss)
// I0818 10:43:56.780602 23632 sgd_solver.cpp:106] Iteration 400, lr = 0.00971013

	Caffe::SetDevice(sync_comm_.config_.device_id_);
  Caffe::set_mode(Caffe::GPU);
  Caffe::set_solver_count(nDevicePerProc);
  caffe::Solver<Dtype>* root_solver_ptr = root_solver.get();
  caffe::Caffe::SetRootSolverPtr( (void*)root_solver_ptr);
  CUDA_CHECK(cudaSetDevice(sync_comm_.config_.device_id_) );


	pthread_mutex_lock(&globalInitMutex);
	if (sync_comm_.config_.clique_rank_ == sync_comm_.config_.clique_root_rank_)
		solver_ = root_solver.get();
	else {
	  SolverParameter param(root_solver->param() );
		param.set_device_id(sync_comm_.config_.device_id_);
		caffe::Caffe::set_root_solver(false);
		solver_ = caffe::SolverRegistry<Dtype>::CreateSolver(param);
		caffe::Caffe::set_root_solver(true);
	}
	pthread_mutex_unlock(&globalInitMutex);

	const std::vector<caffe::Blob<Dtype>*>& blobs_dest =
      solver_->net()->learnable_params();
  const std::vector<caffe::Blob<Dtype>*>& blobs_src =
      root_solver->net()->learnable_params();    
	// // get model size from external blob list and update buf_size_
	buf_size_ = GetModelSize(blobs_src);
	CUDA_CHECK(cudaMalloc(&model_, sizeof(Dtype) * buf_size_) );
	CUDA_CHECK(cudaMalloc(&diff_, sizeof(Dtype) * buf_size_) );
	CopySolverBuffer(blobs_src);
	ReplaceSolverBuffer(blobs_dest);

	// // TODO Jian make this resume code work
	// // if resume from previous snapshot
	// if (FLAGS_snapshot.size()) {
 //    LOG(INFO) << "Resuming from " << ::FLAGS_snapshot;
 //    solver_->Restore(::FLAGS_snapshot.c_str());
 //  } else if (::FLAGS_weights.size()) {
 //    CopyPretrainLayers(::FLAGS_weights);
 //  }

	// init sync_comm and attach to diff_ buffer on GPU
	sync_comm_.Init(buf_size_, diff_);	
	pthread_barrier_init(&data_ready_, NULL, 2);
	// wait for initilization of other workers in the same process
	pthread_barrier_wait(sync_comm_.process_barrier_);
	// wait for MPI sync group to set up
	if (sync_comm_.config_.is_clique_root_)
		MPI_Barrier(*(sync_comm_.mpi_sync_comm_) );
}


template <typename Dtype>
void Worker<Dtype>::ReplaceSolverBuffer(const std::vector<caffe::Blob<Dtype>*>& blobs) {
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
void Worker<Dtype>::CopyPretrainLayers(const std::string& model_list) {
	std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver_->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver_->test_nets().size(); ++j) {
      solver_->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}


template <typename Dtype>
void Worker<Dtype>::CopySolverBuffer(const std::vector<caffe::Blob<Dtype>*>& blobs_src) {
	Dtype* ptr = model_;
	int64_t model_size = 0;
	for (int i = 0; i < blobs_src.size(); i++) {
		int size = blobs_src[i]->count();
		// CHECK_EQ(blobs_dest[i]->count(), blobs_src[i]->count() );
		CUDA_CHECK(cudaMemcpy(ptr, blobs_src[i]->data()->gpu_data(), 
			sizeof(Dtype) * size, cudaMemcpyDeviceToDevice) );
		ptr += size;
		model_size += size;
	}
	if (model_size != buf_size_) {
		std::cout << "copying solver buffer: src size not equal to dest size " << std::endl;
		exit(1);
	}
}


// Buffer size necessary to store given blobs
template<typename Dtype>
int64_t Worker<Dtype>::GetModelSize(const std::vector<caffe::Blob<Dtype>*>& blobs) {
  int64_t size = 0;
  for (int i = 0; i < blobs.size(); ++i)
    size += blobs[i]->count();
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters.
  buf_size_ = size;
  if (buf_size_ == 0) {
  	std::cout << "The given model blob list cost 0 memory!" << std::endl;
  	exit(1);
  }
  return size;
}


template <typename Dtype>
void Worker<Dtype>::SyncComputeLoop() {

	caffe::Caffe::SetDevice(sync_comm_.config_.device_id_);
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::set_solver_count(nDevicePerProc);

	// set device id as this is a function called in a new thread
	CUDA_CHECK(cudaSetDevice(sync_comm_.config_.device_id_) );


#ifdef DEBUG
	Timer timer;
#endif 

#ifdef TEST
	std::vector<Dtype> test_res;
#endif

	// prepare solver for single step looping
	solver_->PrepareStepLoop();

	for (int i = 0; i < this->solver_->param().max_iter(); i++) {

#ifdef HIDE_DATA_FEED 
		// b_data: wait until data loading is done
		pthread_barrier_wait(&(this->data_ready_) );
#endif

		// do computation
		// solver_->Compute();
		solver_->SingleStep();

		/**
		 * do intra-group synchronization, the un-divided data is 
		 * in sync_comm_.gpu_buf_. solver.diff_ is hooked onto this buffer.
		 */
		sync_comm_.SyncGroup(true);
		
		// solver combines the diff and model
		// solver_->UpdateModelFromDiff();
		solver_->ApplyUpdate();
		solver_->IncrementIter();


#ifdef TEST
	Dtype* host_buf = new Dtype[this->solver_->buf_size_];
	CUDA_CHECK(cudaMemcpy(host_buf, this->solver_->model_,
		sizeof(Dtype) * this->solver_->buf_size_, cudaMemcpyDeviceToHost) );
	test_res.push_back(host_buf[0] );
	for (int i = 0; i < this->solver_->buf_size_; i++)
		assert(host_buf[0] == host_buf[i] );
	delete[] host_buf;
#endif

		// std::cout << "rank " << sync_comm_.config_.mpi_rank_ << " round " 
		// 	<< i << " done " << std::endl;
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
	// set every time we have a new thread
	caffe::Caffe::SetDevice(sync_comm_.config_.device_id_);
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
  caffe::Caffe::set_solver_count(nDevicePerProc);
	// b_data: wait until data loading is done 
	// TODO Jian perform data loading 
	
	// set device id as this is a function called in a new thread
	CUDA_CHECK(cudaSetDevice(sync_comm_.config_.device_id_) );

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in LoadDataLoop function");
#endif

	for (int i = 0; i < solver_->param().max_iter(); i++) {
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
// void Worker<Dtype>::Run(const caffe::SolverParameter& solver_param) {
void Worker<Dtype>::Run(caffe::shared_ptr<caffe::Solver<Dtype> > root_solver) {

	Caffe::SetDevice(sync_comm_.config_.device_id_);
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
  Caffe::set_solver_count(nDevicePerProc);
  caffe::Solver<Dtype>* root_solver_ptr = root_solver.get();
  caffe::Caffe::SetRootSolverPtr( (void*)root_solver_ptr);
  CUDA_CHECK(cudaSetDevice(sync_comm_.config_.device_id_) );

#ifdef DEBUG
	DEBUG_PRINT_RANK(MPI_COMM_WORLD, " in run function");
#endif

	// inti first before spawn threads
	Init(root_solver);


#ifdef HIDE_DATA_FEED 
	// spawn data loading thread 
	std::thread data_load_thread(&Worker<Dtype>::LoadDataLoop, this);
#endif

	// spawn computing and group-sync thread
	std::thread compute_sync_thread(&Worker<Dtype>::SyncComputeLoop, this);


#ifdef HIDE_DATA_FEED
	data_load_thread.join();
#endif

	compute_sync_thread.join();
}


template class Worker<float>;
template class Worker<double>;

}



