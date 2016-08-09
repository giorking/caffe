#ifndef SOLVER_H_
#define SOLVER_H_

#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cluster/comm_utils.hpp"
#include "cluster/debug_utils.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/device_alternate.hpp"


// forward declaration
template <typename Dtype>
class Worker;

template <typename Dtype>
class AsyncWorker;


// a solver for debugging before connect to caffe
template <typename Dtype>
class Solver {
public:
	Solver() : model_(NULL), diff_(NULL) { cublasCreate(&cublas_handle_); };
	Solver(int64_t buf_size, int n_iter) : model_(NULL), diff_(NULL) {
		n_iter_ = n_iter;
		buf_size_ = buf_size;
		cublasCreate(&cublas_handle_);

		// model_ = new Dtype[buf_size_];
		// std::cout << " model allocation before " << model_ << std::endl;
		// CUDA_CHECK(cudaMalloc(&model_, sizeof(Dtype) * buf_size_) );
		// CUDA_CHECK(cudaMemset(model_, 0, sizeof(Dtype) * buf_size_) );

		// cublasCreate(&cublas_handle_);
	}
	void Init(int device_id) {
		CUDA_CHECK(cudaSetDevice(device_id) );
		CUDA_CHECK(cudaMalloc(&model_, sizeof(Dtype) * buf_size_) );
		CUDA_CHECK(cudaMemset(model_, 0, sizeof(Dtype) * buf_size_) );
	}
	Solver(const Solver<Dtype>& solver) :
		Solver(solver.buf_size_, solver.n_iter_) {}
	~Solver() {
		if (model_ != NULL)
			CUDA_CHECK(cudaFree(model_) );
		cublasDestroy(cublas_handle_);
	}
	void Compute() {
		usleep(500000);
		Dtype* host_buf = new Dtype[buf_size_];
		for (int i = 0; i < buf_size_; i++)
			host_buf[i] = 1.0;
		CUDA_CHECK(cudaMemcpy(*diff_, host_buf, sizeof(Dtype) * buf_size_, cudaMemcpyHostToDevice) );
		delete[] host_buf;
	}
	void RecvModel(Dtype* buf, int64_t buf_size) {
		// TODO Jian assert buffer size is the same
		// memcpy(model_, buf, sizeof(Dtype) * buf_size_);

		CUDA_CHECK(cudaMemcpy(model_, buf, sizeof(Dtype) * buf_size_, cudaMemcpyHostToDevice) );
	
	}
	void CommitModelDiff(Dtype* buf, int64_t buf_size) {
		// TODO Jian assert buffer size is the same
		// commit delta to model_
		for (int i = 0; i < buf_size_; i++)
			// +1 for test
			buf[i] += 1;
	}
	void SetDiffBuf(Dtype** diff_buf) { diff_ = diff_buf; };
	void UpdateModelFromDiff();
private:
	Dtype* model_;
	Dtype** diff_;
	int64_t buf_size_;
	int n_iter_;
	cublasHandle_t cublas_handle_;

friend class Worker<Dtype>;
friend class AsyncWorker<Dtype>;
};


#endif  // end of SOLVER_H_