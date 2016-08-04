#include "glog/logging.h"
#include "cluster/solver.hpp"

template <>
void Solver<float>::UpdateModelFromDiff() {
	// note here diff_ is Dtype**
	float scalar = 1 / (float)(N_PROC_PER_GROUP * N_DEVICE_PER_PROC);
	cublasSaxpy(cublas_handle_, buf_size_, &scalar, *diff_, 1, model_,1);
}

template <>
void Solver<double>::UpdateModelFromDiff() {
	double scalar = 1 / (float) (N_PROC_PER_GROUP * N_DEVICE_PER_PROC);
	cublasDaxpy(cublas_handle_, buf_size_, &scalar, *diff_, 1, model_,1);
}


template class Solver<float>;
template class Solver<double>;