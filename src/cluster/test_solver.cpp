#include "glog/logging.h"
#include "cluster/solver.hpp"
#include "cluster/comm_utils.hpp"

template <>
void Solver<float>::UpdateModelFromDiff() {
	// note here diff_ is Dtype**
	float scalar = 1 / (float)(nProcPerGroup * nDevicePerProc);
	cublasSaxpy(cublas_handle_, buf_size_, &scalar, *diff_, 1, model_,1);
}

template <>
void Solver<double>::UpdateModelFromDiff() {
	double scalar = 1 / (float) (nProcPerGroup * nDevicePerProc);
	cublasDaxpy(cublas_handle_, buf_size_, &scalar, *diff_, 1, model_,1);
}


template class Solver<float>;
template class Solver<double>;