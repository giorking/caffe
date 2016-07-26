#ifndef DEBUG_UTILS_H_
#define DEBUG_UTILS_H_

#include <vector>
#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include "caffe/util/device_alternate.hpp"

#ifndef DEBUG
#define DEBUG
#endif

using namespace std;

void GetGpuIds(vector<int>& gpu_ids);

template <typename Dtype>
void DisplayGpuArray(Dtype* GpuArray, int64_t n_entry_to_display, string& str);

// template void DisplayGpuArray<float>(float* GpuArray, int64_t n_entry_to_display, string& str);
// template void DisplayGpuArray<double>(double* GpuArray, int64_t n_entry_to_display, string& str);


// void GetGpuIds(vector<int>& gpu_ids) {
//   int n_gpus = 0;
//   CUDA_CHECK(cudaGetDeviceCount(&n_gpus) );
//   gpu_ids.clear();
//   for (int i = 0; i < n_gpus; i++)
//     gpu_ids.push_back(i);
//   return;
// }


// template <typename Dtype>
// void DisplayGpuArray(Dtype* GpuArray, int64_t n_entry_to_display, string& str) {
// 	Dtype *CpuArray = new Dtype [n_entry_to_display];
// 	CUDA_CHECK(cudaMemcpy(CpuArray, GpuArray, 
// 		sizeof(Dtype) * n_entry_to_display, cudaMemcpyDeviceToHost) );
// 	for (int i = 0; i < n_entry_to_display; i++)
// 		std::cout << str << " idx " << i << " val " << CpuArray[i] << std::endl; 
// 	delete CpuArray;
// }


#endif