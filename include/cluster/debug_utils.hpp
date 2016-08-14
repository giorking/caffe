#ifndef DEBUG_UTILS_H_
#define DEBUG_UTILS_H_

#include <mpi.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
// #include "caffe/util/device_alternate.hpp"

// #ifndef DEBUG
// #define DEBUG
// #endif

// #ifndef TEST
// #define TEST
// #endif

#ifndef TIMER
#define TIMER
#endif


#define DEBUG_PRINT(content) do { \
	int debug_mpi_rank; \
	MPI_Comm_rank(MPI_COMM_WORLD, &debug_mpi_rank); \
	if (debug_mpi_rank == 0) \
		std::cout << content; \
} while(0)


#define DEBUG_PRINT_RANK(comm, info) do { \
	int debug_mpi_rank; \
	MPI_Comm_rank(comm, &debug_mpi_rank); \
	std::cout << "rank " << debug_mpi_rank << " " << info << std::endl; \
} while(0)


#define DEBUG_PRINT_TIME(time_micro_sec, info) do { \
	std::cout << info << " " << time_micro_sec << " milli seconds" << std::endl; \
} while(0)


#define DEBUG_PRINT_DEVICE_ID(info) do { \
	int debug_device_id; \
	CUDA_CHECK(cudaGetDevice(&debug_device_id) ); \
	std::cout << info << " device id: " << debug_device_id << std::endl; \
} while(0)


#define DEBUG_PRINT_RANK_DEVICE_ID(comm, info) do { \
	int debug_mpi_rank; \
	MPI_Comm_rank(comm, &debug_mpi_rank); \
	int debug_device_id; \
	CUDA_CHECK(cudaGetDevice(&debug_device_id) ); \
	std::cout << "rank " << debug_mpi_rank \
		<< " device id: " << debug_device_id \
		<< " " << info << std::endl; \
} while(0)


#define DEBUG_PRINT_RANK_DEVICE_ID_ITER(comm, iter, info) do { \
	int debug_mpi_rank; \
	MPI_Comm_rank(comm, &debug_mpi_rank); \
	int debug_device_id; \
	CUDA_CHECK(cudaGetDevice(&debug_device_id) ); \
	std::ostringstream s; \
	s << "rank: " << debug_mpi_rank \
		<< " device id: " << debug_device_id \
		<< " iter: " << iter << " " << info; \
	std::cout << s.str() << std::endl << std::endl; \
} while(0)


#define DEBUG_PRINT_TIME_WITH_RANK_DEVICE_ID(comm, timer, info) do { \
	int debug_mpi_rank; \
	MPI_Comm_rank(comm, &debug_mpi_rank); \
	int debug_device_id; \
	CUDA_CHECK(cudaGetDevice(&debug_device_id) ); \
	std::ostringstream s; \
	s << "rank: " << debug_mpi_rank \
		<< " device id: " << debug_device_id \
		<< info << " time: " << timer.getElapsedTimeInMilliSec() << " ms"; \
	std::cout << s.str() << std::endl; \
} while(0)


using namespace std;

// void GetGpuIds(vector<int>& gpu_ids);

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