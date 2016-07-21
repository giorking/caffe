#include <vector>
#include <iostream>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include "mpi.h"
#include "cluster/communicator.hpp"
#include "caffe/util/device_alternate.hpp"


template<typename Dtype>
void ThreadEntry(CommConfig<Dtype>& config) {
	int64_t buf_size = config.GetGpuBufferSize();
  Dtype* host_buffer = new Dtype[buf_size];
  for (int i = 0; i < buf_size; i++)
    host_buffer[i] = config.GetDeviceId() + 1;
  Communicator<Dtype> comm(config);
  // CUDA_CHECK(cudaMemcpy(comm.GetGpuBuffer(), host_buffer, sizeof(Dtype) * buf_size, cudaMemcpyHostToDevice) ); 
  // comm.SyncGroup();	
}	


int main(int argc, char** argv) {
	int rank;
	int n_proc;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);






}