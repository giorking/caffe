#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include <vector>
#include "cluster/comm_utils.hpp"
#include "nccl/src/nccl.h"


void Entry(ncclUniqueId clique_id, ncclComm_t* comm, int dev_id) {
	std::cout << "in entry function " << std::endl;
	int buf_size = 1000;
	CUDA_CHECK(cudaSetDevice(dev_id) );
	float* device_buf;
	float* host_buf = (float*)malloc(sizeof(float) * buf_size);
	CUDA_CHECK(cudaMalloc(&device_buf, sizeof(float) * buf_size) );
	cudaStream_t* stream_comm = (cudaStream_t*)malloc(sizeof(cudaStream_t) );
  CUDA_CHECK(cudaStreamCreate(stream_comm) );

  std::cout << "check before reduce" << std::endl;


  for (int i = 0; i < 100; i++) {
  	std::cout << "dev " << dev_id << " iter " << i << " start reduce" << std::endl;
  	usleep(dev_id * 10000000);
	  NCCL_CHECK(ncclReduce( (const void*)device_buf, (void*)device_buf, 
	  	buf_size, ncclFloat, ncclSum, 0, 
	  	*comm, *stream_comm) );
	  std::cout << "dev " << dev_id << " iter " << i << " start waiting" << std::endl;
	  cudaStreamSynchronize(*stream_comm);
	}

  std::cout << "check after reduce" << std::endl;

}


int main() {

	MPI_Init(NULL, NULL);

	ncclUniqueId clique_id;
	ncclComm_t* nccl_comm = new ncclComm_t[N_DEVICE_PER_PROC];
	std::vector<int> gpu_ids;
	GetGpuIds(gpu_ids);

  NCCL_CHECK(ncclGetUniqueId(&clique_id) );
  NCCL_CHECK(ncclCommInitAll(nccl_comm, N_DEVICE_PER_PROC, &(gpu_ids[0] ) ) );

  std::vector<std::thread> ts;
  for (int i = 0; i < 2; i++)
	  ts.push_back(std::thread (Entry, clique_id, &(nccl_comm[0] ), 0) );
  // std::thread t2(Entry, clique_id, &(nccl_comm[1] ), 1);

	for (int i = 0; i < 2; i++)
		ts[i].join();
  // t1.join();
  // t2.join();

  MPI_Finalize();
}