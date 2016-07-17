#include <vector>
#include <iostream>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include "cluster/communicator.hpp"
#include "caffe/util/device_alternate.hpp"
// #include "caffe/util/math_functions.hpp"


using namespace std;

template<typename Dtype>
void ThreadEntry(CommConfig<Dtype>& config) {
  // Communicator<Dtype> comm(config); 
  // CUDA_CHECK(cudaSetDevice(config.GetDeviceId() ) );
  std::cout << "done" << std::endl;
}

void GetGpuIds(vector<int>& gpu_ids) {
  int n_gpus = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n_gpus) );
  gpu_ids.clear();
  for (int i = 0; i < n_gpus; i++)
    gpu_ids.push_back(i);
  return;
}


int main() {
  vector<int> gpu_ids;
  GetGpuIds(gpu_ids);
  int n_gpus = gpu_ids.size();
  vector<thread> threads;
  vector<CommConfig<float> > configs;
  int comm_buffer_size = 2e9;
  ncclUniqueId clique_id;
  NCCL_CHECK(ncclGetUniqueId(&clique_id) );


  float* test = NULL;

  for (int i = 0; i < gpu_ids.size(); i++) {
    configs.push_back(CommConfig<float> (0, i, gpu_ids.size(), clique_id, i) ); 
    CUDA_CHECK(cudaSetDevice(i) );
    float* buffer = NULL;
    CUDA_CHECK(cudaMalloc(&buffer, sizeof(float) * comm_buffer_size) );
    for (int j = 0; j < comm_buffer_size; j++)
      buffer[j] == i;
    configs[i].SetGpuBuffer(buffer, comm_buffer_size);
  }

  // // init ncclCommunication
  // ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * n_gpus);
  // NCCLCHECK(ncclCommInitAll(comms, n_gpus, ) );

  for (int i = 0; i < gpu_ids.size(); i++) {
    int left_gpu_id = (i + n_gpus - 1) % n_gpus;
    int right_gpu_id = (i + n_gpus + 1) % n_gpus;
    configs[i].SetLeftGpuBuffer(configs[left_gpu_id].GetGpuBuffer(), 
      configs[left_gpu_id].GetGpuBufferSize() );
    configs[i].SetRightGpuBuffer(configs[right_gpu_id].GetGpuBuffer(), 
      configs[left_gpu_id].GetGpuBufferSize() );
    threads.push_back(thread(ThreadEntry<float>, std::ref(configs[i] ) ) );
  }  

  for (int i = 0; i < gpu_ids.size(); i++) {
    if (configs[i].GetGpuBuffer() != NULL)
      CUDA_CHECK(cudaFree(configs[i].GetGpuBuffer() ) );
  }

  // for (int i = 0; i < gpu_ids.size(); i++)
  //   threads[i].join();


  return 0;
}
