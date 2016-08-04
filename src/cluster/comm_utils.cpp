#include <vector>
#include "cluster/comm_utils.hpp"

void GetGpuIds(std::vector<int>& gpu_ids) {
  int n_gpus = 0;
  // CUDA_CHECK(cudaGetDeviceCount(&n_gpus) );
  cudaGetDeviceCount(&n_gpus);
  gpu_ids.clear();
  for (int i = 0; i < n_gpus; i++)
    gpu_ids.push_back(i);
  return;
}