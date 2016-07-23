// #include <vector>
// #include <iostream>
// #include <thread>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cassert>
// #include "cluster/communicator.hpp"
// #include "caffe/util/device_alternate.hpp"


// using namespace std;
  
// /**
//  * init the buf on device i with i + 1
//  * assert whether the all-reduce results is correct.
//  */
// template<typename Dtype>
// void ThreadEntrySingleNode(CommConfig<Dtype>& config, int n_device) {
//   int64_t buf_size = config.GetGpuBufferSize();
//   Dtype* host_buffer = new Dtype[buf_size];
//   for (int i = 0; i < buf_size; i++)
//     host_buffer[i] = config.GetDeviceId() + 1;
//   Communicator<Dtype> comm(config);
//   CUDA_CHECK(cudaMemcpy(comm.GetGpuBuffer(), host_buffer, sizeof(Dtype) * buf_size, cudaMemcpyHostToDevice) ); 
//   comm.SyncGroup();

//   CUDA_CHECK(cudaMemcpy(host_buffer, comm.GetGpuBuffer(), sizeof(Dtype) * buf_size, cudaMemcpyDeviceToHost) );
//   Dtype correct_ans = 0;
//   for (int i = 0; i < n_device; i++)
//     correct_ans += i + 1;
//   for (int i = 0; i < buf_size; i++)
//     assert(host_buffer[i] == correct_ans);
// }

// void GetGpuIds(vector<int>& gpu_ids) {
//   int n_gpus = 0;
//   CUDA_CHECK(cudaGetDeviceCount(&n_gpus) );
//   gpu_ids.clear();
//   for (int i = 0; i < n_gpus; i++)
//     gpu_ids.push_back(i);
//   return;
// }


// void TestNCCLComm() {
//   vector<int> gpu_ids;
//   GetGpuIds(gpu_ids);
//   int n_gpus = gpu_ids.size();
//   vector<thread> threads;
//   vector<CommConfig<float> > configs;
//   int comm_buffer_size = 100;
//   ncclUniqueId clique_id;
//   NCCL_CHECK(ncclGetUniqueId(&clique_id) );


//   float* buffer = NULL;
//   for (int i = 0; i < gpu_ids.size(); i++) {
//     configs.push_back(CommConfig<float> (0, i, gpu_ids.size(), clique_id, i, 0,
//       comm_buffer_size, comm_buffer_size, comm_buffer_size) ); 
//   }

//   for (int i = 0; i < gpu_ids.size(); i++)
//     threads.push_back(thread(ThreadEntrySingleNode<float>, std::ref(configs[i] ), gpu_ids.size() ) );

//   for (int i = 0; i < gpu_ids.size(); i++)
//     threads[i].join();

//   std::cout << "NCCL Communication test passed!" << std::endl;
// }


// int main() {
  
//   TestNCCLComm();

//   return 0;
// }
