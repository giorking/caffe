#include <vector>
#include <iostream>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include "mpi.h"
#include "cluster/communicator.hpp"
#include "cluster/debug_utils.hpp"
#include "caffe/util/device_alternate.hpp"

using namespace std;

template<typename Dtype>
void GetConfigForClique(vector<CommConfig<Dtype> >& configs,
	vector<int>& gpu_ids, int n_config, bool set_group_root, 
	int buf_size, int mpi_rank) {
	/* Generate configs for a clique of size n_config */
	ncclUniqueId clique_id;
  NCCL_CHECK(ncclGetUniqueId(&clique_id) );

	assert(gpu_ids.size() >= n_config);

  for (int i = 0; i < n_config; i++) {
  	if (set_group_root && i == 0)
	    configs.push_back(CommConfig<Dtype> (gpu_ids[i], 0, n_config, i, 0, clique_id, 
	    	true, buf_size, mpi_rank, buf_size, buf_size) );
	  else
	  	configs.push_back(CommConfig<Dtype> (gpu_ids[i], 0, n_config, i, 0, clique_id, 
	    	false, buf_size, mpi_rank, buf_size, buf_size) ); 
  }	
}


template<typename Dtype>
void MPIAllReduceThread(CommConfig<Dtype> config, Dtype buf_val, 
	vector<Dtype>& rand_val, int machine_id, int n_thread) {
  Communicator<Dtype> comm(config); 
	int64_t buf_size = config.GetGpuBufferSize();
  Dtype* host_buffer = new Dtype[buf_size];
  /*copy into GPU memory*/
  for (int i = 0; i < config.GetGpuBufferSize(); i++)
  	host_buffer[i] = rand_val[machine_id];
  CUDA_CHECK(cudaMemcpy(comm.GetGpuBuffer(), host_buffer, 
  	sizeof(Dtype) * buf_size, cudaMemcpyHostToDevice) );

  // // DEBUG
  // for (int i = 0; i < rand_val.size() ; i++)
  // 	std::cout << "test host " << i << " " << rand_val[i] << std::endl;

  // // std::cout << "host buff " << host_buffer[0] << " " << buf_size << " " << rand_val[machine_id] << std::endl;

  // std::string str1 = "before reduce " + std::to_string(machine_id);
  // DisplayGpuArray(comm.GetGpuBuffer(), 3, str1);

  /*do group synchronization operation for testing*/
  comm.SyncGroup();	
  /*copy GPU memory out for testing*/
  CUDA_CHECK(cudaMemcpy(host_buffer, comm.GetGpuBuffer(), 
  	sizeof(Dtype) * buf_size, cudaMemcpyDeviceToHost) );
  Dtype correct_val = 0.0;
  for (int i = 0; i < rand_val.size(); i++)
  	correct_val += rand_val[i];
  correct_val *= n_thread;

  // // DEBUG 
  // std::cout << "correct value " << correct_val << std::endl;

 	for (int i = 0; i < buf_size; i++)
 		assert(abs(host_buffer[i] - correct_val) <= 1e-8 * correct_val);

  if (host_buffer != NULL)
	  delete host_buffer;
}	

/**
 * Test All reduce with in a synchronized group
 */
template<typename Dtype>
void TestMPIAllReduce(int argc, char** argv) {
	int rank;
	int n_proc;
	MPI::Init(&argc, &argv);
	MPI::Comm_rank(MPI::COMM_WORLD, &rank);
	MPI::Comm_size(MPI::COMM_WORLD, &n_proc);

	/* single-thread test */
	vector<Dtype> rand_value(n_proc, 0.0);
	MPI::Datatype type = DtypeToMPIDtype<Dtype>::type;
	if (rank == 0) {
		srand(100);
		for (int i = 0; i < n_proc; i++) {
			Dtype value = rand() / (Dtype)RAND_MAX;
			rand_value[i] = value;
			if (i != 0)
				MPI::Send(&(rand_value[0] ), n_proc, type, i, 0, MPI::COMM_WORLD);	
		}
	}
	else
		MPI::Recv(&(rand_value[0] ), n_proc, type, 0, 0, MPI::COMM_WORLD, MPI::STATUS_IGNORE);

  vector<int> gpu_ids;
  GetGpuIds(gpu_ids);

	vector<CommConfig<Dtype> > configs;
	if (rank == 0)
		GetConfigForClique(configs, gpu_ids, gpu_ids.size(), true, 10000, rank);
	else
		GetConfigForClique(configs, gpu_ids, gpu_ids.size(), false, 10000, rank);


	vector<std::thread> threads;
	for (int i = 0; i < configs.size(); i++) 
		// MPIAllReduceThread<Dtype>(configs[i], rand_value[rank], rand_value, rank, gpu_ids.size() );
		threads.push_back(std::thread(MPIAllReduceThread<Dtype>, std::ref(configs[i] ), 
			rand_value[rank], std::ref(rand_value), rank, gpu_ids.size() ) );

	for (int i = 0; i < gpu_ids.size(); i++)
    threads[i].join();

  std::cout << "MPI all reduce communication test passed" << std::endl;

	MPI::Finalize();
}


int main(int argc, char** argv) {
	// TestMPIAllReduce<float>(argc, argv);
	TestMPIAllReduce<double>(argc, argv);	 

	return 0;
}