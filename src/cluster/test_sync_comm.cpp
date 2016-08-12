#include <vector>
#include <iostream>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include "mpi.h"
#include "glog/logging.h"
#include "cluster/debug_utils.hpp"
#include "cluster/comm_utils.hpp"
#include "cluster/sync_communicator.hpp"

using namespace std;

template<typename Dtype>
void GetConfigForClique(vector<SyncCommConfig<Dtype> >& configs, vector<int>& gpu_ids) {
	/* Generate configs for a clique of size n_config */
	ncclUniqueId clique_id;
  NCCL_CHECK(ncclGetUniqueId(&clique_id) );

  for (int i = 0; i < gpu_ids.size(); i++) {
	  SyncCommConfig<Dtype> config(gpu_ids[i], clique_id);
	  configs.push_back(config);
	}
}


template<typename Dtype>
void MPIAllReduceThread(SyncCommConfig<Dtype>& config, Dtype buf_val, 
	vector<Dtype>& rand_val, int machine_id, int n_thread, int64_t buf_size) {
  SyncCommunicator<Dtype> comm(config, buf_size); 
  Dtype* host_buffer = new Dtype[buf_size];
  /*copy into GPU memory*/
  for (int i = 0; i < buf_size; i++)
  	host_buffer[i] = rand_val[machine_id];
  CUDA_CHECK(cudaMemcpy(comm.GetGpuBuffer(), host_buffer, 
  	sizeof(Dtype) * buf_size, cudaMemcpyHostToDevice) );

  // // DEBUG
  for (int i = 0; i < rand_val.size() ; i++)
  	std::cout << "test host " << i << " " << rand_val[i] << std::endl;

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

  // DEBUG 
  std::cout << "correct value " << correct_val << std::endl;
  std::cout << "compute value " << host_buffer[0] << std::endl;

 	for (int i = 0; i < buf_size; i++)
 		assert(std::abs(host_buffer[i] - correct_val) <= 1e-8 * correct_val);

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
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

	if (nProcPerGroup == 1) {
		std::cout << "Need multiple processes in one group. Modify nProcPerGroup." << std::endl;
		std::exit(1);		
	}

	// share random value 
	vector<Dtype> rand_value(n_proc, 0.0);
	MPI_Datatype type = DtypeToMPIDtype<Dtype>::type;
	if (rank == 0) {
		srand(100);
		for (int i = 0; i < n_proc; i++) {
			Dtype value = rand() / (Dtype)RAND_MAX;
			rand_value[i] = value;
			if (i != 0)
				MPI_Send(&(rand_value[0] ), n_proc, type, i, 0, MPI_COMM_WORLD);	
		}
	}
	else
		MPI_Recv(&(rand_value[0] ), n_proc, type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  vector<int> gpu_ids;
  GetGpuIds(gpu_ids);

	vector<SyncCommConfig<Dtype> > configs;
	GetConfigForClique(configs, gpu_ids);


	vector<std::thread> threads;
	for (int i = 0; i < configs.size(); i++) 
		// MPIAllReduceThread<Dtype>(configs[i], rand_value[rank], rand_value, rank, gpu_ids.size() );
		threads.push_back(std::thread(MPIAllReduceThread<Dtype>, std::ref(configs[i] ), 
			rand_value[rank], std::ref(rand_value), rank, gpu_ids.size(), 20000000) );

	for (int i = 0; i < gpu_ids.size(); i++)
    threads[i].join();

  std::cout << "MPI all reduce communication test passed" << std::endl;

	MPI_Finalize();
}


int main(int argc, char** argv) {
	TestMPIAllReduce<float>(argc, argv);
	// TestMPIAllReduce<double>(argc, argv);	 

	return 0;
}