#include <vector>
#include <iostream>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include "mpi.h"
#include "glog/logging.h"
// #include "cluster/debug_utils.hpp"
#include "cluster/comm_utils.hpp"
#include "cluster/sync_communicator.hpp"

using std::vector;

namespace caffe {

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
	vector<Dtype>& rand_val, int machine_id, int n_thread, 
	int64_t buf_size, pthread_barrier_t* process_barrier) {
  SyncCommunicator<Dtype> comm(config, process_barrier); 

  Dtype* external_gpu_buf = NULL;
  Dtype* external_cpu_buf = NULL;
  CUDA_CHECK(cudaSetDevice(config.GetDeviceId() ) );
  CUDA_CHECK(cudaMalloc(&external_gpu_buf, sizeof(Dtype) * buf_size) );
#ifndef GPU_DIRECT_MPI
  if (comm.IsCliqueRoot() )
	  external_cpu_buf = (Dtype*)malloc(sizeof(Dtype) * buf_size);
#endif

  comm.Init(buf_size, external_cpu_buf, external_gpu_buf, buf_size);

  Dtype* host_buffer = new Dtype[buf_size];
  /*copy into GPU memory*/
  for (int i = 0; i < buf_size; i++)
  	host_buffer[i] = rand_val[machine_id];

  CUDA_CHECK(cudaMemcpy(comm.GetGpuBuffer(), host_buffer, 
  	sizeof(Dtype) * buf_size, cudaMemcpyHostToDevice) );

  /*do group synchronization operation for testing*/
  comm.SyncGroup(true);	


  /*copy GPU memory out for testing*/
  CUDA_CHECK(cudaMemcpy(host_buffer, comm.GetGpuBuffer(), 
  	sizeof(Dtype) * buf_size, cudaMemcpyDeviceToHost) );

  Dtype correct_val = 0.0;
  for (int i = 0; i < rand_val.size(); i++)
  	correct_val += rand_val[i];
  correct_val *= n_thread;

  int MPI_size;
  MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);

 	for (int i = 0; i < buf_size; i++)
 		assert(std::abs(host_buffer[i] * n_thread * MPI_size - correct_val) <= 1e-7 * correct_val);

  if (host_buffer != NULL)
	  delete host_buffer;
	if (external_gpu_buf != NULL)
		cudaFree(external_gpu_buf);
	if (external_cpu_buf != NULL) 
		free(external_cpu_buf);
}	

/**
 * Test All reduce with in a synchronized group
 */
template<typename Dtype>
void TestMPIAllReduce(int argc, char** argv) {
	int rank;
	int n_proc;
	// MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

	nProcPerGroup = n_proc;

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
	nDevicePerProc = gpu_ids.size();

	vector<SyncCommConfig<Dtype> > configs;
	GetConfigForClique(configs, gpu_ids);

  pthread_barrier_t process_barrier;
  pthread_barrier_init(&process_barrier, NULL, gpu_ids.size() );

	vector<std::thread> threads;
	for (int i = 0; i < configs.size(); i++) 
		// MPIAllReduceThread<Dtype>(configs[i], rand_value[rank], rand_value, rank, gpu_ids.size() );
		threads.push_back(std::thread(MPIAllReduceThread<Dtype>, std::ref(configs[i] ), 
			rand_value[rank], std::ref(rand_value), rank, gpu_ids.size(), 20000000, 
			&process_barrier) );

	for (int i = 0; i < gpu_ids.size(); i++)
    threads[i].join();

  std::cout << "MPI sync group communication test passed" << std::endl;

	// MPI_Finalize();
}

template <typename Dtype>
void TestMPIAllGather(int argc, char** argv) {
	int rank;
	int n_proc;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

	// set those global variables
	MPI_Comm_size(MPI_COMM_WORLD, &nProcPerGroup);
	nDevicePerProc = 1;

	if (nProcPerGroup == 1) {
		std::cout << "Need multiple processes in one group. Modify nProcPerGroup." << std::endl;
		std::exit(1);		
	}

	// share random value
	int64_t buf_size = 0;
	MPI_Datatype type = DtypeToMPIDtype<Dtype>::type;
	if (rank == 0) {
		srand(time(NULL) );
		while (buf_size == 0)
			buf_size = rand() % 1000000;
		for (int i = 1; i < n_proc; i++)
			MPI_Send(&buf_size, 1, MPI_LONG_LONG_INT, i, 0, MPI_COMM_WORLD);
	}
	else
		MPI_Recv(&buf_size, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	vector<Dtype> rand_value(buf_size, 0.0);

	if (rank == 0) {
		for (int i = 0; i < buf_size; i++) {
			Dtype value = rand() / (Dtype)RAND_MAX;
			while (value == 0)
				value = rand() / (Dtype)RAND_MAX;
			rand_value[i] = value;
			assert(rand_value[i] != 0);
		}
		for (int i = 1; i < n_proc; i++)
			MPI_Send(&(rand_value[0] ), buf_size, type, i, 0, MPI_COMM_WORLD);	
	}
	else
		MPI_Recv(&(rand_value[0] ), buf_size, type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	vector<Dtype> copy_value(rand_value);

  vector<int> gpu_ids;
  GetGpuIds(gpu_ids);

	vector<SyncCommConfig<Dtype> > configs;
	GetConfigForClique(configs, gpu_ids);

	int64_t block_size = buf_size / n_proc;

	for (int i = 0; i < block_size * rank; i++)
		copy_value[i] = 0;
	if (rank != n_proc - 1) {
		for (int i = block_size * (rank + 1); i < buf_size; i++)
			copy_value[i] = 0;
	}

	SyncCommunicator<Dtype> comm(configs[0], NULL);

#ifdef GPU_DIRECT_MPI
	if (rank % 2 == 0)
    cudaSetDevice(0);
  else
    cudaSetDevice(0);
	Dtype* mpi_buf;
	cudaMalloc(&mpi_buf, sizeof(Dtype) * buf_size);
	cudaMemcpy(mpi_buf, &(copy_value[0] ), 
		sizeof(Dtype) * buf_size, cudaMemcpyHostToDevice);
	comm.Init(buf_size, NULL, mpi_buf, 0); 
#else
	Dtype* mpi_buf = &(copy_value[0] );	
	comm.Init(buf_size, mpi_buf, NULL, 0);
#endif


	comm.InterMachineAllGather();

#ifdef GPU_DIRECT_MPI
	cudaMemcpy(&(copy_value[0] ), mpi_buf, sizeof(Dtype) * buf_size, cudaMemcpyDeviceToHost );
#endif

	for (int i = 0; i < buf_size; i++) {
		if (copy_value[i] != rand_value[i] )
			std::cout << "wrong " << rank << " ind " << i << std::endl;	
		assert(copy_value[i] == rand_value[i] );
	}

#ifdef GPU_DIRECT_MPI
	cudaFree(mpi_buf);
#endif

  std::cout << "MPI all gather communication test passed" << std::endl;

}


template <typename Dtype>
void TestMPIReduceScatter(int argc, char** argv, Dtype tolerance) {
	int rank;
	int n_proc;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

	// set those global variables
	MPI_Comm_size(MPI_COMM_WORLD, &nProcPerGroup);
	nDevicePerProc = 1;

	if (nProcPerGroup == 1) {
		std::cout << "Need multiple processes in one group. Modify nProcPerGroup." << std::endl;
		std::exit(1);		
	}

	// share random value
	int64_t buf_size = 0;
	MPI_Datatype type = DtypeToMPIDtype<Dtype>::type;
	if (rank == 0) {
		int seed = time(NULL);
		srand(seed);
		while (buf_size == 0)
			buf_size = rand() % 1000000;
		for (int i = 1; i < n_proc; i++)
			MPI_Send(&buf_size, 1, MPI_LONG_LONG_INT, i, 0, MPI_COMM_WORLD);
	}
	else
		MPI_Recv(&buf_size, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	vector<Dtype> rand_value(buf_size, 0.0);

	if (rank == 0) {
		for (int i = 0; i < buf_size; i++) {
			Dtype value = rand() / (Dtype)RAND_MAX;
			while (value == 0)
				value = rand() / (Dtype)RAND_MAX;
			rand_value[i] = value;
			assert(rand_value[i] != 0);
		}
		for (int i = 1; i < n_proc; i++)
			MPI_Send(&(rand_value[0] ), buf_size, type, i, 0, MPI_COMM_WORLD);	
	}
	else
		MPI_Recv(&(rand_value[0] ), buf_size, type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	vector<Dtype> copy_value(rand_value);

  vector<int> gpu_ids;
  GetGpuIds(gpu_ids);

	vector<SyncCommConfig<Dtype> > configs;
	GetConfigForClique(configs, gpu_ids);

	int64_t block_size = buf_size / n_proc;

	for (int i = 0; i < buf_size; i++)
		copy_value[i] *= (rank + 1);

	SyncCommunicator<Dtype> comm(configs[0], NULL);

#ifdef GPU_DIRECT_MPI
  if (rank % 2 == 0)
  	cudaSetDevice(0);
  else
    cudaSetDevice(0);
	Dtype* mpi_buf;
	Dtype* mpi_buf_tmp;
	cudaMalloc(&mpi_buf, sizeof(Dtype) * buf_size);
	cudaMalloc(&mpi_buf_tmp, sizeof(Dtype) * (block_size * (n_proc / 2 - 1) + buf_size - (n_proc - 1) * block_size) );
	cudaMemcpy(mpi_buf, &(copy_value[0] ), 
		sizeof(Dtype) * buf_size, cudaMemcpyHostToDevice);
	comm.Init(buf_size, NULL, mpi_buf, block_size * (n_proc / 2 - 1) + buf_size - (n_proc - 1) * block_size);
#else
	Dtype* mpi_buf = &(copy_value[0] );	
	Dtype* mpi_buf_tmp = new Dtype[block_size * (n_proc / 2 - 1) + buf_size - (n_proc - 1) * block_size];
	comm.Init(buf_size, mpi_buf, NULL, block_size * (n_proc / 2 - 1) + buf_size - (n_proc - 1) * block_size); 
#endif

	comm.InterMachineReduceScatter();

#ifdef GPU_DIRECT_MPI
	cudaMemcpy(&(copy_value[0] ), mpi_buf, sizeof(Dtype) * buf_size, cudaMemcpyDeviceToHost );
#endif

	int64_t tmp_buf_size = (rank == n_proc - 1) ? buf_size - block_size * rank : block_size; 
	for (int i = block_size * rank; i < block_size * rank + tmp_buf_size; i++) {
		if (std::abs( (copy_value[i] - rand_value[i] * (1 + n_proc) * n_proc / 2 ) / copy_value[i] ) > tolerance)
			std::cout << "wrong " << rank << " ind " << i 
				<< " " << copy_value[i] << " " 
				<< rand_value[i] * (1 + n_proc) * n_proc / 2 
				<< copy_value[i] - rand_value[i] * (1 + n_proc) * n_proc / 2 << std::endl;	
		assert(std::abs( (copy_value[i] - rand_value[i] * (1 + n_proc) * n_proc / 2 ) / copy_value[i] ) <= tolerance);
	}

#ifdef GPU_DIRECT_MPI
	cudaFree(mpi_buf);
	cudaFree(mpi_buf_tmp);
#else
	free(mpi_buf_tmp);
#endif

  std::cout << "MPI reduce scatter communication test passed" << std::endl;

}

} // end of namespace caffe



int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	for (int i = 0; i < 2; i++) {
		caffe::TestMPIAllGather<float>(argc, argv);
		caffe::TestMPIAllGather<double>(argc, argv);
	}

	for (int i = 0; i < 2; i++) {
		caffe::TestMPIReduceScatter<float>(argc, argv, 1e-6);
		caffe::TestMPIReduceScatter<double>(argc, argv, 1e-8);
	}

	for (int i = 0; i < 2; i++) {
		caffe::TestMPIAllReduce<float>(argc, argv);
		caffe::TestMPIAllReduce<double>(argc, argv);	 
	}

	MPI_Finalize();

	return 0;
}