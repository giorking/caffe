#include <mpi.h>
#include <thread>
#include <pthread.h>
#include <cassert>
#include "cluster/debug_utils.hpp"
#include "cluster/comm_utils.hpp"
#include "cluster/async_communicator.hpp"
#include "cluster/async_worker.hpp"
#include "cluster/solver.hpp"


template <typename Dtype>
void Train() {
	/**
	 * count the number of GPUs available from this process
	 * and init all the workers.
	 */
	vector<int> gpu_ids;
	GetGpuIds(gpu_ids);

	// check for macro settings from comm_utils.hpp
	if (gpu_ids.size() != N_PROC_PER_MACHINE * N_DEVICE_PER_PROC) {
		std::cout << "Not enough GPU on a machine!" << std::endl;
		std::exit(1);
	}
	int mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	if (mpi_size % N_PROC_PER_MACHINE) {
		std::cout << "Processes can not be equaly distributed to machines!" << std::endl;
		std:exit(1);
	}
	if (mpi_size / N_PROC_PER_GROUP != 1) {
		std::cout << "Need a single group to test sync worker!" << std::endl;
		std::exit(1);
	}

	std::vector<Worker<Dtype> > workers;
	ncclUniqueId clique_id;
	ncclComm_t* nccl_comm = new ncclComm_t[N_DEVICE_PER_PROC];
  // NCCL_CHECK(ncclGetUniqueId(&clique_id) );
  NCCL_CHECK(ncclCommInitAll(nccl_comm, N_DEVICE_PER_PROC, &(gpu_ids[0] ) ) );
	for (int i = 0; i < N_DEVICE_PER_PROC; i++) {
		// TODO Jian: add solvers
		SyncCommConfig<Dtype> sync_config(gpu_ids[i], clique_id, nccl_comm + i);
		Worker<Dtype> worker(sync_config);
		workers.push_back(worker);
	}

	/**
	 * As we have some communication group splitting, we need to 
	 * explicitly set barrier here to prevent one process from 
	 * starting send too early.
	 */
	MPI_Barrier(MPI_COMM_WORLD);
	// start spawn process and compute
	for (int i = 0; i < N_DEVICE_PER_PROC; i++)
		workers[i].Run();

	delete[] nccl_comm;
	std::cout << "async worker test passed!" << std::endl;
}


int main(int argc, char** argv) {
	int rank;
	int size;
	MPI_Init(NULL, NULL);

	Train<float>();
	
	MPI_Finalize();
}