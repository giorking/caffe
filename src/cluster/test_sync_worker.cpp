#include <mpi.h>
#include <thread>
#include <pthread.h>
#include <cassert>
#include "cluster/debug_utils.hpp"
#include "cluster/comm_utils.hpp"
#include "cluster/async_communicator.hpp"
#include "cluster/async_worker.hpp"
#include "cluster/solver.hpp"


// initilize global variable
// the number of processes in a synchronized group.
int nProcPerGroup;

// the number of machines in a synchronized group.
int nMachinePerGroup;

/**
 * the number of process on a single machine. Derived from
 * nProcPerGroup and nMachinePerGroup.
 */
int nProcPerMachine;

// the number of gpu cards each process has. 
int nDevicePerProc;


template <typename Dtype>
void Train(int argc, char** argv) {
	/**
	 * count the number of GPUs available from this process
	 * and init all the workers.
	 */
	std::vector<int> gpu_ids;
	GetGpuIds(gpu_ids);
	int mpi_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	int mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	
	// parse system setting environment
	ParseSysConfigArg(argc, argv);

	// check for macro settings from comm_utils.hpp
	if (gpu_ids.size() != nProcPerMachine * nDevicePerProc) {
		std::cout << "Not enough GPU on a machine!" << std::endl;
		std::exit(1);
	}
	if (mpi_size % nProcPerMachine) {
		std::cout << "Processes can not be equaly distributed to machines!" << std::endl;
		std:exit(1);
	}
	if (mpi_size / nProcPerGroup != 1) {
		std::cout << "Need a single group to test sync worker!" << std::endl;
		std::exit(1);
	}

	std::vector<Worker<Dtype>* > workers(nDevicePerProc, NULL);
	ncclUniqueId clique_id;
  NCCL_CHECK(ncclGetUniqueId(&clique_id) );
  pthread_barrier_t* process_barrier = new pthread_barrier_t;
  pthread_barrier_init(process_barrier, NULL, nDevicePerProc);
	for (int i = 0; i < nDevicePerProc; i++) {
		// TODO Jian: add solvers
		int gpu_id = (mpi_rank % (gpu_ids.size() / nDevicePerProc) ) * nDevicePerProc + i;
		SyncCommConfig<Dtype> sync_config(gpu_id, clique_id);
		workers[i] = new Worker<Dtype>(sync_config, process_barrier);
	}

	/**
	 * As we have some communication group splitting, we need to 
	 * explicitly set barrier here to prevent one process from 
	 * starting send too early.
	 */
	MPI_Barrier(MPI_COMM_WORLD);

	// start spawn process and compute
	std::vector<std::thread*> worker_threads;

	for (int i = 0; i < nDevicePerProc; i++) {
		// std::thread* worker_thread = new std::thread(std::thread (InitAndRunWorker<Dtype>, &worker_init, workers[i] ) );
		std::thread* worker_thread = new std::thread(std::thread (&Worker<Dtype>::Run, std::ref(*workers[i] ) ) );
		worker_threads.push_back(worker_thread);
	}
	for (int i = 0; i < nDevicePerProc; i++)
		worker_threads[i]->join();

	for (int i = 0; i < nDevicePerProc; i++) {
		delete worker_threads[i];
		delete workers[i];
	}
	if (process_barrier != NULL) {
		pthread_barrier_destroy(process_barrier);
		process_barrier = NULL;
	}

	std::cout << "async worker test passed!" << std::endl;
}


// int main(int argc, char** argv) {
// 	int rank;
// 	int size;
// 	MPI_Init(NULL, NULL);

// 	Train<float>(argc, argv);

// 	std::cout << "start finalize" << std::endl;
	
// 	MPI_Finalize();
// }