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
	vector<int> gpu_ids;
	GetGpuIds(gpu_ids);
	int mpi_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	int mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	// parse system setting environment
	ParseCmdArg(argc, argv);

	// check for macro settings from comm_utils.hpp
	if (gpu_ids.size() != nProcPerMachine * nDevicePerProc) {
		std::cout << "Not enough GPU on a machine!" << std::endl;
		std::exit(1);
	}
	if (mpi_size % nProcPerMachine) {
		std::cout << "Processes can not be equaly distributed to machines!" << std::endl;
		std:exit(1);
	}
	if (mpi_size / nProcPerGroup <= 1) {
		std::cout << "Need multiple group to test async worker!" << std::endl;
		std::exit(1);
	}

	std::vector<AsyncWorker<Dtype>* > workers(nDevicePerProc, NULL);
	// on clique root worker, we have one comm thread and compute thread to acess it
	ncclUniqueId clique_id;
  NCCL_CHECK(ncclGetUniqueId(&clique_id) );
  pthread_barrier_t* process_barrier = new pthread_barrier_t;
  pthread_barrier_init(process_barrier, NULL, nDevicePerProc);
	// construct the memory shared by all the async workers in the current process
	AsyncMem<Dtype> async_mem;
	for (int i = 0; i < nDevicePerProc; i++) {
		// TODO Jian: add solvers
		// E.g. there may be 8 gpus, but we optimize for NUMA where each proc have 4 device
		int gpu_id = (mpi_rank % (gpu_ids.size() / nDevicePerProc) ) * nDevicePerProc + i;
		SyncCommConfig<Dtype> sync_config(gpu_id, clique_id);
		AsyncCommConfig<Dtype> async_config;
		workers[i] = new AsyncWorker<Dtype>(sync_config, process_barrier, 
			async_config, &async_mem);
	}

	/**
	 * As we have some communication group splitting, we need to 
	 * explicitly set barrier here to prevent one process from 
	 * starting send too early.
	 */
	MPI_Barrier(MPI_COMM_WORLD);

	// start spawn process and compute
	std::vector<std::thread*> worker_threads(nDevicePerProc, NULL);
	for (int i = 0; i < nDevicePerProc; i++) {
		// worker_threads[i] = new std::thread(std::thread (&AsyncWorker<Dtype>::Run, std::ref(*workers[i] ) ) );
		std::thread* worker_thread = new std::thread(std::thread (&AsyncWorker<Dtype>::Run, std::ref(*workers[i] ) ) );
		worker_threads[i] = worker_thread;
	}

	for (int i = 0; i < nDevicePerProc; i++)
		worker_threads[i]->join();

	for (int i = 0; i < nDevicePerProc; i++) {
		if (worker_threads[i] != NULL)
			delete worker_threads[i];
		if (workers[i] != NULL)
			delete workers[i];
	}
	if (process_barrier != NULL) {
		pthread_barrier_destroy(process_barrier);
		process_barrier = NULL;
	}

	std::cout << "Async worker test passed!" << std::endl;

}


int main(int argc, char** argv) {
	int rank;
	int size;
	int requested_thread_level = MPI_THREAD_MULTIPLE;
	int provided_thread_level;
	MPI_Init_thread(NULL, NULL, requested_thread_level, &provided_thread_level);
	if (provided_thread_level != requested_thread_level) {
		std::cout << "MPI multiple thread support is not provided!" << std::endl;
		exit(1);
	}

	Train<float>(argc, argv);

	MPI_Finalize();
	return 0;
}