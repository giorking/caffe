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
	int mpi_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	int mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	// check for macro settings from comm_utils.hpp
	if (gpu_ids.size() != N_PROC_PER_MACHINE * N_DEVICE_PER_PROC) {
		std::cout << "Not enough GPU on a machine!" << std::endl;
		std::exit(1);
	}
	if (mpi_size % N_PROC_PER_MACHINE) {
		std::cout << "Processes can not be equaly distributed to machines!" << std::endl;
		std:exit(1);
	}
	if (mpi_size / N_PROC_PER_GROUP <= 1) {
		std::cout << "Need multiple group to test async worker!" << std::endl;
		std::exit(1);
	}

	std::vector<AsyncWorker<Dtype>* > workers(N_DEVICE_PER_PROC, NULL);
	// on clique root worker, we have one comm thread and compute thread to acess it
	ncclUniqueId clique_id;
  NCCL_CHECK(ncclGetUniqueId(&clique_id) );
  pthread_barrier_t* process_barrier = new pthread_barrier_t;
  pthread_barrier_init(process_barrier, NULL, N_DEVICE_PER_PROC);
	// construct the memory shared by all the async workers in the current process
	AsyncMem<Dtype> async_mem;
	for (int i = 0; i < N_DEVICE_PER_PROC; i++) {
		// TODO Jian: add solvers
		// E.g. there may be 8 gpus, but we optimize for NUMA where each proc have 4 device
		int gpu_id = (mpi_rank % (gpu_ids.size() / N_DEVICE_PER_PROC) ) * N_DEVICE_PER_PROC + i;
		
		// DEBUG
		std::cout << " test gpu_id" << std::endl;
		DEBUG_PRINT_RANK(MPI_COMM_WORLD, gpu_id);

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
	std::vector<std::thread*> worker_threads(N_DEVICE_PER_PROC, NULL);
	for (int i = 0; i < N_DEVICE_PER_PROC; i++) {
		// worker_threads[i] = new std::thread(std::thread (&AsyncWorker<Dtype>::Run, std::ref(*workers[i] ) ) );
		std::thread* worker_thread = new std::thread(std::thread (&AsyncWorker<Dtype>::Run, std::ref(*workers[i] ) ) );
		worker_threads[i] = worker_thread;
	}

	for (int i = 0; i < N_DEVICE_PER_PROC; i++)
		worker_threads[i]->join();

	for (int i = 0; i < N_DEVICE_PER_PROC; i++) {
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
	MPI_Init(NULL, NULL);

	Train<float>();

	MPI_Finalize();
	return 0;
}