#include <mpi.h>
#include <thread>
#include <pthread.h>
#include <cassert>
#include "cluster/debug_utils.hpp"
#include "cluster/comm_utils.hpp"
#include "cluster/async_communicator.hpp"
#include "cluster/async_worker.hpp"
#include "cluster/solver.hpp"


// template <typename Dtype>
// void InitAndRunWorker(pthread_barrier_t* init_barrier, Worker<Dtype>* worker) { 
// 	worker->Init();
// 	pthread_barrier_wait(init_barrier);
// 	worker->Run(); 
// };

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

	std::vector<Worker<Dtype>* > workers(N_DEVICE_PER_PROC, NULL);
	ncclUniqueId clique_id;
  NCCL_CHECK(ncclGetUniqueId(&clique_id) );
  pthread_barrier_t* process_barrier = new pthread_barrier_t;
  pthread_barrier_init(process_barrier, NULL, N_DEVICE_PER_PROC);
	for (int i = 0; i < N_DEVICE_PER_PROC; i++) {
		// TODO Jian: add solvers
		SyncCommConfig<Dtype> sync_config(gpu_ids[i], clique_id);
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

	for (int i = 0; i < N_DEVICE_PER_PROC; i++) {
		// std::thread* worker_thread = new std::thread(std::thread (InitAndRunWorker<Dtype>, &worker_init, workers[i] ) );
		std::thread* worker_thread = new std::thread(std::thread (&Worker<Dtype>::Run, std::ref(*workers[i] ) ) );
		worker_threads.push_back(worker_thread);
	}
	for (int i = 0; i < N_DEVICE_PER_PROC; i++)
		worker_threads[i]->join();

	for (int i = 0; i < N_DEVICE_PER_PROC; i++) {
		delete worker_threads[i];
		delete workers[i];
	}
	if (process_barrier != NULL) {
		pthread_barrier_destroy(process_barrier);
		process_barrier = NULL;
	}

	std::cout << "async worker test passed!" << std::endl;
}


int main(int argc, char** argv) {
	int rank;
	int size;
	MPI_Init(NULL, NULL);

	Train<float>();

	std::cout << "start finalize" << std::endl;
	
	MPI_Finalize();
}