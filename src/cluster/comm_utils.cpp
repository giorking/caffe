#include <vector>
#include <thread>
#include "cluster/comm_utils.hpp"
#include "cluster/worker.hpp"


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

pthread_mutex_t globalInitMutex;


void GetGpuIds(std::vector<int>& gpu_ids) {
  int n_gpus = 0;
  // CUDA_CHECK(cudaGetDeviceCount(&n_gpus) );
  cudaGetDeviceCount(&n_gpus);
  gpu_ids.clear();
  for (int i = 0; i < n_gpus; i++)
    gpu_ids.push_back(i);
  return;
}


void ParseSysConfigArg(int argc, char** argv) {
  if ( (argc <= 1) || (argv[argc-1] == NULL) || (argv[argc-1][0] == '-') )
    std::cerr << "No argument provided!" << std::endl;
  static struct option long_options[] = {
    {"n_device_per_proc", required_argument, 0, 'a'},
    {"n_machine_per_group", required_argument, 0, 'b'},
    {"n_proc_per_machine", required_argument, 0, 'c'},
    {0, 0, 0, 0}
  };
  int c;
  while (1) {
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long (argc, argv, "a:b:c:",
                     long_options, &option_index);

    // Detect the end of the options. 
    if (c == -1)
    	break;
    switch (c) {
    	case 'a':
    		nDevicePerProc = std::atoi(optarg);
    	case 'b':
    		nMachinePerGroup = std::atoi(optarg);
    	case 'c':
    		nProcPerMachine = std::atoi(optarg);
    	// case '?':
    	// 	break;
    	// 	this function only deals with argument related to sys configuration
    	default:
    		break;
    		// std::cerr << "wrong argument!" << std::endl;
    		// std::exit(1);
    }
	}
	nProcPerGroup = nProcPerMachine * nMachinePerGroup;
}


namespace caffe {


template <typename Dtype>
void RunSyncWorkers(caffe::shared_ptr<caffe::Solver<Dtype> > root_solver) {
	int mpi_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	int mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	std::vector<int> gpu_ids;
	GetGpuIds(gpu_ids);

	// check for macro settings from comm_utils.hpp
	nProcPerGroup = nProcPerMachine * nMachinePerGroup;
	if (gpu_ids.size() != nProcPerMachine * nDevicePerProc) {
		std::cout << "Not enough GPU on a machine!" << std::endl;
		// std::exit(1);
	}
	if (mpi_size % nProcPerMachine) {
		std::cout << "Processes can not be equaly distributed to machines!" << std::endl;
		// std::exit(1);
	}
	if (mpi_size / nProcPerGroup != 1) {
		std::cout << "Need a single group to run sync worker!" << std::endl;
		// std::exit(1);
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
		std::thread* worker_thread = new std::thread(&Worker<Dtype>::Run, workers[i], root_solver);
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
}


// template void RunSyncWorkers<float>(const caffe::SolverParameter& solver_param);
// template void RunSyncWorkers<double>(const caffe::SolverParameter& solver_param);

template void RunSyncWorkers<float>(caffe::shared_ptr<caffe::Solver<float> > root_solver);
template void RunSyncWorkers<double>(caffe::shared_ptr<caffe::Solver<double> > root_solver);

}