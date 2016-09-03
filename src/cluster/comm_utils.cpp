#include <vector>
#include <thread>
#include "cluster/comm_utils.hpp"
// #include "cluster/worker.hpp"


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


const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}


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
