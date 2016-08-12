#include <vector>
#include "cluster/comm_utils.hpp"


void GetGpuIds(std::vector<int>& gpu_ids) {
  int n_gpus = 0;
  // CUDA_CHECK(cudaGetDeviceCount(&n_gpus) );
  cudaGetDeviceCount(&n_gpus);
  gpu_ids.clear();
  for (int i = 0; i < n_gpus; i++)
    gpu_ids.push_back(i);
  return;
}


void ParseCmdArg(int argc, char** argv) {
  if ( (argc <= 1) || (argv[argc-1] == NULL) || (argv[argc-1][0] == '-') )
    std::cerr << "No argument provided!" << std::endl;
  static struct option long_options[] = {
    {"n_device_per_proc", required_argument, 0, 'a'},
    {"n_machine_per_group", required_argument,      0, 'b'},
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
    	case '?':
    		break;
    	default:
    		std::cerr << "wrong argument!" << std::endl;
    		std::exit(1);
    }
	}
	nProcPerGroup = nProcPerMachine * nMachinePerGroup;
}
