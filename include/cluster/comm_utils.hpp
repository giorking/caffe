#ifndef COMM_UTILS_HPP_
#define COMM_UTILS_HPP_

/**
 * In our communication protocal, each process spawns multiple threads.
 * Each thread controls one GPU device. A group refers to multiple processes
 * working synchronously. Each group has a root, which is the only thread in
 * the group dealing with inter-group communication (asynchrony). A group 
 * consists of multiple clique. A Clique is the part of group on 
 * a single machine. More intitively, a process corresponds to a
 * clique. The root thread of the clique deals with communication among
 * multiple node in the group. In our protocol, we assume if a thread is
 * group_root, it is also guaranteed to be a clique root. This avoids addtional
 * communication between inter and intra operations.
 */

// for definition of NULL 
#include <cstddef>
#include <mpi.h>
#include <pthread.h>
#include <getopt.h>
#include <iostream>
#include <cublas_v2.h>
// for definition of CHECK_EQ
#include <glog/logging.h>
// #include <gflags/gflags.h>
#include "nccl/src/nccl.h"

// #include "caffe/solver.hpp"


// #ifndef HIDE_DATA_FEED
// #define HIDE_DATA_FEED
// #endif


// set if MPI allow gpu memory direct access
// #ifndef GPU_DIRECT_MPI
// #define GPU_DIRECT_MPI
// #endif


// related to gflags
DECLARE_string(snapshot);
DECLARE_string(weights);


#ifndef CLIQUE_ROOT_RANK
#define CLIQUE_ROOT_RANK 0
#endif

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)


const char* cublasGetErrorString(cublasStatus_t status);

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
   		<< cublasGetErrorString(status); \
  } while (0)


// distinguish message within sync group or asynchronized inter-groups 
enum MsgType { ASYNC_MSG = 0, SYNC_MSG = 1};

// global variables
// the number of processes in a synchronized group.
extern int nProcPerGroup;

// the number of machines in a synchronized group.
extern int nMachinePerGroup;

/**
 * the number of process on a single machine. Derived from
 * nProcPerGroup and nMachinePerGroup.
 */
extern int nProcPerMachine;

// the number of gpu cards each process has. 
extern int nDevicePerProc;

// a global mutex for initilization coordination
extern pthread_mutex_t globalInitMutex;


// Compile time mapping from typename Dtype to ncclDataType_t 
template <typename Dtype>
struct DtypeToNCCLDtype {};

template<> struct DtypeToNCCLDtype<float> { 
  const static ncclDataType_t type = ncclFloat; 
};

template<> struct DtypeToNCCLDtype<double> { 
  const static ncclDataType_t type = ncclDouble; 
};


// Compile time mapping from typename Dtype to MPI_Datatype
template <typename Dtype>
struct DtypeToMPIDtype {};

template<> struct DtypeToMPIDtype<float> {
  const static MPI_Datatype type = MPI_FLOAT;
};
template<> struct DtypeToMPIDtype<double> {
  const static MPI_Datatype type = MPI_DOUBLE;
};


// get the gpu ids accessible from a single process
void GetGpuIds(std::vector<int>& gpu_ids);

// parse system setting argument
void ParseSysConfigArg(int argc, char** argv);


// namespace caffe {

// // setup sync workers 
// template <typename Dtype>
// void RunSyncWorkers(caffe::shared_ptr<caffe::Solver<Dtype> > root_solver);

// // // setup async workers
// // template <typename Dtype>


// }

// y = y +  a * x 
inline cublasStatus_t cublasAxpy(cublasHandle_t handle, int n, 
  const float a, const float* x, float* y) {
  return cublasSaxpy(handle, n, &a, x, 1, y, 1);
}

inline cublasStatus_t cublasAxpy(cublasHandle_t handle, int n, 
  const double a, const double* x, double* y) {
  return cublasDaxpy(handle, n, &a, x, 1, y, 1);
}

// y = y * a
inline cublasStatus_t cublasScal(cublasHandle_t handle, int n,
  const float* a, float* x) {
  return cublasSscal(handle, n, a, x, 1);
}

inline cublasStatus_t cublasScal(cublasHandle_t handle, int n,
  const double* a, double* x) {
  return cublasDscal(handle, n, a, x, 1);
}


#endif // end of COMM_UTILS_HPP_