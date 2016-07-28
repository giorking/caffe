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


/* for definition of NULL */
#include <cstddef>
#include <mpi.h>
#include <pthread.h>
/* for definition of CHECK_EQ*/
#include "glog/logging.h"
#include "nccl/src/nccl.h"

#ifndef N_PROC_PER_GROUP
#define N_PROC_PER_GROUP 4
#endif

#ifndef GROUP_ROOT_COMM_ID
#define GROUP_ROOT_COMM_ID 0
#endif

#ifndef NON_GROUP_ROOT_COMM_ID
#define NON_GROUP_ROOT_COMM_ID 0
#endif

#ifndef CLIQUE_ROOT_RANK
#define CLIQUE_ROOT_RANK 0
#endif

enum MsgType { ASYNC_MSG = 0, SYNC_MSG = 1};

/* Compile time mapping from typename Dtype to ncclDataType_t */
template <typename Dtype>
struct DtypeToNCCLDtype {};

template<> struct DtypeToNCCLDtype<float> { 
  const static ncclDataType_t type = ncclFloat; 
};

template<> struct DtypeToNCCLDtype<double> { 
  const static ncclDataType_t type = ncclDouble; 
};


/* Compile time mapping from typename Dtype to MPI_Datatype*/
template <typename Dtype>
struct DtypeToMPIDtype {};

template<> struct DtypeToMPIDtype<float> {
  const static MPI_Datatype type = MPI_FLOAT;
};
template<> struct DtypeToMPIDtype<double> {
  const static MPI_Datatype type = MPI_DOUBLE;
};


#endif // end of COMM_UTILS_HPP_