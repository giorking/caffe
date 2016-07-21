#ifndef COMMUNICATOR_HPP_
#define COMMUNICATOR_HPP_

#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nccl/src/nccl.h"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/math_functions.hpp"

#ifndef N_PROC_PER_GROUP
#define N_PROC_PER_GROUP 0
#endif

#ifndef GRUOP_ROOT_COMM_ID
#define GRUOP_ROOT_COMM_ID 0
#endif

#ifndef NON_GROUP_ROOT_COMM_ID
#define NON_GROUP_ROOT_COMM_ID 0
#endif

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


/* Derive or set differently for different communication */
template<typename Dtype>
class Communicator;

/* base communicator configuration class*/
template<typename Dtype>
class CommConfig {
public:
  CommConfig(int device_id, int machine_id, int group_id, 
    int n_dev_clique_local, ncclUniqueId clique_id, 
    bool is_clique_root, bool is_group_root, int64_t gpu_buf_size,
    int64_t mpi_diff_buf_size, int64_t mpi_model_buf_size) :
    device_id_(device_id), 
    machine_id_(machine_id),
    // group_id_(group_id),
    n_dev_clique_local_(n_dev_clique_local),
    clique_id_(clique_id),
    is_clique_root_(is_clique_root),
    is_group_root_(is_group_root),
    gpu_buf_size_(gpu_buf_size),
    mpi_diff_buf_size_(mpi_diff_buf_size),
    mpi_model_buf_size_(mpi_model_buf_size) {}
  CommConfig(const CommConfig<Dtype>& config) : 
    device_id_(config.device_id_), 
    machine_id_(config.machine_id_),
    // group_id_(config.group_id_),
    n_dev_clique_local_(config.n_dev_clique_local_),
    clique_id_(config.clique_id_),
    is_clique_root_(config.is_clique_root_),
    is_group_root_(config.is_group_root_),
    gpu_buf_size_(config.gpu_buf_size_),
    mpi_diff_buf_size_(config.mpi_diff_buf_size_),
    mpi_model_buf_size_(config.mpi_model_buf_size_) {}

  /* access function*/
  inline int64_t GetGpuBufferSize() { return gpu_buf_size_; }
  inline int GetDeviceId() { return device_id_; }
  inline int GetMachineId() { return machine_id_; }
private:

  int device_id_;
  int machine_id_;
  int group_id_;
  /**
   * A clique is the part of the group on the current local machine.
   * clique id specifies the GPU group in which current GPU is in.
   * clique_rank specifies the device rank in the clique.
   */
  ncclUniqueId clique_id_;
  /* define how many dev on the local machine is in the same clique*/
  int n_dev_clique_local_;

  /* specify the size of the mem buffer on GPU*/
  int64_t gpu_buf_size_;

  /**
   * For communication between different nodes. 
   * Associated with MPI interfaces.
   */
  int mpi_rank_;
  // int mpi_root_rank_;
  bool is_clique_root_;
  bool is_group_root_;
  
  int64_t mpi_diff_buf_size_;
  int64_t mpi_model_buf_size_;


friend class Communicator<Dtype>;
}; 

/*base communicator class*/
template<typename Dtype>
class Communicator {
public:
  Communicator(CommConfig<Dtype>& config);
  ~Communicator() {
    if (gpu_buf_ != NULL)
      CUDA_CHECK(cudaFree(gpu_buf_) );
    if (mpi_model_buf_ != NULL)
      delete mpi_model_buf_;
    if (mpi_diff_buf_ != NULL)
      delete mpi_diff_buf_;
    if (nccl_comm_ != NULL)
      ncclCommDestroy(*nccl_comm_);
    if (stream_comm_ != NULL)
      CUDA_CHECK(cudaStreamDestroy(*stream_comm_) );
  }
  void InitComm();
   /**
   * Building blocks for different synchronization setting
   * Group may include gpus on multiple nodes. We call the 
   * part of a group as the local clique on a node. 
   */
  virtual void CliqueReduce();
  virtual void CliqueBroadcast();
  virtual void InterMachineAllReduce();
  virtual void SyncGroup();
  // virtual int SyncInterMachine();
  // /* communication primitives */
  // virtual int SendToPointInterMachine();
  // virtual int SendToPointIntraMachine();
  // virtual int AllReduceInterMachine();
  // virtual int AllReduceIntraMachine();
  // virtual int BroadCastIntraMachine();
  // /* Behaviors in computing process */
  // virtual int OnStart();
  // virtual int OnGradientReady();
  
  /* access function */
  inline Dtype* GetGpuBuffer() { return gpu_buf_; }
  
private:  
  /* configuration */
  CommConfig<Dtype> config_;
  /**
   * communicator for local clique
   * stream_comm_ is used by nccl_comm_
   */
  ncclComm_t* nccl_comm_;
  cudaStream_t* stream_comm_;

  /* communicator for intra/inter-group multi-node communication*/
  MPI_Comm* mpi_intra_group_comm_;
  MPI_Comm* mpi_inter_group_comm_;

  /* buffer for intra-node gpu communication */
  Dtype* gpu_buf_;

  /* inter-node communication using mpi */
  Dtype* mpi_diff_buf_;
  Dtype* mpi_model_buf_;
};

/* Compile time mapping from typename Dtype to ncclDataType_t */
template <typename Dtype>
struct DtypeToNCCLDType {};

template<> struct DtypeToNCCLDType<float> { const static ncclDataType_t type = ncclFloat; };
template<> struct DtypeToNCCLDType<double> { const static ncclDataType_t type = ncclDouble; };

#endif  // COMMUNICATOR_HPP_