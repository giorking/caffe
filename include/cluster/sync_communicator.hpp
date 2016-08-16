#ifndef COMMUNICATOR_HPP_
#define COMMUNICATOR_HPP_

#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <cassert>
#include "nccl/src/nccl.h"
#include "cluster/comm_utils.hpp"
#include "caffe/util/device_alternate.hpp"


/* Derive or set differently for different communication */
template<typename Dtype>
class SyncCommunicator;

template<typename Dtype>
class Worker;

template<typename Dtype>
class AsyncWorker;


/* base communicator configuration class*/
template<typename Dtype>
class SyncCommConfig {
public:
  // we use ncclComm_t** like we use ptr of ptr for cudaMalloc
  SyncCommConfig() {};
  SyncCommConfig(int device_id, ncclUniqueId clique_id) :
    device_id_(device_id), clique_id_(clique_id) {
    int n_device;
    int n_proc;
    CUDA_CHECK(cudaGetDeviceCount(&n_device) );
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    assert(n_proc % nProcPerGroup == 0);
    group_id_ = mpi_rank_ / nProcPerGroup;
    // current we assume each process control a clique
    n_dev_in_clique_= nDevicePerProc;
    clique_rank_ = device_id_ % nDevicePerProc;
    clique_root_rank_ = CLIQUE_ROOT_RANK;
    if (clique_rank_ == clique_root_rank_)
      is_clique_root_ = true;
    else
      is_clique_root_ = false;   
    if (mpi_rank_ % nProcPerGroup == 0 && is_clique_root_)
      is_group_root_ = true;
    else
      is_group_root_ = false;
  }
  SyncCommConfig(const SyncCommConfig<Dtype>& config) : 
    device_id_(config.device_id_), 
    group_id_(config.group_id_),
    n_dev_in_clique_(config.n_dev_in_clique_),
    clique_rank_(config.clique_rank_),
    clique_root_rank_(config.clique_root_rank_),
    clique_id_(config.clique_id_),
    is_clique_root_(config.is_clique_root_),
    is_group_root_(config.is_group_root_),
    mpi_rank_(config.mpi_rank_) {}
  // we use default assignment operator
  // operator=(const SyncCommConfig<Dtype>& config) {}

  /* access function*/
  // inline int64_t GetGpuBufferSize() { return gpu_buf_size_; }
  inline int GetDeviceId() { return device_id_; }
  // inline int GetMachineId() { return machine_id_; }
private:

  int device_id_;
  /* group id is initialized when initializing the communicator*/
  int group_id_;
  /**
   * A clique is the part of the group on the current local machine.
   * clique id specifies the GPU group in which current GPU is in.
   * clique_rank specifies the device rank in the clique.
   */
  ncclUniqueId clique_id_;
  /* define how many dev on the local machine is in the same clique*/
  int n_dev_in_clique_;
  int clique_rank_;
  int clique_root_rank_;
  bool is_clique_root_;
  //  specify the size of the mem buffer on GPU
  // int64_t gpu_buf_size_;

  /**
   * For communication between different nodes. 
   * Associated with MPI interfaces.
   */
  int mpi_rank_;
  // int mpi_root_rank_;
  bool is_group_root_;

friend class SyncCommunicator<Dtype>;
friend class Worker<Dtype>;
}; 

/*base communicator class*/
template<typename Dtype>
class SyncCommunicator {
public:
  SyncCommunicator() :
    nccl_comm_(NULL),
    stream_comm_(NULL),
    mpi_sync_comm_(NULL),
    gpu_buf_(NULL),
    mpi_sync_buf_(NULL),
    gpu_buf_size_(0),
    mpi_sync_buf_size_(0),
    process_barrier_(NULL) {}
  SyncCommunicator(const SyncCommConfig<Dtype>& config, 
    pthread_barrier_t* process_barrier) : 
    config_(config),
    nccl_comm_(NULL),
    stream_comm_(NULL),
    mpi_sync_comm_(NULL),
    gpu_buf_(NULL),
    mpi_sync_buf_(NULL),
    gpu_buf_size_(0),
    mpi_sync_buf_size_(0),
    process_barrier_(process_barrier) {}
  // we use default assignment communicator
  SyncCommunicator(const SyncCommunicator<Dtype>& comm) :
    SyncCommunicator<Dtype> (comm.config_, comm.process_barrier_) {}
  ~SyncCommunicator() {
    // if (gpu_buf_ != NULL)
    //   CUDA_CHECK(cudaFree(gpu_buf_) );
    gpu_buf_ = NULL;
    if (mpi_sync_buf_ != NULL)
      delete mpi_sync_buf_;
    if (nccl_comm_ != NULL) {
      ncclCommDestroy(*nccl_comm_);
      nccl_comm_ = NULL;
    }
    if (stream_comm_ != NULL) {
      CUDA_CHECK(cudaStreamDestroy(*stream_comm_) );
      stream_comm_ = NULL;
    }
  }


  void Init(int64_t buf_size, Dtype* external_gpu_buf);
  /**
  * Building blocks for different synchronization setting
  * Group may include gpus on multiple nodes. We call the 
  * part of a group as the local clique on a node. 
  */
  virtual void CliqueReduce();
  virtual void CliqueBroadcast();
  virtual void InterMachineAllReduce();
  virtual void SyncGroup(bool do_broadcast);
  
  void ProcessBarrierWait() { pthread_barrier_wait(process_barrier_); };
  /* access function */
  inline Dtype* GetGpuBuffer() { return gpu_buf_; }
  inline Dtype* GetMpiSyncBuffer() { return mpi_sync_buf_; }
  inline int64_t GetMpiSyncBufferSize() { return mpi_sync_buf_size_; }
  inline bool IsCliqueRoot() { return config_.is_clique_root_; }
  
private:  
  // configuration 
  SyncCommConfig<Dtype> config_;
  /**
   * communicator for local clique
   * stream_comm_ is used by nccl_comm_
   */
  ncclComm_t* nccl_comm_;
  cudaStream_t* stream_comm_;

  // communicator for intra/inter-group multi-node communication
  MPI_Comm* mpi_sync_comm_;

  // buffer for intra-node gpu communication. We only attach gpu memory, never allocate here
  int64_t gpu_buf_size_;
  Dtype* gpu_buf_;

  // inter-node intra-group communication using mpi 
  int64_t mpi_sync_buf_size_;
  Dtype* mpi_sync_buf_;

  /**
   * copy barrier from external code, regularize behavior of workers
   * in the same process.
   */
  pthread_barrier_t* process_barrier_;

friend class Worker<Dtype>;
friend class AsyncWorker<Dtype>;
};

#endif  // COMMUNICATOR_HPP_