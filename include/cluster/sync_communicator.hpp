#ifndef COMMUNICATOR_HPP_
#define COMMUNICATOR_HPP_

#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <cassert>
#include "nccl/src/nccl.h"
#include "cluster/comm_utils.hpp"
// #include "caffe/util/device_alternate.hpp"


namespace caffe {

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
  SyncCommConfig()  {};
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
    clique_id_(config.clique_id_),
    n_dev_in_clique_(config.n_dev_in_clique_),
    clique_rank_(config.clique_rank_),
    clique_root_rank_(config.clique_root_rank_),
    is_clique_root_(config.is_clique_root_),
    mpi_rank_(config.mpi_rank_),
    is_group_root_(config.is_group_root_) {}

  /* access function*/
  // inline int64_t GetGpuBufferSize() { return gpu_buf_size_; }
  inline int GetDeviceId() { return device_id_; }
  inline bool IsCliqueRoot() { return is_clique_root_; }
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


/**
 * base communicator class.
 * 1. If we want to directly perform MPI operation on specific 
 * buffers, the external buffer should be given to gpu_buf_
 * if we use GPU direct MPI and we do not need cpu_buf_. 
 * Otherwise, the external buffer should be given as cpu_buf_.
 * Make sure the cpu_buf_ is attached to a communicator whose
 * this->IsCliqueRoot() == true.
 * 
 * 2. For MPI all reduce the required tmp_buf_size is 
 * block_size * (n_proc / 2 - 1) + buf_size - (n_proc - 1) * block_size
 * with block_size = buf_size / n_proc. It is enough for the halving and
 * doubleing algorithm for reduce scatter, all gather and all reduce.
 *
 * 3. If we want to first do clique reduce and clique broadcast over
 * multiple gpus on the same machine first, give the gpu buf to gpu_buf_.
 * When not using GPU direct MPI, cpu_buf_ should be set for the clique
 * root gpu so that data can be accessed via MPI. 
 * Other cards do not need cpu_buf_. Otherwise, no cpu_buf_ is needed.
 * Note if multiple gpus(equivalently multiple threads) are involved,
 * We need to give process_barrier when we instantiate communicator.
 */
template<typename Dtype>
class SyncCommunicator {
public:
  SyncCommunicator() :
    config_(),
    nccl_comm_(NULL),
    stream_comm_(NULL),
    mpi_sync_comm_(NULL),
    gpu_buf_(NULL),
    gpu_buf_size_(0),
    gpu_buf_tmp_(NULL),
    gpu_buf_tmp_size_(0),
    cpu_buf_(NULL),
    cpu_buf_size_(0),
    cpu_buf_tmp_(NULL),
    cpu_buf_tmp_size_(0),
    process_barrier_(NULL) {}
  SyncCommunicator(const SyncCommConfig<Dtype>& config, 
    pthread_barrier_t* process_barrier = NULL) : 
    config_(config),
    nccl_comm_(NULL),
    stream_comm_(NULL),
    mpi_sync_comm_(NULL),
    gpu_buf_(NULL),
    gpu_buf_size_(0),
    gpu_buf_tmp_(NULL),
    gpu_buf_tmp_size_(0),
    cpu_buf_(NULL),
    cpu_buf_size_(0),
    cpu_buf_tmp_(NULL),
    cpu_buf_tmp_size_(0),
    process_barrier_(process_barrier) {}
  // we use default assignment communicator
  SyncCommunicator(const SyncCommunicator<Dtype>& comm) :
    SyncCommunicator<Dtype> (comm.config_, comm.process_barrier_) {}
  ~SyncCommunicator() {
    // if (gpu_buf_ != NULL)
    //   CUDA_CHECK(cudaFree(gpu_buf_) );
    // we only allow external buf to be hooked onto gpu/cpu_buf
    gpu_buf_ = NULL;
    cpu_buf_ = NULL;
    if (gpu_buf_tmp_ != NULL)
      CUDA_CHECK(cudaFree(gpu_buf_tmp_) );
    if (cpu_buf_tmp_ != NULL)
      free(cpu_buf_tmp_);
    // if (cpu_buf_ != NULL)
    //   CUDA_CHECK(cudaFreeHost(cpu_buf_) );
    if (nccl_comm_ != NULL) {
      ncclCommDestroy(*nccl_comm_);
      nccl_comm_ = NULL;
    }
    if (stream_comm_ != NULL) {
      CUDA_CHECK(cudaStreamDestroy(*stream_comm_) );
      stream_comm_ = NULL;
    }
    CUBLAS_CHECK(cublasDestroy(cublas_handle_) );
  }

  /**
   * external_buf must be gpu memory if GPU_DIRECT_MPI
   * it must be cpu memory if GPU_DIRECT_MPI is not
   * defined in cluster/comm_utils.hpp. The same for external_buf_tmp.
   * if using GPU Direct MPI, external_cpu_buf should be NULL; otherwise
   * it should be allocated outside and given here.
   * external_gpu_buf is necessary for CliqueReduce and CliqueBroadcast
   */
  void Init(int64_t buf_size, Dtype* external_cpu_buf, 
    Dtype* external_gpu_buf, int64_t tmp_buf_size);
  /**
  * Building blocks for different synchronization setting
  * Group may include gpus on multiple nodes. We call the 
  * part of a group as the local clique on a node. 
  */
  virtual void CliqueReduce();
  virtual void CliqueBroadcast();
  /**
   * the reduce scatter and all gather are designed for large blocks.
   * They use recursive doubling/halving algorithm. 
   */
  virtual void InterMachineAllReduce();
  virtual void InterMachineReduceScatter();
  virtual void InterMachineAllGather();
  virtual void SyncGroup(bool do_broadcast);
  
  void ProcessBarrierWait() { pthread_barrier_wait(process_barrier_); };
  /* access function */
  inline Dtype* GetGpuBuffer() { return gpu_buf_; }
  inline Dtype* GetCpuBuffer() { return cpu_buf_; }
  inline Dtype* GetMpiSyncBuffer() { return cpu_buf_; }
  inline int64_t GetMpiSyncBufferSize() { return cpu_buf_size_; }
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
  Dtype* gpu_buf_;
  int64_t gpu_buf_size_;
  // temporary cpu memory for communication purpose, utilized when GPU_DIRECT_MPI in comm_utils.hpp
  // it is initialized in Init function with given tmp buffer size.
  Dtype* gpu_buf_tmp_;
  int64_t gpu_buf_tmp_size_;

  // inter-node intra-group communication using mpi 
  Dtype* cpu_buf_;
  int64_t cpu_buf_size_;
  // temporary cpu memory for communication purpose, utilized when not GPU_DIRECT_MPI in comm_utils.hpp
  // it is initialized in Init function with given tmp buffer size.
  Dtype* cpu_buf_tmp_;
  int64_t cpu_buf_tmp_size_;

  /**
   * copy barrier from external code, regularize behavior of workers
   * in the same process.
   */
  pthread_barrier_t* process_barrier_;

  // used for division on gpu
  cublasHandle_t cublas_handle_;

friend class Worker<Dtype>;
friend class AsyncWorker<Dtype>;
};


}


#endif  // COMMUNICATOR_HPP_