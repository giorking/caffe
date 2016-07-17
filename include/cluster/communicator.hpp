#ifndef COMMUNICATOR_HPP_
#define COMMUNICATOR_HPP_

#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include "nccl/src/nccl.h"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/math_functions.hpp"


/**
 * Derive or set differently for different communication  
 */
template<typename Dtype>
class Communicator;

/* base communicator configuration class*/
template<typename Dtype>
class CommConfig {
public:
  CommConfig(int device_id, int machine_id, 
    int n_dev_clique_local, ncclUniqueId clique_id, int clique_rank) :
    device_id_(device_id), 
    machine_id_(machine_id),
    n_dev_clique_local_(n_dev_clique_local),
    clique_id_(clique_id),
    clique_rank_(clique_rank),
    clique_root_rank_(0),
    gpu_buffer_(NULL),
    left_gpu_buffer_(NULL), 
    right_gpu_buffer_(NULL),
    nccl_comm_(NULL),
    stream_comm_(NULL) {}
  CommConfig(const CommConfig<Dtype>& config) : 
    device_id_(config.device_id_), 
    machine_id_(config.machine_id_),
    n_dev_clique_local_(config.n_dev_clique_local_),
    clique_id_(config.clique_id_),
    clique_rank_(config.clique_rank_),
    clique_root_rank_(config.clique_root_rank_),
    gpu_buffer_(config.gpu_buffer_),
    left_gpu_buffer_(config.left_gpu_buffer_),
    right_gpu_buffer_(config.right_gpu_buffer_),
    nccl_comm_(NULL),
    stream_comm_(NULL) {}
  ~CommConfig() {
    if (nccl_comm_ != NULL)
      ncclCommDestroy(*nccl_comm_);
    if (stream_comm_ != NULL)
      CUDA_CHECK(cudaStreamDestroy(*stream_comm_) );
  }

  inline int GetDeviceId() { return device_id_; }
  inline int GetMachineId() { return machine_id_; } 
  Dtype** GetGpuBufferPtrAddr() { return &gpu_buffer_; }
  Dtype* GetGpuBuffer() {return gpu_buffer_; }
  int64_t GetGpuBufferSize() {return buf_size_; }
  /**
   * CommConfig and Communicator only operate on existing
   * buffers (buffer_, left/right_gpu_buffer)
   */
  void SetGpuBuffer(Dtype* buffer, int64_t buf_size);
  void SetLeftGpuBuffer(Dtype* buffer, int64_t buf_size);
  void SetRightGpuBuffer(Dtype* buffer, int64_t buf_size);  
private:
  /**
   * Communicator.SyncIntraMachine uses sync_gpu_ids_ to sync
   * GPU on the same machine. 
   */
  // vector<int64_t> sync_gpu_ids;
  /**
   * Add a MPI barrier, Communicator.SyncInterMachine will
   * use it to wait for synchronized workers in different machines
   */
  int device_id_;
  int machine_id_;
  /* define how many dev on the local machine is in the same clique*/
  int n_dev_clique_local_;
  /**
   * clique id specifies the GPU group in which current GPU is in.
   * clique_rank specifies the device rank in the clique.
   */
  // int clique_id_;
  ncclUniqueId clique_id_;
  int clique_rank_;
  int clique_root_rank_;
  /**
   * buffer on the current, left and right machine for ring-base 
   * GPU communications.
   */
  Dtype* gpu_buffer_;
  Dtype* left_gpu_buffer_;
  Dtype* right_gpu_buffer_;
  int64_t buf_size_;
  int64_t left_buf_size_;
  int64_t right_buf_size_;
  ncclComm_t* nccl_comm_;
  /* stream for communication in local clique */
  cudaStream_t* stream_comm_;

friend class Communicator<Dtype>;
}; 

/*base communicator class*/
template<typename Dtype>
class Communicator {
public:
  Communicator(CommConfig<Dtype>& config) : config_(config) {}
  ~Communicator() {}
  void InitCommConfig();
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
private:  
  /* configuration */
  CommConfig<Dtype> config_;
};

/* Compile time mapping from typename Dtype to ncclDataType_t */
template <typename Dtype>
struct DtypeToNCCLDType {};

// template<> struct DtypeToNCCLDType<float> { enum { TYPE = ncclFloat}; };
// template<> struct DtypeToNCCLDType<double> { enum { TYPE = ncclDouble}; };

template<> struct DtypeToNCCLDType<float> { const static ncclDataType_t type = ncclFloat; };
template<> struct DtypeToNCCLDType<double> { const static ncclDataType_t type = ncclDouble; };

#endif  // COMMUNICATOR_HPP_