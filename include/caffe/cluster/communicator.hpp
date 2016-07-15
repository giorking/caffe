#ifndef COMMUNICATOR_HPP_
#define COMMUNICATOR_HPP_

#include <cstddef>

/**
 * Derive or set differently for different communication  
 */
/* base communicator configuration class*/
template<typename Dtype>
class CommConfig {
public:
  CommConfig(int device_id, int machine_id, 
    int n_dev_clique_local, int clique_id, int clique_rank) :
    device_id_(device_id), 
    machine_id_(machine_id),
    n_dev_clique_local_(n_dev_clique_local),
    clique_id_(clique_id),
    clique_rank_(clique_rank),
    buffer_(NULL),
    left_gpu_buffer_(NULL), 
    right_gpu_buffer_(NULL),
    ncclComm_(NULL) {}
  CommConfig(const CommConfig<Dtype>& config) : 
    device_id_(config.device_id_), 
    machine_id_(config.machine_id_),
    n_dev_clique_local_(config.n_dev_clique_local_),
    clique_id_(config.clique_id_),
    clique_rank_(config.clique_rank_),
    buffer_(config.buffer_),
    left_gpu_buffer_(config.left_gpu_buffer_),
    right_gpu_buffer_(config.right_gpu_buffer_),
    ncclComm_(NULL) {}
  ~CommConfig() {}

  inline int GetDeviceId() { return device_id_; }
  inline int GetMachineId() { return machine_id_; } 
  Dtype** GetBufferPtrAddr() { return &buffer_; }
  Dtype* GetBuffer() {return buffer_; }
  void BufferMalloc(int64_t buf_size);
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
  int clique_id_;
  int clique_rank_;
  /**
   * buffer on the current, left and right machine for ring-base 
   * GPU communications.
   */
  Dtype* buffer_;
  Dtype* left_gpu_buffer_;
  Dtype* right_gpu_buffer_;
  int64_t buf_size_;
  int64_t left_buf_size_;
  int64_t right_buf_size_;
  ncclComm_t* nccl_comm_;
}; 

/*base communicator class*/
template<typename Dtype>
class Communicator {
public:
  Communicator(CommConfig<Dtype>& config) : config_(config) {}
  ~Communicator() {}
  /**
   * Building blocks for different synchronization setting
   * Group may include gpus on multiple nodes. We call the 
   * part of a group as the local clique on a node. 
   */
  virtual int SyncClique();
  virtual int SyncGroup();
  virtual int SyncIntraMachine();
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

friend Communicator<Dtype>;
};

#endif  // COMMUNICATOR_HPP_