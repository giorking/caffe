/**
 * Derive or set differently for different communication  
 */
/* base communicator configuration class*/
template<typename Dtype>
class CommConfig {
public:
  CommConfig(device_id, machine_id) :
    device_id_(device_id), 
    machine_id_(machine_id),
    buffer_(NULL),
    left_gpu_buffer_(NULL), 
    right_gpu_buffer_(NULL) {}
  CommConfig(CommConfig& config) : 
    device_id_(config.device_id_), machine_id_(config.machine_id_),
    left_gpu_buffer_(config.left_gpu_buffer_),
    right_gpu_buffer_(config.right_gpu_buffer_) {}
  ~CommConfig() {}
  inline int GetDeviceId() { return device_id; }
  inline int GetMachineId() { return machine_id; } 
  inline Dtype* GetBuffer() { return buffer_; }
  inline Dtype* SetLeftGpuBuffer() { return left_gpu_buffer_; }
  inline Dtype* SetRightGpuBuffer() { return right_gpu_buffer_; }
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
  /**
   * buffer on the current, left and right machine for ring-base 
   * GPU communications.
   */
  Dtype* buffer_;
  Dtype* left_gpu_buffer_;
  Dtype* right_gpu_buffer_;
}; 

/*base communicator class*/
template<typename Dtype>
class Communicator {
public:
  Communicator(CommConfig<Dtype>& config) : config_(config) {}
  ~Communicator() {}
  /* Building blocks for different synchronization setting */
  virtual int SyncGroup();
  virtual int SyncIntraMachine();
  virtual int SyncInterMachine();
  /* communication primitives */
  virtual int SendToPointInterMachine();
  virtual int SendToPointIntraMachine();
  virtual int AllReduceInterMachine();
  virtual int AllReduceIntraMachine();
  virtual int BroadCastIntraMachine();
  /* Behaviors in computing process */
  virtual int OnStart();
  virtual int OnGradientReady();
private:  
  /* configuration */
  CommConfig config_;
};