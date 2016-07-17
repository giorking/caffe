#include <iostream>
#include "cluster/communicator.hpp"
#include "nccl/src/nccl.h"


template <typename Dtype>
void CommConfig<Dtype>::SetGpuBuffer(Dtype* buffer, int64_t buf_size) {
  buf_size_ = buf_size;
  gpu_buffer_ = buffer;
}


template <typename Dtype>
void CommConfig<Dtype>::SetLeftGpuBuffer(Dtype* buffer, int64_t buf_size) {
  if (buffer == NULL) {
    std::cout << "left_gpu_buffer set to NULL!" << std::endl;
    exit(1);
  }
  left_gpu_buffer_ = buffer;
}


template <typename Dtype>
void CommConfig<Dtype>::SetRightGpuBuffer(Dtype* buffer, int64_t buf_size) {
  if (buffer == NULL) {
    std::cout << "right_gpu_buffer set to NULL!" << std::endl;
    exit(1);
  }
  right_gpu_buffer_ = buffer;
}


template <typename Dtype>
void Communicator<Dtype>::InitCommConfig() {
  /* init nccl communicator with global information */
  NCCL_CHECK(ncclCommInitRank(config_.nccl_comm_, 
    config_.n_dev_clique_local_, config_.clique_id_, 
    config_.clique_rank_) );
  CUDA_CHECK(cudaStreamCreate(config_.stream_comm_) );
}


template <typename Dtype>
void Communicator<Dtype>::CliqueReduce() {
  ncclDataType_t type = DtypeToNCCLDType<Dtype>::type;
  if (config_.clique_rank_ == config_.clique_root_rank_)
    NCCL_CHECK(ncclReduce( (const void*)config_.gpu_buffer_, 
      (void*)config_.gpu_buffer_, config_.buf_size_, 
      type, ncclSum, config_.clique_root_rank_, 
      *(config_.nccl_comm_), *(config_.stream_comm_) ) );
  else
    NCCL_CHECK(ncclReduce( (const void*)config_.gpu_buffer_, NULL,
      config_.buf_size_, type, ncclSum, config_.clique_root_rank_, 
      *(config_.nccl_comm_), *(config_.stream_comm_) ) ); 
}


template <typename Dtype>
void Communicator<Dtype>::CliqueBroadcast() {
  ncclDataType_t type = DtypeToNCCLDType<Dtype>::type;
  if (config_.clique_rank_ == config_.clique_root_rank_)
    NCCL_CHECK(ncclBcast( (void*)config_.gpu_buffer_, 
      config_.buf_size_, type, config_.clique_root_rank_, 
      *(config_.nccl_comm_), *(config_.stream_comm_) ) );
  else
    NCCL_CHECK(ncclBcast( (void*)config_.gpu_buffer_, 
      config_.buf_size_, type, config_.clique_root_rank_, 
      *(config_.nccl_comm_), *(config_.stream_comm_) ) ); 
}


template <typename Dtype>
void Communicator<Dtype>::InterMachineAllReduce() {

}


template <typename Dtype>
void Communicator<Dtype>::SyncGroup() {
  /**
   * recvbuf may be NULL for non-clique-lead GPU
   * We first use reduce to gether within clique
   * Then reduce-all with MPI
   * Then broadcast with clique from clique lead
   */
  /* reduce within clique */
  CliqueReduce();

  /* TODO Jian: inter ndoe communication */
  InterMachineAllReduce();
  
  /* TODO Jian: broadcast within clique */
  CliqueBroadcast();
}

// template <typename Dtype>
// int Communicator<Dtype>::SyncIntraMachine() {

// } 


template class CommConfig<float>;
template class CommConfig<double>;
template class Communicator<float>;
template class Communicator<double>; 