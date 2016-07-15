#include <iostream>
#include "communicator.hpp"
#include "nccl.h"


template <typename Dtype>
void CommConfig<Dtype>::BufferMalloc(int64_t buf_size) {
  buf_size_ = buf_size;
  CUDA_CHECK(cudaMalloc(buffer_, sizeof(Dtype) * buf_size) );
}


template <typename Dtype>
void CommConfig<Dtype>::SetLeftGpuBuffer(Dtype* buffer) {
  if (buffer == NULL) {
    std::cout << "left_gpu_buffer set to NULL!" << std::endl;
    exit(1);
  }
  left_gpu_buffer_ = buffer;
}


template <typename Dtype>
void CommConfig<Dtype>::SetRightGpuBuffer(Dtype* buffer) {
  if (buffer == NULL) {
    std::cout << "right_gpu_buffer set to NULL!" << std::endl;
    exit(1);
  }
  right_gpu_buffer_ = buffer;
}


template <typename Dtype>
void Communicator<Dtype>::InitCliqueLocalComm() {
  /* init nccl communicator with global information */
  NCCLCHECK(ncclCommInitRank(config_.nccl_comm_, 
    config_.n_dev_clique_local_, config_.clique_id, 
    config_.clique_rank_) );
}


template <typename Dtype>
int Communicator<Dtype>::SyncClique() {
  
}

template <typename Dtype>
int Communicator<Dtype>::SyncGroup() {
  /**
   * recvbuf may be NULL for non-clique-lead GPU
   * We first use reduce to gether within clique
   * Then reduce-all with MPI
   * Then broadcast with clique from clique lead
   */
  /* reduce within clique */
  if (config_.clique_rank == 0)
    NCCLCHECK(ncclReduce( (void*)config_.buffer_, (void*)config_.buffer_,
      buf_size_, Dtype, root, comm, stream) );
  else
    NCCLCHECK(ncclReduce( (void*)config_.buffer_, NULL,
      buf_size_, Dtype, root, comm, stream) );

  /* TODO Jian: inter ndoe communication */
  
  /* TODO Jian: broadcast within clique */

}

template <typename Dtype>
int Communicator<Dtype>::SyncIntraMachine() {

} 


template class CommConfig<float>;
template class CommConfig<double>;
template class Communicator<float>;
template class Communicator<double>; 