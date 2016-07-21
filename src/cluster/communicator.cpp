#include <iostream>
#include "cluster/communicator.hpp"
#include "nccl/src/nccl.h"


// template <typename Dtype>
// void CommConfig<Dtype>::SetGpuBufferPtr(Dtype** buf_ptr, int64_t buf_size) {
//   gpu_buf_size_ = buf_size;
//   gpu_buf_ptr_ = buf_ptr;
// }


// template <typename Dtype>
// void CommConfig<Dtype>::SetLeftGpuBufferPtr(Dtype** buf_ptr, int64_t buf_size) {
//   // if (buf == NULL || *buf == NULL) {
//   //   std::cout << "left_gpu_buffer set to NULL!" << std::endl;
//   //   exit(1);
//   // }
//   // left_gpu_buf_size_ = buf_size;
//   // left_gpu_buf_ptr_ = buf_ptr;
// }


// template <typename Dtype>
// void CommConfig<Dtype>::SetRightGpuBufferPtr(Dtype** buf_ptr, int64_t buf_size) {
//   // if (buf == NULL || *buf == NULL) {
//   //   std::cout << "right_gpu_buffer set to NULL!" << std::endl;
//   //   exit(1);
//   // }
//   // right_gpu_buf_size_ = buf_size;
//   // right_gpu_buf_ptr_ = buf_ptr;
// }


template <typename Dtype>
Communicator<Dtype>::Communicator(CommConfig<Dtype>& config) : 
  config_(config),
  nccl_comm_(NULL),
  stream_comm_(NULL),
  mpi_intra_group_comm_(NULL),
  mpi_inter_group_comm_(NULL),
  gpu_buf_(NULL),
  mpi_diff_buf_(NULL) {
  /* initialize communication on GPU*/  
  CUDA_CHECK(cudaSetDevice(config_.device_id_) );
  int64_t buf_size = config_.gpu_buf_size_;
  CUDA_CHECK(cudaMalloc(&gpu_buf_, sizeof(Dtype) * buf_size) );

  nccl_comm_ = (ncclComm_t*)malloc(sizeof(ncclComm_t) );
  NCCL_CHECK(ncclCommInitRank(nccl_comm_, 
    config_.n_dev_clique_local_, config_.clique_id_, 
    config_.clique_rank_) );

  stream_comm_ = (cudaStream_t*)malloc(sizeof(cudaStream_t) );
  CUDA_CHECK(cudaStreamCreate(stream_comm_) );

  /* initialize communication for MPI */
  if (config_.is_group_root_ && !config_.is_clique_root_) {
    std::cerr << "Error: each clique may have only one thread involving mpi" << std::endl; 
    exit(1);
  }
  if (config_.is_clique_root_) {
    mpi_diff_buf_ = new Dtype[config_.mpi_diff_buf_size_];
    // int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &(config_.mpi_rank_) );
    // MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    config_.group_id = config_.mpi_rank_ / N_PROC_PER_GROUP;
    mpi_intra_group_comm_ = new MPI_Comm;
    MPI_Comm_split(MPI_COMM_WORLD, config_.group_id_, 
      config_.mpi_rank_, mpi_intra_group_comm_);
    if (config_.is_group_root_) {
      mpi_model_buf_ = new Dtype[config_.mpi_model_buf_size_];
      MPI_Comm_split(MPI_COMM_WORLD, GROUP_ROOT_COMM_ID, 
        config_.mpi_rank_, mpi_inter_group_comm_);
    }
    else { 
      /** MPI_Comm_split needs to be called from all process. 
       * We create the mpi_inter_group_comm_ for this purpose.
       * Other communicator creation function did not fit in 
       * our design, where we need to create a new group distributedly
       * from all processes. 
       **/
      MPI_Comm_split(MPI_COMM_WORLD, NON_GROUP_ROOT_COMM_ID, 
        config_.mpi_rank_, mpi_inter_group_comm_);
      MPI_Comm_free(mpi_inter_group_comm_);
      mpi_inter_group_comm_ = NULL;
    }
  }
}


// template <typename Dtype>
// void Communicator<Dtype>::SetLeftGpuBuffer(Dtype** buffer_addr, int64_t buf_size) {
//   config_.left_gpu_buf_ptr_ = buffer_addr;
//   config_.left_gpu_buf_size_ = buf_size;
// }


// template <typename Dtype>
// void Communicator<Dtype>::SetRightGpuBuffer(Dtype** buffer_addr, int64_t buf_size) {
//   config_.right_gpu_buf_ptr_ = buffer_addr;
//   config_.right_gpu_buf_size_ = buf_size;
// }


template <typename Dtype>
void Communicator<Dtype>::CliqueReduce() {
  ncclDataType_t type = DtypeToNCCLDType<Dtype>::type;
  if (config_.is_clique_root_)
    NCCL_CHECK(ncclReduce( (const void*)gpu_buf_, 
      (void*)gpu_buf_, config_.gpu_buf_size_, 
      type, ncclSum, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) );
  else
    NCCL_CHECK(ncclReduce( (const void*)gpu_buf_, NULL,
      config_.gpu_buf_size_, type, ncclSum, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) ); 
}


template <typename Dtype>
void Communicator<Dtype>::CliqueBroadcast() {
  ncclDataType_t type = DtypeToNCCLDType<Dtype>::type;
  if (config_.is_clique_root_)
    NCCL_CHECK(ncclBcast( (void*)gpu_buf_, 
      config_.gpu_buf_size_, type, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) );
  else
    NCCL_CHECK(ncclBcast( (void*)gpu_buf_, 
      config_.gpu_buf_size_, type, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) ); 
}


template <typename Dtype>
void Communicator<Dtype>::InterMachineAllReduce() {
  /* copy GPU memory to CPU memory */
  if (config_.is_clique_root_)

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


template class CommConfig<float>;
template class CommConfig<double>;
template class Communicator<float>;
template class Communicator<double>; 