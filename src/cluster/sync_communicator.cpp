#include <iostream>
#include <mpi.h>
#include "cluster/sync_communicator.hpp"
#include "nccl/src/nccl.h"
#include "cluster/debug_utils.hpp"


template <typename Dtype>
SyncCommunicator<Dtype>::SyncCommunicator(SyncCommConfig<Dtype>& config) : 
  config_(config),
  nccl_comm_(NULL),
  stream_comm_(NULL),
  mpi_sync_comm_(NULL),
  gpu_buf_(NULL),
  mpi_sync_buf_(NULL) {
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
    std::cout << "rank " << config_.mpi_rank_ << " as clique root." << std::endl;
    mpi_sync_buf_ = new Dtype[config_.mpi_sync_buf_size_];
    mpi_sync_comm_ = new MPI_Comm;
    MPI_Comm_split(MPI_COMM_WORLD, config_.group_id_, 
      config_.mpi_rank_, mpi_sync_comm_);
    // if (config_.is_group_root_) {
    //   std::cout << "rank " << config_.mpi_rank_ << " as group root." << std::endl;
    //   mpi_recv_buf_ = new Dtype[config_.mpi_recv_buf_size_];
    //   mpi_inter_group_comm_ = new MPI_Comm;
    //   MPI_Comm_split(MPI_COMM_WORLD, GROUP_ROOT_COMM_ID, 
    //     config_.mpi_rank_, mpi_inter_group_comm_);
    // }
    // else { 
    //   *
    //    * MPI_Comm_split needs to be called from all process. 
    //    * We create the mpi_inter_group_comm_ for this purpose.
    //    * Other communicator creation function did not fit in 
    //    * our design, where we need to create a new group distributedly
    //    * from all processes. 
       
    //   mpi_inter_group_comm_ = new MPI_Comm;
    //   MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, 
    //     config_.mpi_rank_, mpi_inter_group_comm_);
    //   // MPI_Comm_free(mpi_inter_group_comm_);
    //   delete mpi_inter_group_comm_;
    //   mpi_inter_group_comm_ = NULL;
    // }
  }
}


template <typename Dtype>
void SyncCommunicator<Dtype>::CliqueReduce() {
  ncclDataType_t type = DtypeToNCCLDtype<Dtype>::type;

// // DEBUG
//   std::string str1 = "before reduce ";
//   DisplayGpuArray(gpu_buf_, config_.gpu_buf_size_, str1);



  if (config_.is_clique_root_)
    NCCL_CHECK(ncclReduce( (const void*)gpu_buf_, 
      (void*)gpu_buf_, config_.gpu_buf_size_, 
      type, ncclSum, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) );
  else
    NCCL_CHECK(ncclReduce( (const void*)gpu_buf_, NULL,
      config_.gpu_buf_size_, type, ncclSum, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) ); 

  // // DEBUG
  // std::string str2 = "after reduce ";
  // DisplayGpuArray(gpu_buf_, config_.gpu_buf_size_, str2);

}


template <typename Dtype>
void SyncCommunicator<Dtype>::CliqueBroadcast() {
  ncclDataType_t type = DtypeToNCCLDtype<Dtype>::type;
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
void SyncCommunicator<Dtype>::InterMachineAllReduce() {
  /* Only clique root will call this function */
  /* TODO Jian: modify to adapt to the IB setting*/
  /* copy GPU memory to CPU memory */
  if (config_.gpu_buf_size_ > config_.mpi_sync_buf_size_) {
    std::cout << "Can not do inter machine allReduce." 
      << " mpi buffer is smaller than gpu buffer." << std::endl;
  }
  CUDA_CHECK(cudaMemcpy(mpi_sync_buf_, gpu_buf_, 
    sizeof(Dtype) * config_.gpu_buf_size_, cudaMemcpyDeviceToHost) );
  MPI_Datatype type = DtypeToMPIDtype<Dtype>::type;

// // DEBUG
//   std::cout << "mpi rank " << config_.mpi_rank_ 
//     << " first send " << mpi_send_buf_[0] 
//     << " last send " << mpi_send_buf_[config_.gpu_buf_size_ - 1]
//     << std::endl;
    MPI_Allreduce(MPI_IN_PLACE, (void*)mpi_sync_buf_,
      config_.gpu_buf_size_, type, MPI_SUM, *mpi_sync_comm_);
// // DEBUG
//   std::cout << "mpi rank " << config_.mpi_rank_ 
//     << " first recv " << mpi_recv_buf_[0] 
//     << " last recv " << mpi_recv_buf_[config_.gpu_buf_size_ - 1]
//     << std::endl;

  /* copy from CPU memory to GPU memory */
  CUDA_CHECK(cudaMemcpy(gpu_buf_, mpi_sync_buf_, 
    sizeof(Dtype) * config_.gpu_buf_size_, cudaMemcpyHostToDevice) );
}


template <typename Dtype>
void SyncCommunicator<Dtype>::SyncGroup() {
  /**
   * recvbuf may be NULL for non-clique-lead GPU
   * We first use reduce to gether within clique
   * Then reduce-all with MPI Then broadcast with 
   * clique from clique lead
   */
  
  /* reduce within clique */
  CliqueReduce();

  /* TODO Jian: inter ndoe communication */
  if (config_.is_clique_root_)
    InterMachineAllReduce();

    /* TODO Jian: broadcast within clique */
  CliqueBroadcast();
}


template class SyncCommConfig<float>;
template class SyncCommConfig<double>;
template class SyncCommunicator<float>;
template class SyncCommunicator<double>; 