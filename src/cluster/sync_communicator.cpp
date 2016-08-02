#include <iostream>
#include <mpi.h>
#include "cluster/sync_communicator.hpp"
#include "nccl/src/nccl.h"
#include "cluster/debug_utils.hpp"


template <typename Dtype>
SyncCommunicator<Dtype>::SyncCommunicator(const SyncCommConfig<Dtype>& config, 
  const int64_t buf_size) : 
  config_(config),
  nccl_comm_(NULL),
  stream_comm_(NULL),
  mpi_sync_comm_(NULL),
  gpu_buf_(NULL),
  mpi_sync_buf_(NULL) {
  // set buffer size
  gpu_buf_size_ = buf_size;
  mpi_sync_buf_size_ = buf_size;
  /* initialize communication on GPU*/  
  CUDA_CHECK(cudaSetDevice(config_.device_id_) );
  CUDA_CHECK(cudaMalloc(&gpu_buf_, sizeof(Dtype) * gpu_buf_size_) );
  nccl_comm_ = (ncclComm_t*)malloc(sizeof(ncclComm_t) );

  int n_device;
  CUDA_CHECK(cudaGetDeviceCount(&n_device) );
  if (n_device % N_DEVICE_PER_PROC != 0) {
    std::cout << "device on the machine should be " 
      << "fully devided into cliques(procs)" << std::endl;
    exit(0);
  }

  NCCL_CHECK(ncclCommInitRank(nccl_comm_, N_DEVICE_PER_PROC, 
    config_.clique_id_, config_.clique_rank_) );

  stream_comm_ = (cudaStream_t*)malloc(sizeof(cudaStream_t) );
  CUDA_CHECK(cudaStreamCreate(stream_comm_) );

  if (config_.is_clique_root_) {
    mpi_sync_comm_ = new MPI_Comm;
    mpi_sync_buf_ = new Dtype[mpi_sync_buf_size_];
    MPI_Comm_split(MPI_COMM_WORLD, config_.group_id_, 
      config_.mpi_rank_, mpi_sync_comm_);
  }
}



template <typename Dtype>
void SyncCommunicator<Dtype>::CliqueReduce() {
  ncclDataType_t type = DtypeToNCCLDtype<Dtype>::type;
  if (config_.is_clique_root_)
    NCCL_CHECK(ncclReduce( (const void*)gpu_buf_, 
      (void*)gpu_buf_, gpu_buf_size_, 
      type, ncclSum, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) );
  else
    NCCL_CHECK(ncclReduce( (const void*)gpu_buf_, NULL,
      gpu_buf_size_, type, ncclSum, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) ); 
}


template <typename Dtype>
void SyncCommunicator<Dtype>::CliqueBroadcast() {
  ncclDataType_t type = DtypeToNCCLDtype<Dtype>::type;
  if (config_.is_clique_root_)
    NCCL_CHECK(ncclBcast( (void*)gpu_buf_, 
      gpu_buf_size_, type, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) );
  else
    NCCL_CHECK(ncclBcast( (void*)gpu_buf_, 
      gpu_buf_size_, type, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) ); 
}


template <typename Dtype>
void SyncCommunicator<Dtype>::InterMachineAllReduce() {
  /* Only clique root will call this function */
  /* TODO Jian: modify to adapt to the IB setting*/
  /* copy GPU memory to CPU memory */
  if (gpu_buf_size_ > mpi_sync_buf_size_) {
    std::cout << "Can not do inter machine allReduce." 
      << " mpi buffer is smaller than gpu buffer." << std::endl;
  }
  CUDA_CHECK(cudaMemcpy(mpi_sync_buf_, gpu_buf_, 
    sizeof(Dtype) * gpu_buf_size_, cudaMemcpyDeviceToHost) );
  MPI_Datatype type = DtypeToMPIDtype<Dtype>::type;

  // if (config_.is_group_root_)
  MPI_Allreduce(MPI_IN_PLACE, (void*)mpi_sync_buf_,
    gpu_buf_size_, type, MPI_SUM, *mpi_sync_comm_);

  /* copy from CPU memory to GPU memory */
  CUDA_CHECK(cudaMemcpy(gpu_buf_, mpi_sync_buf_, 
    sizeof(Dtype) * gpu_buf_size_, cudaMemcpyHostToDevice) );
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