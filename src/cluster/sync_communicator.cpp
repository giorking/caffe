#include <iostream>
#include <cstring>
#include <mpi.h>
#include <thread>
#include "cluster/sync_communicator.hpp"
#include "nccl/src/nccl.h"
#include "cluster/debug_utils.hpp"
#include "cluster/comm_utils.hpp"


template <typename Dtype>
void SyncCommunicator<Dtype>::Init(int64_t buf_size) {
  // set buffer size
  gpu_buf_size_ = buf_size;
  mpi_sync_buf_size_ = buf_size;
  /* initialize communication on GPU*/  
  CUDA_CHECK(cudaSetDevice(config_.device_id_) );
  CUDA_CHECK(cudaMalloc(&gpu_buf_, sizeof(Dtype) * gpu_buf_size_) );
  CUDA_CHECK(cudaMemset(gpu_buf_, 0, sizeof(Dtype) * gpu_buf_size_) );

  int n_device;
  CUDA_CHECK(cudaGetDeviceCount(&n_device) );
  if (n_device % N_DEVICE_PER_PROC != 0) {
    std::cout << "device on the machine should be " 
      << "fully devided into cliques(procs)" << std::endl;
    exit(0);
  }

  nccl_comm_ = new ncclComm_t;
  CUDA_CHECK(cudaSetDevice(config_.device_id_) );
  NCCL_CHECK(ncclCommInitRank(nccl_comm_, config_.n_dev_in_clique_, 
    config_.clique_id_, config_.clique_rank_) );

  CUDA_CHECK(cudaSetDevice(config_.device_id_) );
  stream_comm_ = (cudaStream_t*)malloc(sizeof(cudaStream_t) );
  CUDA_CHECK(cudaStreamCreate(stream_comm_) );

  if (config_.is_clique_root_) {
    mpi_sync_comm_ = new MPI_Comm;
    mpi_sync_buf_ = new Dtype[mpi_sync_buf_size_];
    std::memset(mpi_sync_buf_, 0, sizeof(Dtype) * mpi_sync_buf_size_);
    MPI_Comm_split(MPI_COMM_WORLD, config_.group_id_, 
      config_.mpi_rank_, mpi_sync_comm_);
  }

  // // DEBUG
  // std::cout << "sync comm initilization done!" << std::endl;


}


template <typename Dtype>
void SyncCommunicator<Dtype>::CliqueReduce() {

  // // DEBUG
  // DEBUG_PRINT_DEVICE_ID("before clique device id ");

  ncclDataType_t type = DtypeToNCCLDtype<Dtype>::type;
  pthread_barrier_wait(process_barrier_);
  if (config_.is_clique_root_) 
    NCCL_CHECK(ncclReduce( (const void*)gpu_buf_, 
      (void*)gpu_buf_, gpu_buf_size_, 
      type, ncclSum, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) );
  else
    NCCL_CHECK(ncclReduce( (const void*)gpu_buf_, NULL,
      gpu_buf_size_, type, ncclSum, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) );
  CUDA_CHECK(cudaStreamSynchronize(*stream_comm_) ); 
}


template <typename Dtype>
void SyncCommunicator<Dtype>::CliqueBroadcast() {
  ncclDataType_t type = DtypeToNCCLDtype<Dtype>::type;
  pthread_barrier_wait(process_barrier_);
  if (config_.is_clique_root_) {
    // copy from CPU memory to GPU memory 
    CUDA_CHECK(cudaMemcpy(gpu_buf_, mpi_sync_buf_, 
    sizeof(Dtype) * gpu_buf_size_, cudaMemcpyHostToDevice) );
    
    // do broadcast
    NCCL_CHECK(ncclBcast( (void*)gpu_buf_, 
      gpu_buf_size_, type, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) );
  }
  else
    NCCL_CHECK(ncclBcast( (void*)gpu_buf_, 
      gpu_buf_size_, type, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) ); 
  CUDA_CHECK(cudaStreamSynchronize(*stream_comm_) ); 
}


template <typename Dtype>
void SyncCommunicator<Dtype>::InterMachineAllReduce() {
  // Only clique root will call this function 
  // TODO Jian: modify to adapt to the IB setting
  // copy GPU memory to CPU memory 
  if (gpu_buf_size_ > mpi_sync_buf_size_) {
    std::cout << "Can not do inter machine allReduce." 
      << " mpi buffer is smaller than gpu buffer." << std::endl;
  }

  CUDA_CHECK(cudaMemcpy(mpi_sync_buf_, gpu_buf_, 
    sizeof(Dtype) * gpu_buf_size_, cudaMemcpyDeviceToHost) );
  MPI_Datatype type = DtypeToMPIDtype<Dtype>::type;


  std::ostringstream address0;
  address0 << (void const *)mpi_sync_comm_;
  // std:string name = address.str();
  std::string s0 = " sync communicator addr " + address0.str();
  DEBUG_PRINT_RANK_DEVICE_ID(MPI_COMM_WORLD, s0);
  DEBUG_PRINT_RANK_DEVICE_ID(*mpi_sync_comm_, s0);

  pthread_mutex_lock(mpi_mutex_);
  MPI_Allreduce(MPI_IN_PLACE, (void*)mpi_sync_buf_,
    gpu_buf_size_, type, MPI_SUM, *mpi_sync_comm_);
  pthread_mutex_unlock(mpi_mutex_);

  /* copy from CPU memory to GPU memory */
  CUDA_CHECK(cudaMemcpy(gpu_buf_, mpi_sync_buf_, 
    sizeof(Dtype) * gpu_buf_size_, cudaMemcpyHostToDevice) );
}


template <typename Dtype>
void SyncCommunicator<Dtype>::SyncGroup(bool do_broadcast) {
  /**
   * recvbuf may be NULL for non-clique-lead GPU
   * We first use reduce to gether within clique
   * Then reduce-all with MPI Then broadcast with 
   * clique from clique lead
   */
  // reduce within clique 
  CliqueReduce();

  // // DEBUG
  // std::cout << "syncGroup : reduce done " << std::endl;

  // inter ndoe communication 
  if (config_.is_clique_root_)
    InterMachineAllReduce();

  // // DEBUG
  // std::cout << "syncGroup : allreduce done " << std::endl;  


  // broadcast within clique 
  if (do_broadcast) {
    CliqueBroadcast();

    // // DEBUG
    // std::cout << "syncGroup : broadcast done " << std::endl;      
  }
}


template class SyncCommConfig<float>;
template class SyncCommConfig<double>;
template class SyncCommunicator<float>;
template class SyncCommunicator<double>; 