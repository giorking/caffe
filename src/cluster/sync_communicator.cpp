#include <iostream>
#include <cstring>
#include <mpi.h>
#include <thread>
#include "cluster/sync_communicator.hpp"
#include "nccl/src/nccl.h"
#include "cluster/debug_utils.hpp"
#include "cluster/comm_utils.hpp"
#include "cluster/timer.hpp"


namespace caffe {


template <typename Dtype>
void SyncCommunicator<Dtype>::Init(int64_t buf_size, Dtype* external_cpu_buf, 
  Dtype* external_gpu_buf, int64_t buf_tmp_size) {
  /* initialize communication on GPU*/  
  CUDA_CHECK(cudaSetDevice(config_.device_id_) );
  
  gpu_buf_size_ = buf_size;
  gpu_buf_ = external_gpu_buf;
#ifdef GPU_DIRECT_MPI
  gpu_buf_tmp_size_ = buf_tmp_size;  
  if (buf_tmp_size != 0)
    CUDA_CHECK(cudaMalloc(&gpu_buf_tmp_, sizeof(Dtype) * gpu_buf_tmp_size_) );
#else
  // we only need GPU buffer if we are not using GPU direct MPI
  cpu_buf_size_ = buf_size;
  cpu_buf_tmp_size_ = buf_tmp_size;
  cpu_buf_ = external_cpu_buf;
  if (buf_tmp_size != 0)
    cpu_buf_tmp_ = (Dtype*)malloc(sizeof(Dtype) * cpu_buf_tmp_size_);
#endif

  int n_device;
  CUDA_CHECK(cudaGetDeviceCount(&n_device) );
  if (n_device % nDevicePerProc != 0) {
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
    MPI_Comm_split(MPI_COMM_WORLD, config_.group_id_, 
      config_.mpi_rank_, mpi_sync_comm_);
    /** 
     * if we do not have GPU direct MPI, we need to allocate intermediate
     * buffer on cpu.
     */
// #ifndef GPU_DIRECT_MPI
//     cpu_buf_ = new Dtype[cpu_buf_size_];
//     // we use pinned memory as cpu_buf may write large block to device.
//     CUDA_CHECK(cudaMallocHost(&cpu_buf_, sizeof(Dtype) * cpu_buf_size_) );
//     CUDA_CHECK(cudaMemset(cpu_buf_, 0, sizeof(Dtype) * cpu_buf_size_) );
//     std::memset(cpu_buf_, 0, sizeof(Dtype) * cpu_buf_size_);
// #endif
  }

  CUBLAS_CHECK(cublasCreate(&cublas_handle_) );

}


template <typename Dtype>
void SyncCommunicator<Dtype>::CliqueReduce() {
  ncclDataType_t type = DtypeToNCCLDtype<Dtype>::type;
  pthread_barrier_wait(process_barrier_);
  if (config_.is_clique_root_) {
    NCCL_CHECK(ncclReduce( (const void*)gpu_buf_, 
      (void*)gpu_buf_, gpu_buf_size_, 
      type, ncclSum, config_.clique_root_rank_, 
      *nccl_comm_, *stream_comm_) );
  }
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
void SyncCommunicator<Dtype>::InterMachineReduceScatter() {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // test if the size is a power of 2
  if (size & (size - 1) != 0) {
    std::cout << "Only support power-of-2 number of machines" << std::endl;
    std::exit(1);
  }

#ifdef GPU_DIRECT_MPI
  Dtype* buffer = gpu_buf_;
  Dtype* tmp_buf = gpu_buf_tmp_;
  int64_t buf_size = gpu_buf_size_;
  int64_t block_size = gpu_buf_size_ / size;
#else
  Dtype* buffer = cpu_buf_;
  Dtype* tmp_buf = cpu_buf_tmp_;
  int64_t buf_size = cpu_buf_size_;
  int64_t block_size = cpu_buf_size_ / size;
#endif

  int rank = config_.mpi_rank_;
  MPI_Datatype type = DtypeToMPIDtype<Dtype>::type;
  MPI_Status status;

  int64_t send_buf_size;
  int64_t recv_buf_size;
  int send_start_block = 0;
  int recv_start_block = 0;

  int interval = size;
  while(interval > 1) {
    // send_start_block = std::min(send_start_block, recv_start_block);
    if (rank % interval < interval / 2) {
      recv_start_block = recv_start_block;
      send_start_block = recv_start_block + interval / 2;
      send_buf_size = (send_start_block + interval / 2 == size) ?
        buf_size - block_size * send_start_block : block_size * interval / 2;
      recv_buf_size = (recv_start_block + interval / 2 == size) ?
        buf_size - block_size * recv_start_block : block_size * interval / 2;

#ifdef DEBUG
      std::cout << "check send recv 1: rank " << rank  << " interval " << interval 
        << " target " << rank + interval / 2 
        // << " send start " << send_start_block * block_size
        // << " send end " << send_start_block * block_size + send_buf_size
        // << " total buf size " << buf_size << std::endl;
        << " send block " << send_start_block
        << " recv block " << recv_start_block
        << " send buf size " << send_buf_size
        << " recv buf size " << recv_buf_size 
        << " total buf size " << buf_size << std::endl;
#endif

#ifdef GPU_DIRECT_MPI  
      // we need to guarantee the the stream work is done before next transimission
      cudaStream_t cublas_stream;
      cublasGetStream(cublas_handle_, &cublas_stream);
      cudaStreamSynchronize(cublas_stream);
#endif
      MPI_Sendrecv(buffer + block_size * send_start_block, send_buf_size, 
        type, rank + interval / 2, rank, tmp_buf, 
        recv_buf_size, type, rank + interval / 2, 
        rank + interval / 2, MPI_COMM_WORLD, &status);
    }
    else {
      send_start_block = recv_start_block;
      recv_start_block = recv_start_block + interval / 2;
      send_buf_size = (send_start_block + interval / 2 == size) ?
        buf_size - block_size * send_start_block : block_size * interval / 2;
      recv_buf_size = (recv_start_block + interval / 2 == size) ?
        buf_size - block_size * recv_start_block : block_size * interval / 2;

#ifdef DEBUG
      std::cout << "check send recv 2: rank " << rank << " interval " << interval 
        << " target " << rank - interval / 2
        // << " send start " << send_start_block * block_size
        // << " send end " << send_start_block * block_size + send_buf_size
        // << " total buf size " << buf_size << std::endl;
        << " send block " << send_start_block
        << " recv block " << recv_start_block 
        << " send buf size " << send_buf_size
        << " recv buf size " << recv_buf_size
        << " total buf size " << buf_size << std::endl;
#endif

#ifdef GPU_DIRECT_MPI  
      // we need to guarantee the the stream work is done before next transimission
      cudaStream_t cublas_stream;
      cublasGetStream(cublas_handle_, &cublas_stream);
      cudaStreamSynchronize(cublas_stream);
#endif
      MPI_Sendrecv(buffer + block_size * send_start_block, send_buf_size, 
        type, rank - interval / 2, rank, tmp_buf, 
        recv_buf_size, type, rank - interval / 2, 
        rank - interval / 2, MPI_COMM_WORLD, &status);
    }

#ifdef GPU_DIRECT_MPI
    Dtype factor = 1.0;
    cublasAxpy(cublas_handle_, recv_buf_size, &factor, tmp_buf, buffer + recv_start_block * block_size);
#else
    Dtype* recv_buf = buffer + recv_start_block * block_size;
    for (int i = 0; i < recv_buf_size; i++)
      recv_buf[i] += tmp_buf[i];
#endif
    interval /= 2;
  }
#ifdef GPU_DIRECT_MPI  
  // we need to guarantee the the stream work is done before next transimission
  cudaStream_t cublas_stream;
  cublasGetStream(cublas_handle_, &cublas_stream);
  cudaStreamSynchronize(cublas_stream);
#endif
}


template <typename Dtype>
void SyncCommunicator<Dtype>::InterMachineAllGather() {
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // test if the size is a power of 2
  if (size & (size - 1) != 0) {
    std::cout << "Only support power-of-2 number of machines" << std::endl;
    std::exit(1);
  }
  int interval = 2;
  int rank = config_.mpi_rank_;
  MPI_Datatype type = DtypeToMPIDtype<Dtype>::type;
  MPI_Status status;

#ifdef GPU_DIRECT_MPI
  int64_t block_size = gpu_buf_size_ / size;
  Dtype* buffer = gpu_buf_;
  int64_t buf_size = gpu_buf_size_;
#else
  int64_t block_size = cpu_buf_size_ / size;
  Dtype* buffer = cpu_buf_;
  int64_t buf_size = cpu_buf_size_;
#endif

  int64_t send_buf_size;
  int64_t recv_buf_size;
  int send_start_block = rank;
  int recv_start_block = rank;

  while(interval <= size) {
    send_start_block = std::min(send_start_block, recv_start_block);
    if (rank % interval < interval / 2) {
      recv_start_block = std::min(send_start_block, 
        recv_start_block) + interval / 2;
      send_buf_size = (send_start_block + interval / 2 == size) ?
        buf_size - block_size * send_start_block : block_size * interval / 2;
      recv_buf_size = (recv_start_block + interval / 2 == size) ?
        buf_size - block_size * recv_start_block : block_size * interval / 2;

#ifdef DEBUG
      std::cout << "check send recv 1: rank " << rank  << " interval " << interval 
        << " target " << rank + interval / 2 
        << " send block " << send_start_block
        << " recv block " << recv_start_block
        << " send buf size " << send_buf_size
        << " recv buf size " << recv_buf_size << std::endl;
#endif

      MPI_Sendrecv(buffer + block_size * send_start_block, send_buf_size, 
        type, rank + interval / 2, rank, buffer + block_size * recv_start_block, 
        recv_buf_size, type, rank + interval / 2, 
        rank + interval / 2, MPI_COMM_WORLD, &status);
    }
    else {
      recv_start_block = std::min(send_start_block, 
        recv_start_block) - interval / 2;
      send_buf_size = (send_start_block + interval / 2 == size) ?
        buf_size - block_size * send_start_block : block_size * interval / 2;
      recv_buf_size = (recv_start_block + interval / 2 == size) ?
        buf_size - block_size * recv_start_block : block_size * interval / 2;

#ifdef DEBUG
      std::cout << "check send recv 2: rank " << rank << " interval " << interval 
        << " target " << rank - interval / 2 
        << " send block " << send_start_block
        << " recv block " << recv_start_block 
        << " send buf size " << send_buf_size
        << " recv buf size " << recv_buf_size << std::endl;
#endif

      MPI_Sendrecv(buffer + block_size * send_start_block, send_buf_size, 
        type, rank - interval / 2, rank, buffer + block_size * recv_start_block, 
        recv_buf_size, type, rank - interval / 2, 
        rank - interval / 2, MPI_COMM_WORLD, &status);
    }
    interval *= 2;
  }

}


template <typename Dtype>
void SyncCommunicator<Dtype>::InterMachineAllReduce() {
#ifdef TIMER
  Timer1 timer;
  timer.start();
#endif

  InterMachineReduceScatter();

#ifdef TIMER
  timer.stop();
  std::cout << "Reduce scatter in " << timer.getElapsedTimeInMilliSec() << std::endl;
#endif

  // divide the results by the number of workers
  Dtype scalar = 1.0 / (nDevicePerProc * nProcPerGroup);
#ifdef GPU_DIRECT_MPI  
  CUBLAS_CHECK(cublasScal(cublas_handle_, gpu_buf_size_, 
    &scalar, gpu_buf_) );
  // we need to guarantee the the stream work is done before next transimission
  cudaStream_t cublas_stream;
  cublasGetStream(cublas_handle_, &cublas_stream);
  cudaStreamSynchronize(cublas_stream);
#else
  for (int64_t i = 0; i < cpu_buf_size_; i++)
    cpu_buf_[i] *= scalar;
#endif

#ifdef TIMER
  timer.start();
#endif

  InterMachineAllGather();

#ifdef TIMER
  timer.stop();
  std::cout << "All gather in " << timer.getElapsedTimeInMilliSec() << std::endl;
#endif

}

// template <typename Dtype>
// void SyncCommunicator<Dtype>::InterMachineAllReduce() {
//   // Only clique root will call this function 
//   // TODO Jian: modify to adapt to the IB setting
//   // copy GPU memory to CPU memory 
//   if (gpu_buf_size_ > cpu_buf_size_) {
//     std::cout << "Can not do inter machine allReduce." 
//       << " mpi buffer is smaller than gpu buffer." << std::endl;
//   }

// #ifdef TIMER
//   Timer timer_outer;
//   timer_outer.start();
// #endif

//   CUDA_CHECK(cudaMemcpy(cpu_buf_, gpu_buf_, 
//     sizeof(Dtype) * gpu_buf_size_, cudaMemcpyDeviceToHost) );
//   MPI_Datatype type = DtypeToMPIDtype<Dtype>::type;

// #ifdef TIMER
//   Timer timer_inner;
//   timer_inner.start();
// #endif

//   // pthread_mutex_lock(mpi_mutex_);
//   MPI_Allreduce(MPI_IN_PLACE, (void*)cpu_buf_,
//     gpu_buf_size_, type, MPI_SUM, *mpi_sync_comm_);
//   // pthread_mutex_unlock(mpi_mutex_);

// #ifdef TIMER
//   timer_inner.stop();
//   DEBUG_PRINT_TIME_WITH_RANK_DEVICE_ID(MPI_COMM_WORLD, timer_inner, " Sync COMM: MPI Allreduce for SyncGroup in ");
// #endif

//   /* copy from CPU memory to GPU memory */
//   CUDA_CHECK(cudaMemcpy(gpu_buf_, cpu_buf_, 
//     sizeof(Dtype) * gpu_buf_size_, cudaMemcpyHostToDevice) );

// #ifdef TIMER
//   timer_outer.stop();
//   DEBUG_PRINT_TIME_WITH_RANK_DEVICE_ID(MPI_COMM_WORLD, timer_inner, " Sync COMM: MPI Allreduce + memcpy for SyncGroup in ");
// #endif

// }


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


  // inter ndoe communication 
  if (this->IsCliqueRoot() ) {

#ifndef GPU_DIRECT_MPI
    CUDA_CHECK(cudaMemcpy(cpu_buf_, gpu_buf_, 
    sizeof(Dtype) * gpu_buf_size_, cudaMemcpyDeviceToHost) );
#endif

    InterMachineAllReduce();

#ifndef GPU_DIRECT_MPI 
    // copy from CPU memory to GPU memory 
    CUDA_CHECK(cudaMemcpy(gpu_buf_, cpu_buf_, 
    sizeof(Dtype) * gpu_buf_size_, cudaMemcpyHostToDevice) );
#endif

  }

  // broadcast within clique 
  if (do_broadcast)
    CliqueBroadcast();
}


template class SyncCommConfig<float>;
template class SyncCommConfig<double>;
template class SyncCommunicator<float>;
template class SyncCommunicator<double>; 

}