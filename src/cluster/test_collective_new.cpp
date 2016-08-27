#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
//#include "cluster/debug_utils.hpp"
//#include "cluster/comm_utils.hpp"
#include <iostream>
#include "cluster/timer.hpp"


// #ifndef GPU_MEM
// #define GPU_MEM
// #endif

#ifndef ALL_REDUCE
#define ALL_REDUCE
#endif

#ifndef COLLECTIVE
#define COLLECTIVE
#endif

void MasterTimeAndBandWidth(Timer1 timer, int64_t size) {
	std::cout << "master time " << timer.getElapsedTimeInMilliSec() << " milli sec" << std::endl;
	std::cout << "master bandwidth " << size / 1024.0 / 1024.0 / 1024.0 / (double)(timer.getElapsedTimeInMilliSec() / 1e3) << std::endl;
}

void SlaveTimeAndBandWidth(Timer1 timer, int64_t size) {
	std::cout << "slave time " << timer.getElapsedTimeInMilliSec() << " milli sec" << std::endl;
	std::cout << "slave bandwidth " << size / 1024.0 / 1024.0 / 1024.0 / (double)(timer.getElapsedTimeInMilliSec() / 1e3) << std::endl;
}


template <typename Dtype>
void TestAllReduce(int64_t buf_size) {
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	MPI_Status recv_status;
	Dtype* buf = NULL;
#ifdef GPU_MEM
	cudaMalloc(&buf, sizeof(Dtype) * buf_size);
#else
	buf = (Dtype*)malloc(sizeof(Dtype) * buf_size);
#endif

	int64_t data_size = buf_size * sizeof(Dtype);
	
	Timer1 timer;
	timer.start();

#ifdef COLLECTIVE
	// master
	if (rank == 0) {
		// for (int i = 1; i < size; i++)
		// 	MPI_Send(buf, buf_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, buf, buf_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		timer.stop();
		MasterTimeAndBandWidth(timer, data_size);
	}
	// slave
	else {
		// MPI_Recv(buf, buf_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, buf, buf_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
		timer.stop();
		SlaveTimeAndBandWidth(timer, data_size);		
	}	

#else

	// master
	if (rank == 0) {
		for (int i = 1; i < size; i++)
			MPI_Recv(buf, buf_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &recv_status);
		for (int i = 1; i < size; i++)
			MPI_Send(buf, buf_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
		timer.stop();
		MasterTimeAndBandWidth(timer, data_size);
	}
	// slave
	else {
		MPI_Send(buf, buf_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		MPI_Recv(buf, buf_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &recv_status);
		timer.stop();
		SlaveTimeAndBandWidth(timer, data_size);
	}
#endif


#ifdef GPU_MEM
	cudaFree(buf);
#else
	free(buf);
#endif

}



template <typename Dtype>
void TestBroadCast(int64_t buf_size) {
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	MPI_Status recv_status;
	Dtype* buf = NULL;
#ifdef GPU_MEM
	cudaMalloc(&buf, sizeof(Dtype) * buf_size);
#else
	buf = (Dtype*)malloc(sizeof(Dtype) * buf_size);
#endif

	int64_t data_size = buf_size * sizeof(Dtype);
	
	Timer1 timer;
	timer.start();

#ifdef COLLECTIVE
	// master
	if (rank == 0) {
		// for (int i = 1; i < size; i++)
		// 	MPI_Send(buf, buf_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
		MPI_Bcast(buf, buf_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
		timer.stop();
		MasterTimeAndBandWidth(timer, data_size);
	}
	// slave
	else {
		// MPI_Recv(buf, buf_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		MPI_Bcast(buf, buf_size, MPI_FLOAT, 0, MPI_COMM_WORLD);	
		timer.stop();
		SlaveTimeAndBandWidth(timer, data_size);		
	}	

#else

	// master
	if (rank == 0) {
		for (int i = 1; i < size; i++)
			MPI_Send(buf, buf_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
		timer.stop();
		MasterTimeAndBandWidth(timer, data_size);
	}
	// slave
	else {
		MPI_Recv(buf, buf_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &recv_status);
		timer.stop();
		SlaveTimeAndBandWidth(timer, data_size);
	}

#endif


#ifdef GPU_MEM
	cudaFree(buf);
#else
	free(buf);
#endif

}


template <typename Dtype>
void Test(int64_t buf_size) {

	for (int i = 0; i < 3; i++) {
	  std::cout << "test broadcast " << std::endl;
	 	TestBroadCast<Dtype>(buf_size);
	 }

	for (int i = 0; i < 3; i++) {
		std::cout << "test all reduce " << std::endl;
		TestAllReduce<Dtype>(buf_size);	
	}

}


int main() {
	MPI_Init(NULL, NULL);

	int64_t buf_size = 50000000;
	Test<float>(buf_size);
	// for (int i = 0; i < 100; i++) {
	// 	TestBroadCast<float>(buf_size);

	// 	// usleep(1000000);

	// }

	MPI_Finalize();
}
