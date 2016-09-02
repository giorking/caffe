#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <cmath>
//#include "cluster/debug_utils.hpp"
//#include "cluster/comm_utils.hpp"
#include <iostream>
#include "cluster/timer.hpp"


 #ifndef GPU_MEM
 #define GPU_MEM
 #endif

#ifndef ALL_REDUCE
#define ALL_REDUCE
#endif

// #ifndef COLLECTIVE
// #define COLLECTIVE
// #endif

void MasterTimeAndBandWidth(Timer1 timer, int64_t size) {
	std::cout << "master time " << timer.getElapsedTimeInMilliSec() << " milli sec" << std::endl;
	std::cout << "master bandwidth " << size / 1024.0 / 1024.0 / 1024.0 / (double)(timer.getElapsedTimeInMilliSec() / 1e3) << std::endl;
}

void SlaveTimeAndBandWidth(Timer1 timer, int64_t size) {
	std::cout << "slave time " << timer.getElapsedTimeInMilliSec() << " milli sec" << std::endl;
	std::cout << "slave bandwidth " << size / 1024.0 / 1024.0 / 1024.0 / (double)(timer.getElapsedTimeInMilliSec() / 1e3) << std::endl;
}


template <typename Dtype>
void TestSendRecv(int buf_size) {
	// we need to make sure n_proc is 2^m
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	Dtype* send_buf = (Dtype*)malloc(sizeof(Dtype) * buf_size);
	Dtype* recv_buf = (Dtype*)malloc(sizeof(Dtype) * buf_size);

	MPI_Status status;

	Timer1 timer;
	timer.start();

	int interval = size;

	for (int i = 0; ; i++) {

		if (rank % interval < interval / 2) {
			MPI_Sendrecv(send_buf, buf_size, MPI_FLOAT,
	                rank + interval / 2, rank,
	                recv_buf, buf_size, MPI_FLOAT,
	                rank + interval / 2, rank + interval / 2,
	                MPI_COMM_WORLD, &status);
		}
		else {
			MPI_Sendrecv(send_buf, buf_size, MPI_FLOAT,
	                rank - interval / 2, rank,
	                recv_buf, buf_size, MPI_FLOAT,
	                rank - interval / 2, rank - interval / 2,
	                MPI_COMM_WORLD, &status);
		}
		interval = interval / 2;
		if (interval == 1)
			break;
	}

	timer.stop();
	SlaveTimeAndBandWidth(timer, buf_size);

	free(send_buf);
	free(recv_buf);
}

template <typename Dtype>
void TestSendRecvDouble(int buf_size) {
	// we need to make sure n_proc is 2^m
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


#ifdef GPU_MEM
	Dtype* send_buf1;
	Dtype* send_buf2;
	Dtype* recv_buf1;
	Dtype* recv_buf2;
	cudaMalloc(&send_buf1, sizeof(Dtype) * buf_size);
	cudaMalloc(&send_buf2, sizeof(Dtype) * buf_size);
	cudaMalloc(&recv_buf1, sizeof(Dtype) * buf_size);
	cudaMalloc(&recv_buf1, sizeof(Dtype) * buf_size);


	Dtype* cpu_rand_buf = (Dtype*)malloc(sizeof(Dtype) * buf_size);
	for (int i = 0; i < buf_size; i++)
		cpu_rand_buf[i] = static_cast <Dtype> (rand()) / static_cast <Dtype> (RAND_MAX);
	cudaMemcpy(send_buf1, cpu_rand_buf, sizeof(Dtype) * buf_size, cudaMemcpyHostToDevice);

#else

	Dtype* send_buf1 = (Dtype*)malloc(sizeof(Dtype) * buf_size);
	Dtype* recv_buf1 = (Dtype*)malloc(sizeof(Dtype) * buf_size);
	Dtype* send_buf2 = (Dtype*)malloc(sizeof(Dtype) * buf_size);
	Dtype* recv_buf2 = (Dtype*)malloc(sizeof(Dtype) * buf_size);

#endif

	// MPI_Status status1;
	// MPI_Status status2;
	// MPI_Request request1;
	// MPI_Request request2;
	MPI_Status* status = new MPI_Status[2];
	MPI_Request* request = new MPI_Request[2];


	Timer1 timer;
	timer.start();

	int interval = size;

	for (int i = 0; ; i++) {

		if (rank % interval < interval / 2) {
			// MPI_Sendrecv(send_buf1 + buf_size / (int)pow(2, i + 1), buf_size / pow(2, i + 1), MPI_FLOAT,
	  //               rank + interval / 2, rank,
	  //               recv_buf1 + buf_size / (int)pow(2, i + 1), buf_size / pow(2, i + 1), MPI_FLOAT,
	  //               rank + interval / 2, rank + interval / 2,
	  //               MPI_COMM_WORLD, &status);
	                 
			MPI_Isend(send_buf1 + buf_size / (int)pow(2, i + 1), buf_size / pow(2, i + 1), MPI_FLOAT, rank + interval / 2, 
				rank * 1000 + 1, MPI_COMM_WORLD, request);
			MPI_Irecv(recv_buf1 + buf_size / (int)pow(2, i + 1), buf_size / pow(2, i + 1), MPI_FLOAT, rank + interval / 2,
        (rank + interval / 2) * 1000 + 1, MPI_COMM_WORLD, request + 1);
			// MPI_Isend(send_buf2, buf_size, MPI_FLOAT, rank + interval / 2, 
			// 	rank * 1000 + 2, MPI_COMM_WORLD, request + 1);
			MPI_Waitall(2, request, status);
			// MPI_Wait(&request2, &status2);
		}
		else {
			// MPI_Sendrecv(send_buf1 + buf_size / (int)pow(2, i + 1), buf_size / pow(2, i + 1), MPI_FLOAT,
	  //               rank - interval / 2, rank,
	  //               recv_buf1 + buf_size / (int)pow(2, i + 1), buf_size / pow(2, i + 1), MPI_FLOAT,
	  //               rank - interval / 2, rank - interval / 2,
	  //               MPI_COMM_WORLD, &status);

	  	MPI_Isend(send_buf1 + buf_size / (int)pow(2, i + 1), buf_size / pow(2, i + 1), MPI_FLOAT, rank - interval / 2, 
				rank * 1000 + 1, MPI_COMM_WORLD, request);
	  	MPI_Irecv(recv_buf1 + buf_size / (int)pow(2, i + 1), buf_size / pow(2, i + 1), MPI_FLOAT, rank - interval / 2,
        (rank - interval / 2) * 1000 + 1, MPI_COMM_WORLD, request + 1);
	  	// MPI_Irecv(recv_buf2, buf_size, MPI_FLOAT, rank - interval / 2,
	  	// 				(rank - interval / 2) * 1000 + 2, MPI_COMM_WORLD, request + 1);
			MPI_Waitall(2, request, status);
			// MPI_Wait(&request2, &status2);
		}
		interval = interval / 2;
		if (interval == 1)
			break;
	}

	timer.stop();
	SlaveTimeAndBandWidth(timer, buf_size);


#ifdef GPU_MEM
	cudaFree(send_buf1);
	cudaFree(send_buf2);
	cudaFree(recv_buf1);
	cudaFree(recv_buf2);
#else
	free(send_buf1);
	free(recv_buf1);
	free(send_buf2);
	free(recv_buf2);
#endif

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
	
	MPI_Status* status = new MPI_Status[size - 1];
	MPI_Request* request = new MPI_Request[size - 1];

	if (rank == 0) {
		for (int i = 1; i < size; i++)
			// MPI_Recv(buf, buf_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &recv_status);
			MPI_Irecv(buf, buf_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, request + (i - 1) );
		MPI_Waitall(size - 1, request, status);
		for (int i = 1; i < size; i++)
			MPI_Isend(buf, buf_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, request + (i - 1) );
			// MPI_Send(buf, buf_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
		MPI_Waitall(size - 1, request, status);
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

#ifndef COLLECTIVE
	free(status);
	free(request);
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

	// for (int i = 0; i < 3; i++) {
	//   std::cout << "test broadcast " << std::endl;
	//  	TestBroadCast<Dtype>(buf_size);
	//  }

//	for (int i = 0; i < 10; i++) {
//		std::cout << "test all reduce " << std::endl;
//		TestAllReduce<Dtype>(buf_size);	
//	}

//	for (int i = 0; i < 5; i++)
//		std::cout << std::endl;
	
	for (int i = 0; i < 10; i++) {
		std::cout << "test send recv " << std::endl;
		TestSendRecvDouble<Dtype>(buf_size);	
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
