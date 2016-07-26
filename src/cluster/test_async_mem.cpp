#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <unistd.h>
#include <pthread.h>
#include <cassert>
#include <thread>
#include <vector>
#include "cluster/async_mem.hpp"

#define N_SEND 50
#define N_RECV 100 

template <typename Dtype>
void SendThreadEntry(AsyncMem<Dtype>& mem, std::vector<int>& res_vec, int thread_id) {
	// std::srand(seed);
	for (int i = 0; i < N_SEND; i++) {
		usleep(rand() % 100 + 5000) ;
		mem.Request(true, thread_id);
		Dtype* buf = mem.GetBuf();
		for (int j = 0; j < mem.GetBufSize(); j++) {
			buf[j] =  i * 2 + 1;
			assert(buf[j] == buf[0] );
		}
		/* record the buf content after this operation */
		res_vec.push_back(buf[0] );
		mem.Release();
	}
}


template <typename Dtype>
void RecvThreadEntry(AsyncMem<Dtype>& mem, std::vector<int>& res_vec, int thread_id) {
	// std::srand(seed + 1);
	for (int i = 0; i < N_RECV; i++) {
		usleep(rand() % 100);
		mem.Request(false, thread_id);
		Dtype* buf = mem.GetBuf();
		for (int j = 0; j < mem.GetBufSize(); j++) {
			buf[j] = i * 2;
			assert(buf[j] == buf[0] );
		}
		res_vec.push_back(buf[0] );

		mem.Release();
	}
}

/**
 * The test assert two things. 
 * The content in the mem buffer should be uniform.
 * By checking the result std::vector, we test whether it aligns 
 * with the order the job is posted (the prior task should not
 * be delayed). The values after each job will be incremented by 2. 
 * SendThread (prior) produce odd numbers while receiver produces even ones.
 * 
 */
template <typename Dtype>
void TestAsyncMemTwoTreads() {
	std::srand(time(NULL) );
	int64_t buf_size = std::rand() % 1000000;
	AsyncMem<Dtype> mem(2, buf_size);
	std::vector<int> res_vec;
	std::thread recv_thread(RecvThreadEntry<Dtype>, std::ref(mem), 
		std::ref(res_vec), 0);
	std::thread send_thread(SendThreadEntry<Dtype>, std::ref(mem), 
		std::ref(res_vec), 1);

	recv_thread.join();
	send_thread.join();

	std::vector<bool> debug_op = mem.GetDebugOp();
	assert(res_vec.size() == debug_op.size() );
	int odd = 1;
	int even = 0;
	for (int i = 0; i < res_vec.size(); i++) {
		std::cout << "job " << i << " " << debug_op[i] 
			<< " value " << res_vec[i] << std::endl;
		if (debug_op[i] ) {
			assert(res_vec[i] == odd);
			odd += 2;
		}
		else {
			assert(res_vec[i] == even);
			even += 2;
		}
	}
	std::cout << "AsyncMem test (one send and one receive) passed!" << std::endl;
}


int main(int argc, char** argv) {
	
	TestAsyncMemTwoTreads<float>();
	// TestAsyncMemTwoTreads<double>();

	return 0;
}