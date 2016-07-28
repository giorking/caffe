#include <mpi.h>
#include <thread>
#include <pthread.h>
#include <cassert>
#include "cluster/comm_utils.hpp"
#include "cluster/async_communicator.hpp"


/**
 * async communicator. 
 * the order of operation goes in the following
 * 1. inter-group thread : receive from last to mpi async mem
 * (barrier)
 * 2. intra-group thread : add delta to mpi async mem
 * (barrier)
 * 3. intra-group thread : take from mpi async mem to compute
 * 		inter-group thread : send out async mem. The next receive 
 * 		operation directly follows the send. We need mutex to prevent 
 * 		the this receive overlaps with the intra-group operation to 
 * 		"take from mpi async mem to compute"
 */
template <typename Dtype>
void IntraGroupThreadEntry(AsyncCommunicator<Dtype>* comm, int n_iter) {
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	// wait for communicator setup
	comm->ThreadBarrierWait();
	// prevent trigger send overlaps with recv thread in wait on the same buf. 
	comm->ThreadBarrierWait();

	comm->SendRecvLoop(n_iter);
}


template <typename Dtype>
void InterGroupThreadEntry(AsyncCommunicator<Dtype>* comm, int n_iter) {
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int n_group = size / N_PROC_PER_GROUP;
	MPI_Datatype type = DtypeToMPIDtype<Dtype>::type;
	// wait for communicator setup
	comm->ThreadBarrierWait();

	AsyncMem<Dtype>* mem = comm->GetAsyncMem();

	// last group need to initilize the ring based computation
	Dtype* comp_buf = new Dtype [mem->GetBufSize() ];
	memset(comp_buf, 0, sizeof(Dtype) * mem->GetBufSize() );
	MPI_Comm* comm_ptr = comm->GetMPIComm();
	// wait for everything in new MPI group to setup
	// MPI_Barrier(MPI_COMM_WORLD);

	if (rank / N_PROC_PER_GROUP == n_group - 1)
		MPI_Send(comp_buf, mem->GetBufSize(), type,
			0, 0, *comm_ptr);
	// prevent trigger send overlaps with recv thread in wait on the same buf. 
	comm->ThreadBarrierWait();


	vector<Dtype> test_res;
	for (int i = 0; i < n_iter; i++) {
		// b1: wait until comm thread finish receive
		comm->ThreadBarrierWait();

		// commit delta to mem_
		for (int i = 0; i < mem->GetBufSize(); i++)
			(mem->GetBuf() )[i] += 1;
		// b2: wait until finish update delta to mem_
		comm->ThreadBarrierWait();

		// read mem_ for compute
		memcpy(comp_buf, mem->GetBuf(), sizeof(Dtype) * mem->GetBufSize() );	

		// b3 : wait until finish read mem_ out before recv
		comm->ThreadBarrierWait();

		// simulate a computation of roughly 500ms
		usleep(500000);

		// // verify the comp_buf has uniform content
		// for (int i = 0; i < mem->GetBufSize(); i++)
		// 	assert(comp_buf[0] == comp_buf[i] );
		test_res.push_back(comp_buf[0] );

		std::cout << "rank " << rank << " round " << i << " done " << std::endl;
	}

	// verify the pattern of the result
	assert(test_res[0] = rank / N_PROC_PER_GROUP + 1);
	std::cout << "rank " << rank << " value " << test_res[0] << std::endl;
	for (int i = 1; i < test_res.size(); i++) {
		std::cout << "rank " << rank << " value " << test_res[i] << std::endl;
		assert(test_res[i] - (test_res[i - 1] ) == n_group);
	}

	if (rank / N_PROC_PER_GROUP == 0) {
		MPI_Status recv_status;
		MPI_Recv(comp_buf, mem->GetBufSize(), type,
			n_group - 1, ASYNC_MSG, *comm_ptr, &recv_status);
	}

	delete comp_buf;
}


/**
 * Test async communication. We check the model is updated
 * in a round robin fashion. Each time we increment the model with 1.
 * If we have four machine in the communication group. We need
 * to see that the value got from mem_ for computing is 0, 4, 8, 12...
 */
template <typename Dtype>
void TestJoinComm() {
	MPI_Init(NULL, NULL);

	// we simulate the real setting where we have roughly 200M parameter
	AsyncMem<Dtype> mem(2, 20000000);
	AsyncCommConfig<Dtype> config(true);
	AsyncCommunicator<Dtype> comm(config);
	comm.AttachAsyncMem(&mem);
	int n_iter = 10;

	std::thread intra_group_thread(IntraGroupThreadEntry<Dtype>, &comm, n_iter);
	std::thread inter_group_thread(InterGroupThreadEntry<Dtype>, &comm, n_iter);

	intra_group_thread.join();
	inter_group_thread.join();

	comm.Destroy();

	std::cout << "Asynchronized communication test passed!" << std::endl;

	MPI_Finalize();
}



int main(int argc, char** argv) {
	TestJoinComm<float>();
	// TestJoinComm<double>();

	return 0;
}