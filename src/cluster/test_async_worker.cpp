#include <mpi.h>
#include <thread>
#include <pthread.h>
#include <cassert>
#include "cluster/debug_utils.hpp"
#include "cluster/comm_utils.hpp"
#include "cluster/async_communicator.hpp"
#include "cluster/worker.hpp"


template <typename Dtype>
void Train() {
	/**
	 * count the number of GPUs available from this process
	 * and init all the workers.
	 */
	vector<int> gpu_ids;
	GetGpuIds(gpu_ids);
	std::vector<AsyncWorker<Dtype> > Workers;
	ncclUniqueId clique_id;
  NCCL_CHECK(ncclGetUniqueId(&clique_id) );
	for (int i = 0; i < gpu_ids.size(); i++) {
		// TODO Jian: add solvers
		SyncCommConfig<Dtype> sync_config(gpu_ids[i], clique_id);
		AsyncCommConfig<Dtype> async_config;
		Workers.push_back(AsyncWorker<Dtype> (sync_config, async_config) );
	}
	// start spawn process and compute
	for (int i = 0; i < gpu_ids.size(); i++)
		Workers[i].Run();
}


int main(int argc, char** argv) {
	int rank;
	int size;
	MPI_Init(NULL, NULL);

	Train<float>();
	
	MPI_Finalize();
}