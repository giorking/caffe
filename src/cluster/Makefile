# compilers
NVCC = /usr/local/cuda/bin/nvcc
CXX = /usr/bin/g++

# compiler flags
CFLAGS = -std=c++11
LDFLAGS = -lcuda -lglog -lnccl
INCFLAGS = -I/home/zjian/Code/novu_caffe_v2/caffe/include/

# communication design test
NCCLOBJ = ./test_nccl
MPIOBJ = ./test_mpi
NCCLSRC = test_nccl_comm.cpp communicator.cpp
MPISRC = test_mpi_comm.cpp communicator.cpp


nccl:
	$(NVCC) $(CFLAGS) $(NCCLSRC) $(LDFLAGS) $(INCFLAGS) -o $(NCCLOBJ) 

mpi:
	$(NVCC) $(CFLAGS) $(MPISRC) $(LDFLAGS) $(INCFLAGS) -o $(MPIOBJ)