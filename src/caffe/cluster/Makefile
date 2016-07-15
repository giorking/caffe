# compilers
NVCC = /usr/local/cuda/bin/nvcc
CXX = /usr/bin/g++

# compiler flags
CFLAGS = -std=c++11
LDFLAGS = -lcuda
INCFLAGS = -I../../../include/caffe/cluster -I../../../include/caffe/util -I../../../nccl/src

# communication design test
COMMOBJ = ./comm_test
SRC = comm_main.cpp communicator.cpp ../../../nccl/src/*.cu 

comm:
	$(NVCC) $(CFLAGS) $(SRC) $(LDFLAGS) $(INCFLAGS) -o $(COMMOBJ) 
