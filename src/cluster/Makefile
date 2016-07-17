# compilers
NVCC = /usr/local/cuda/bin/nvcc
CXX = /usr/bin/g++

# compiler flags
CFLAGS = -std=c++11
LDFLAGS = -lcuda -lglog
INCFLAGS = -I/home/zjian/Code/novu_caffe_v2/caffe/include/

# communication design test
COMMOBJ = ./comm_test
SRC = comm_main.cpp communicator.cpp /home/zjian/Code/novu_caffe_v2/caffe/include/nccl/src/*.cu 

comm:
	$(NVCC) $(CFLAGS) $(SRC) $(LDFLAGS) $(INCFLAGS) -o $(COMMOBJ) 
