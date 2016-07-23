# compilers
NVCC = /usr/local/cuda/bin/nvcc
CXX = /usr/bin/g++
MPICXX = /usr/local/bin/mpic++

# compiler flags
CFLAGS = -std=c++11
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lcuda -lglog -lnccl -lmpi 
INCFLAGS = -I/home/zjian/Code/novu_caffe_v2/caffe/include/ -I/usr/local/cuda/include

# # communication design test
# NCCLOBJ = ./test_nccl
# # MPIOBJ = ./test_mpi
# NCCLSRC = test_nccl_comm.cpp communicator.cpp

MPIEXE = test_mpi
MPISRC = test_mpi_comm.cpp communicator.cpp
MPIObJ = $(MPISRC:%.cpp=%.o)

%.o: %.cpp
	@echo Compiling $<
	$(NVCC) $(INCFLAGS) $(LDFLAGS) -c $< -o $@


# nccl:
# 	$(NVCC) $(CFLAGS) $(NCCLSRC) $(LDFLAGS) $(INCFLAGS) -o $(NCCLOBJ) 

mpi: $(MPIOBJ)
	$(MPICXX) $(CFLAGS) $(MPISRC) $(LDFLAGS) $(INCFLAGS) -o $(MPIEXE)
	scp ../../include/cluster/* zjian@192.168.0.245:/home/zjian/Code/novu_caffe_v2/caffe/src/cluster 
	scp ./* zjian@192.168.0.245:/home/zjian/Code/novu_caffe_v2/caffe/src/cluster
	mpirun --hostfile hosts.txt -np 2 ./$(MPIEXE)
