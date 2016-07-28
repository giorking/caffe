# compilers
NVCC = /usr/local/cuda/bin/nvcc
CXX = /usr/bin/g++
MPICXX = /usr/local/bin/mpic++

# compiler flags
CFLAGS = -std=c++11
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lcuda -lglog -lnccl -lmpi 
INCFLAGS = -I/home/zjian/Code/novu_caffe_v2/caffe/include/ -I/usr/local/cuda/include
PTHREADFLAGS = -pthread

# all exe and obj
ALLEXE =
ALLOBJ =


# communication design test
NCCLEXE = test_nccl
NCCLSRC = test_nccl_comm.cpp sync_communicator.cpp
NCCLOBJ = $(NCCLSRC:%.cpp=%.o)
ALLEXE += $(NCCLEXE)
ALLOBJ += $(NCCLOBJ)

# sync communication test
SYNCCOMMEXE = test_sync_comm
SYNCCOMMSRC = test_sync_comm.cpp sync_communicator.cpp debug_utils.cpp
SYNCCOMMObJ = $(SYNCCOMMSRC:%.cpp=%.o)
ALLEXE += $(SYNCCOMMEXE)
ALLOBJ += $(SYNCCOMMOBJ)

# async memory test
ASYNCMEMEXE = test_async_mem
ASYNCMEMSRC = test_async_mem.cpp async_mem.cpp
ASYNCMEMOBJ = $(ASYNCMEMSRC:%.cpp=%.o)
ALLEXE += $(ASYNCMEMEXE)
ALLOBJ += $(ASYNCMEMOBJ)

# async communication test
ASYNCCOMMEXE = test_async_comm
ASYNCCOMMSRC = test_async_comm.cpp async_mem.cpp async_communicator.cpp
ASYNCCOMMOBJ = $(ASYNCCOMMSRC:%.cpp=%.o)
ALLEXE += $(ASYNCCOMMEXE)
ALLOBJ += $(ASYNCCOMMOBJ)


%.o: %.cpp
	@echo Compiling $<
	$(NVCC) $(CFLAGS) $(INCFLAGS) $(LDFLAGS) -c $< -o $@


nccl: $(NCCLOBJ)
	# @echo Linking for NCCL test
	$(MPICXX) $(CFLAGS) $(NCCLSRC) $(LDFLAGS) $(INCFLAGS) -o $(NCCLEXE) 
	./$(NCCLEXE)


sync_comm: $(SYNCCOMMOBJ)
	$(MPICXX) $(CFLAGS) $(SYNCCOMMSRC) $(LDFLAGS) $(INCFLAGS) -o $(SYNCCOMMEXE)
	scp ../../include/cluster/* zjian@192.168.0.245:/home/zjian/Code/novu_caffe_v2/caffe/src/cluster 
	scp ./* zjian@192.168.0.245:/home/zjian/Code/novu_caffe_v2/caffe/src/cluster
	mpirun --hostfile hosts.txt -np 2 ./$(SYNCCOMMEXE)


async_mem: $(ASYNCMEMOBJ)
	$(NVCC) $(CFLAGS) $(ASYNCMEMSRC) $(LDFLAGS) $(INCFLAGS) -o $(ASYNCMEMEXE)
	./$(ASYNCMEMEXE)


async_comm: $(ASYNCCOMMOBJ)
	@echo Linking
	$(MPICXX) $(CFLAGS) $(PTHREADFLAGS) $(ASYNCCOMMOBJ) $(LDFLAGS) $(INCFLAGS) -o $(ASYNCCOMMEXE)
	scp ../../include/cluster/* zjian@192.168.0.245:/home/zjian/Code/novu_caffe_v2/caffe/src/cluster 
	scp ./* zjian@192.168.0.245:/home/zjian/Code/novu_caffe_v2/caffe/src/cluster
	mpirun --hostfile hosts.txt -np 8 ./$(ASYNCCOMMEXE)


clean:
	rm -f $(ALLOBJ)
	rm -f $(ALLEXE)
