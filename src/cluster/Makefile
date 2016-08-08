# compilers
NVCC = /usr/local/cuda/bin/nvcc
CXX = /usr/bin/g++
MPICXX = /usr/local/bin/mpic++
MPICXXSERVER = /home/zjian/mpich-install/bin/mpic++

# compiler flags
CFLAGS = -std=c++11
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas -lglog -lnccl -lmpi 
INCFLAGS = -I/home/zjian/Code/novu_caffe_v2/caffe/include/ -I/usr/local/cuda/include \
	-I/home/zjian/mpich-install/include
PTHREADFLAGS = -pthread
ERRFLAG = -fmax-errors=3

# all exe and obj
ALLEXE =
ALLOBJ =


# communication design test
NCCLEXE = test_nccl
NCCLSRC = test_nccl_multi.cpp comm_utils.cpp
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

# sync worker test
SYNCWORKEREXE = test_sync_worker
SYNCWORKERSRC = test_sync_worker.cpp sync_communicator.cpp \
	worker.cpp debug_utils.cpp timer.cpp solver.cpp comm_utils.cpp
SYNCWORKEROBJ = $(SYNCWORKERSRC:%.cpp=%.o)
# SYNCWORKEROBJ += $(SYNCWORKERSRC:%.cu=%.cu.o)
ALLEXE += $(SYNCWORKEREXE)
ALLOBJ += $(SYNCWORKEROBJ)

# async worker test
ASYNCWORKEREXE = test_async_worker
ASYNCWORKERSRC = test_async_worker.cpp async_mem.cpp sync_communicator.cpp \
	async_communicator.cpp worker.cpp async_worker.cpp debug_utils.cpp \
	timer.cpp solver.cpp comm_utils.cpp
ASYNCWORKEROBJ = $(ASYNCWORKERSRC:%.cpp=%.o)
ALLEXE += $(ASYNCWORKEREXE)
ALLOBJ += $(ASYNCWORKEROBJ)


%.o: %.cpp
	@echo Compiling $<
	$(NVCC) $(CFLAGS) $(INCFLAGS) $(LDFLAGS) -c $< -o $@


# %.cu.o: %.cu
# 	@echo Compiling $<
# 	$(NVCC) $(CFLAGS) $(INCFLAGS) $(LDFLAGS) -c $< -o $@	


nccl: $(NCCLOBJ)
	# @echo Linking for NCCL test
	$(MPICXXSERVER) $(CFLAGS) $(NCCLSRC) $(LDFLAGS) $(INCFLAGS) -o $(NCCLEXE) 
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


sync_worker: $(SYNCWORKEROBJ)
	@echo Linking
	$(MPICXX) $(CFLAGS) $(PTHREADFLAGS) $(SYNCWORKEROBJ) $(LDFLAGS) $(INCFLAGS) -o $(SYNCWORKEREXE)
	scp ../../include/cluster/* zjian@192.168.0.245:/home/zjian/Code/novu_caffe_v2/caffe/src/cluster 
	scp ./* zjian@192.168.0.245:/home/zjian/Code/novu_caffe_v2/caffe/src/cluster
	mpirun --hostfile hosts.txt -np 2 ./$(SYNCWORKEREXE)


async_worker: $(ASYNCWORKEROBJ)
	@echo Linking
	$(MPICXX) $(CFLAGS) $(PTHREADFLAGS) $(ASYNCWORKEROBJ) $(LDFLAGS) $(INCFLAGS) -o $(ASYNCWORKEREXE)
	scp ../../include/cluster/* zjian@192.168.0.245:/home/zjian/Code/novu_caffe_v2/caffe/src/cluster 
	scp ./* zjian@192.168.0.245:/home/zjian/Code/novu_caffe_v2/caffe/src/cluster
	mpirun --hostfile hosts.txt -np 2 ./$(ASYNCWORKEREXE)


sync_worker_server: $(SYNCWORKEROBJ)
	@echo Linking
	$(MPICXXSERVER) $(CFLAGS) $(PTHREADFLAGS) $(SYNCWORKEROBJ) $(LDFLAGS) $(INCFLAGS) -o $(SYNCWORKEREXE)
	scp ../../include/cluster/* zjian@192.168.0.132:/home/zjian/Code/novu_caffe_v2/caffe/src/cluster 
	scp ./* zjian@192.168.0.132:/home/zjian/Code/novu_caffe_v2/caffe/src/cluster
	mpirun --hostfile hosts.txt -np 1 ./$(SYNCWORKEREXE)


async_worker_server: $(ASYNCWORKEROBJ)
	@echo Linking
	$(MPICXXSERVER) $(CFLAGS) $(PTHREADFLAGS) $(ASYNCWORKEROBJ) $(LDFLAGS) $(INCFLAGS) -o $(ASYNCWORKEREXE)
	scp ../../include/cluster/* zjian@192.168.0.131:/home/zjian/Code/novu_caffe_v2/caffe/src/cluster 
	scp ./* zjian@192.168.0.131:/home/zjian/Code/novu_caffe_v2/caffe/src/cluster
	mpirun --hostfile hosts.txt -np 2 ./$(ASYNCWORKEREXE)



clean:
	rm -f $(ALLOBJ)
	rm -f $(ALLEXE)
