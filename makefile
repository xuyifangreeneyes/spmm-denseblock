CC = g++
NVCC = nvcc
CFLAGS = -fPIC -Wall -std=c++14 -O3
NVFLAGS = -std=c++14 -O3 -arch=sm_35
LIBS = -L/usr/local/cuda/lib64 -lcusparse -lcudart -lcuda
INCLUDE = -I/usr/local/cuda/include

load_data.o: load_data.cc
	$(CC) $(INCLUDE) $(CFLAGS) $(LIBS) -c -o $@ $^ 

utility.o: utility.cc
	$(CC) $(INCLUDE) $(CFLAGS) $(LIBS) -c -o $@ $^

reorder_strategy.o: reorder_strategy.cc
	$(CC) $(INCLUDE) $(CFLAGS) $(LIBS) -c -o $@ $^

reorder_graph.o: reorder_graph.cc
	$(CC) $(INCLUDE) $(CFLAGS) $(LIBS) -c -o $@ $^

reorder_graph: reorder_graph.o reorder_strategy.o load_data.o utility.o
	$(NVCC) $(INCLUDE) $(NVFLAGS) $(LIBS) -o $@ $^

rabbit_reorder.o: rabbit_reorder.cc
	$(CC) $(INCLUDE) $(CFLAGS) $(LIBS) -c -o $@ $^

rabbit_reorder: rabbit_reorder.o reorder_strategy.o load_data.o utility.o
	$(NVCC) $(INCLUDE) $(NVFLAGS) $(LIBS) -o $@ $^

test_bsrmm.o: test_bsrmm.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $^ 

test_bsrmm: test_bsrmm.o load_data.o utility.o
	$(NVCC) $(NVFLAGS) $(LIBS) -o $@ $^

test_csrmm.o: test_csrmm.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $^ 

test_csrmm: test_csrmm.o load_data.o utility.o
	$(NVCC) $(NVFLAGS) $(LIBS) -o $@ $^

run_csrmm.o: run_csrmm.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $^ 

run_csrmm: run_csrmm.o load_data.o utility.o
	$(NVCC) $(NVFLAGS) $(LIBS) -o $@ $^

run_bsrmm.o: run_bsrmm.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $^ 

run_bsrmm: run_bsrmm.o load_data.o utility.o
	$(NVCC) $(NVFLAGS) $(LIBS) -o $@ $^

divide.o: divide.cu
	$(NVCC) $(NVFLAGS) -c -o $@ $^ 

divide: divide.o load_data.o utility.o
	$(NVCC) $(NVFLAGS) $(LIBS) -o $@ $^

.PHONY: clean
clean:
	rm -rf *.o