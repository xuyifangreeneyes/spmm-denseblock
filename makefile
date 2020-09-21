CC = g++
NVCC = nvcc
CFLAGS = -std=c++14 -O3
NVFLAGS = -arch=sm_35
LIBS = -lcusparse

load_data.o: load_data.cc
	$(NVCC) $(CFLAGS) $(LIBS) -c -o $@ $^ 

utility.o: utility.cc
	$(NVCC) $(CFLAGS) $(LIBS) -c -o $@ $^

test_bsrmm.o: test_bsrmm.cu
	$(NVCC) $(CFLAGS) $(NVFLAGS) -c -o $@ $^ 

test_bsrmm: test_bsrmm.o load_data.o utility.o
	$(NVCC) $(CFLAGS) $(NVFLAGS) $(LIBS) -o $@ $^

test_csrmm.o: test_csrmm.cu
	$(NVCC) $(CFLAGS) $(NVFLAGS) -c -o $@ $^ 

test_csrmm: test_csrmm.o load_data.o utility.o
	$(NVCC) $(CFLAGS) $(NVFLAGS) $(LIBS) -o $@ $^

.PHONY: clean
clean:
	rm -rf *.o