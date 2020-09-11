/usr/local/opt/llvm/bin/clang -fopenmp -L/usr/local/opt/llvm/lib -std=c++14 -lc++ -O3 -o main_no_omp main.cpp

/home/ubuntu/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/clang -fopenmp -L/home/ubuntu/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/lib -std=c++14 -lc++ -O3 -o main_omp main.cpp

/home/ubuntu/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/clang -fopenmp -L/home/ubuntu/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/lib -std=c++14 -lstdc++ -lm -O3 -o main_omp main.cpp

nvcc -gencode arch=compute_30,code=sm_30 \
     -gencode arch=compute_35,code=sm_35 \
     -gencode arch=compute_50,code=[sm_50,compute_50] \
     -gencode arch=compute_52,code=[sm_52,compute_52] \
     -std=c++14 --compiler-options -fPIC -c -o hello.o main.cu

g++ -L/usr/local/cuda/lib -lcutil -lcuda -lcudart -lcublas -o hello_cuda hello.o


nvcc -gencode arch=compute_61,code=sm_61 \
     -gencode arch=compute_70,code=sm_70 \
     -std=c++14 --compiler-options -fPIC -c -o spmm.o spmm.cu

nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64  -lcuda -lcudart -lcublas -o hello2 hello_kk.o

g++ -std=c++14 -fPIC -c -o matrix.o matrix.cc

nvcc -gencode arch=compute_61,code=sm_61 -std=c++14 --compiler-options -fPIC -c -o spmm.o spmm.cu

nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64  -lcuda -lcudart -lcublas -o hello_2 matrix.o spmm.o

nvcc try_cusparse.cu -lcusparse -o hello_cusparse

nvcc run_csr.cu -lcusparse -O3 -o run_csr
nvcc run_bsr.cu -lcusparse -O3 -o run_bsr

/home/ubuntu/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/clang -fopenmp -L/home/ubuntu/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/lib -std=c++14 -lstdc++ -lm -O3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64  -lcuda -lcudart -lcublas -o test_block_cublas test_block_cublas.cu

nvcc -fopenmp -L/home/ubuntu/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/lib -std=c++14 -lstdc++ -lm -O3 -I/usr/local/cuda/include -L/usr/local/cuda/lib64  -lcuda -lcudart -lcublas -o test_block_cublas test_block_cublas.cu


/home/ubuntu/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/clang -fopenmp -L/home/ubuntu/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/lib -std=c++14 -lstdc++ -lm -lcublas_static -lculibos -lcudart_static -lpthread -ldl -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -O3 -o test_block_cublas test_block_cublas.cu

sudo -E env PATH=$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH /usr/local/cuda/NsightCompute-1.0/target/linux-desktop-glibc_2_11_3-glx-x64/nv-nsight-cu-cli -f --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./myspmm_backup 0.002 64 128 rocsparse
/usr/local/cuda/NsightCompute-1.0/target/linux-desktop-glibc_2_11_3-glx-x64/nv-nsight-cu-cli -f --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum ./myspmm_backup 0.002 64 128 rocsparse