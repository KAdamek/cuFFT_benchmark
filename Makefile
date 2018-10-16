INC := -I${CUDA_HOME}/include
LIB := -L${CUDA_HOME}/lib64 -lcudart -lcuda -lcufft

# use this compilers
# g++ just because the file write
GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_61 --ptxas-options=-v -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -Wextra $(INC)

ANALYZE = cuFFT_benchmark.exe


ifdef reglim
NVCCFLAGS += --maxrregcount=$(reglim)
endif

ifdef fastmath
NVCCFLAGS += --use_fast_math
endif

all: clean analyze

analyze: cuFFT_benchmark.o cuFFT.o Makefile
	$(NVCC) -o $(ANALYZE) cuFFT_benchmark.o cuFFT.o $(LIB) $(NVCCFLAGS) 

cuFFT.o: timer.h utils_cuda.h FFT_clases.h
	$(NVCC) -c cuFFT.cu $(NVCCFLAGS)
	
cuFFT_benchmark.o: cuFFT_benchmark.cpp
	$(GCC) -c cuFFT_benchmark.cpp $(GCC_OPTS)

clean:	
	rm -f *.o *.~ $(ANALYZE)


