comp=pgi

COMPUTE_CAPABILITY=60

CUDA_HOME=/usr/local/cuda-9.1

PGI_HOME=/shared/opt/pgi/linuxpower/18.3

FLIBS += -L$(PGI_HOME)/lib -L$(CUDA_HOME)/lib64 -lpgf90rtl -lpgf90 -lpgf90_rpm1 -lpgf902 -lpgf90rtl \
       -lpgftnrtl -lpgkomp -lcudapgi -lcudafor -lcudart -lcudaforblas

# needed for pgi 18.3
FLIBS += -lcudafor2 -lpgm

# other needed libraries (pgi 18.1 bug, pgc needs to link after all cuda links)
FLIBS += -lpgc  -lrt -ldl -lpthread -lmass -lm
FLIBS += -lmass_simdp8


CLIBS = -L$(CUDA_HOME)/lib64 -lstdc++

#INCPATH         = $(MPI_INC_PATH) -I$(CUDA_HOME)/include

F90FLAGS    = -Mcuda=cc$(COMPUTE_CAPABILITY),nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.1 -fast -Mfprelaxed -O3 -Kpic

LINKFLAGS       = -g -Wl,-rpath=$(PGI_HOME)/lib:$(CUDA_HOME)/lib64

F90=pgf90
CC=nvcc




#build: driver.o c_code.o GPU_add.o
#	${F90} ${F90FLAGS} -o ${comp}test driver.o c_code.o GPU_add.o  ${LINKFLAGS}

build: driver.o c_code.o GPU_add.o
	${F90} ${F90FLAGS} ${LINKFLAGS} -o ${comp}test driver.o c_code.o GPU_add.o  ${FLIBS} ${CLIBS}


driver.o: driver.F90
	${F90} -c ${F90FLAGS} -o driver.o driver.F90 

#nvtx_mod.o: ../nvtx/nvtx_mod.F90
#	${F90} -c -o ../nvtx/nvtx_mod.o ../nvtx/nvtx_mod.F90 ${F90FLAGS} ${CUDAFLAGS}

%.o : %.cu
	nvcc --maxrregcount=64 -gencode=arch=compute_70,code=sm_70 --resource-usage -Xptxas -v -O3 -c $< -o $@

clean:
	rm -f ${comp}test *.o *.lst *.mod ../nvtx/*.o

