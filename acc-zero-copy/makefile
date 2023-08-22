	F90=nvfortran
	CC=nvcc
	#CUDAFLAGS= -Mcuda=cc70,nordc,maxregcount:64,ptxinfo -fast -Mfprelaxed -L/usr/local/cuda/lib64 -lnvToolsExt
	CUDAFLAGS= -Mcuda=cc80,nordc,maxregcount:64,ptxinfo -fast -Mfprelaxed 
	#CUDAFLAGS= -Mcuda=cc60,nordc,maxregcount:48,ptxinfo,loadcache:L1 -fast -Mfprelaxed
	F90FLAGS=-O3 -mp 
	CUDAFLAGS+= -acc
	#LINKFLAGS = -ta=tesla:pinned -lnvhpcwrapnvtx
	LINKFLAGS = -gpu=pinned -lnvhpcwrapnvtx



build: driver.o
	${F90} -o test driver.o ${F90FLAGS} ${CUDAFLAGS} ${LINKFLAGS}


driver.o: driver.F90
	${F90} -c -o driver.o driver.F90 ${F90FLAGS} ${CUDAFLAGS}

clean:
	rm -f test *.o *.lst *.mod

