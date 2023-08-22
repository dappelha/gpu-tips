comp=pgi

ifeq '${comp}' 'pgi'
	# PGI compiler
	export OMPI_FC=nvfortran
	F90=mpif90
	CC=nvc
	CUDAFLAGS= -Mcuda=cc80,nordc,maxregcount:64,ptxinfo -fast -Mfprelaxed -L/usr/local/cuda/lib64
	#CUDAFLAGS= -Mcuda=cc60,nordc,maxregcount:48,ptxinfo,loadcache:L1 -fast -Mfprelaxed
	F90FLAGS=-O3 -mp
	CUDAFLAGS+= -acc
	LINKFLAGS = -ta=tesla:pinned -lnvhpcwrapnvtx
else
	# xlcuf compiler
	export OMPI_FC=xlf_r
	F90=mpif90
	CC=xlc_r
	#CUDAFLAGS= -qcuda -qtgtarch=sm_60 -W@,"-v,--maxrregcount=48" -qpath=IL:/home/dappelh/ibmcmp/specialfixes
	CUDAFLAGS= -qcuda -qtgtarch=sm_70 -W@,"-v,--maxrregcount=64" -lnvToolsExt
	CUDAFLAGS+= -qcheck -qsigtrap -g1
	#F90FLAGS=-O3 -qhot=novector -qsimd=auto -qarch=pwr8 -qtune=pwr8 -qsmp=omp -qoffload -lessl
	F90FLAGS=-O3 -qhot=novector -qsimd=auto -qarch=pwr9 -qtune=pwr9 -qsmp=omp -qoffload #-lomp
endif

	MPI_COMPILE_FLAGS = $(shell mpif90 --showme:compile)
	MPI_LINK_FLAGS = $(shell mpif90 --showme:link)


build: driver-mpi.o
	${F90} -o ${comp}test driver-mpi.o ${F90FLAGS} ${CUDAFLAGS} ${LINKFLAGS}


driver-mpi.o: driver-mpi.F90
	${F90} -c -o driver-mpi.o driver-mpi.F90 ${F90FLAGS} ${CUDAFLAGS}

clean:
	rm -f ${comp}test *.o *.lst *.mod

