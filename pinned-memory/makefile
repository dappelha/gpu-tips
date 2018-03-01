comp=pgi

ifeq '${comp}' 'pgi'
	# PGI compiler
	F90=pgf90
	CC=pgcc
	CUDAFLAGS= -Mcuda=cc60,nordc,maxregcount:64,ptxinfo -fast -Mfprelaxed -L/usr/local/cuda/lib64 -lnvToolsExt
	#CUDAFLAGS= -Mcuda=cc60,nordc,maxregcount:48,ptxinfo,loadcache:L1 -fast -Mfprelaxed
	F90FLAGS=-O3 -mp 
	CUDAFLAGS+= -acc
	LINKFLAGS = -ta=tesla:pinned
else
	# xlcuf compiler
	F90=xlf_r
	CC=xlc_r
	#CUDAFLAGS= -qcuda -qtgtarch=sm_60 -W@,"-v,--maxrregcount=48" -qpath=IL:/home/dappelh/ibmcmp/specialfixes
	CUDAFLAGS= -qcuda -qtgtarch=sm_70 -W@,"-v,--maxrregcount=64" -lnvToolsExt
	CUDAFLAGS+= -qcheck -qsigtrap -g1
	#F90FLAGS=-O3 -qhot=novector -qsimd=auto -qarch=pwr8 -qtune=pwr8 -qsmp=omp -qoffload -lessl
	F90FLAGS=-O3 -qhot=novector -qsimd=auto -qarch=pwr9 -qtune=pwr9 -qsmp=omp -qoffload -lomp
endif



build: driver.o nvtx_mod.o
	${F90} -o ${comp}test driver.o ../nvtx/nvtx_mod.o ${F90FLAGS} ${CUDAFLAGS} ${LINKFLAGS}


driver.o: driver.F90 nvtx_mod.o
	${F90} -c -o driver.o driver.F90 ${F90FLAGS} ${CUDAFLAGS}

nvtx_mod.o: ../nvtx/nvtx_mod.F90
	${F90} -c -o ../nvtx/nvtx_mod.o ../nvtx/nvtx_mod.F90 ${F90FLAGS} ${CUDAFLAGS}

run: test
	./test

prof: test
	nvprof ./test


clean:
	rm -f ${comp}test *.o *.lst *.mod ../nvtx/*.o

