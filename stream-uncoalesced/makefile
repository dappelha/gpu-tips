# PGI compiler
	F90=nvfortran
	CC=pgcc
	#CUDAFLAGS = -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,lineinfo,nounroll -fast -Mfprelaxed -L/usr/local/cuda/lib64 -lnvToolsExt
	CUDAFLAGS = -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,lineinfo,nounroll -Mfprelaxed -L/usr/local/cuda/lib64 -lnvToolsExt
	#CUDAFLAGS+= -Minfo=inline,accel -acc=noautopar
	CUDAFLAGS+= -acc=noautopar	
	F90FLAGS=-gopt -mp -Minfo -Mnounroll -Mnovect -fast
	#F90FLAGS=-gopt -mp -Minfo -Munroll -Mnovect
	LINKFLAGS = -ta=tesla:pinned


build: driver.o nvtx_mod.o
	${F90} -o ${comp}test driver.o ../nvtx/nvtx_mod.o ${F90FLAGS} ${CUDAFLAGS} ${LINKFLAGS}


driver.o: driver.f90 nvtx_mod.o
	${F90} -c -o driver.o driver.f90 ${F90FLAGS} ${CUDAFLAGS}

driver.f90: driver.F90
	${F90} -cpp -F driver.F90

nvtx_mod.o: ../nvtx/nvtx_mod.F90
	${F90} -c -o ../nvtx/nvtx_mod.o ../nvtx/nvtx_mod.F90 ${F90FLAGS} ${CUDAFLAGS}

clean:
	rm -f ${comp}test *.o *.lst *.mod ../nvtx/*.o

