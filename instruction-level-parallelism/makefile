# important flags here are -fast or -02. unroll or nounroll do not seem to make any difference.
	F90=nvfortran
	CUDAFLAGS = -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,lineinfo,unroll -fast -Mfprelaxed -L/usr/local/cuda/lib64 -lnvToolsExt
	#CUDAFLAGS = -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,lineinfo,nounroll -Mfprelaxed -L/usr/local/cuda/lib64 -lnvToolsExt
	CUDAFLAGS+= -acc=noautopar
	F90FLAGS=-gopt -mp -Minfo -Mnounroll -Mnovect
	LINKFLAGS = -ta=tesla:pinned


build: driver.o nvtx_mod.o
	${F90} -o ${comp}test driver.o ../nvtx/nvtx_mod.o ${F90FLAGS} ${CUDAFLAGS} ${LINKFLAGS}


driver.o: driver.F90 nvtx_mod.o
	${F90} -c -o driver.o driver.F90 ${F90FLAGS} ${CUDAFLAGS}

nvtx_mod.o: ../nvtx/nvtx_mod.F90
	${F90} -c -o ../nvtx/nvtx_mod.o ../nvtx/nvtx_mod.F90 ${F90FLAGS} ${CUDAFLAGS}

clean:
	rm -f ${comp}test *.o *.lst *.mod ../nvtx/*.o

