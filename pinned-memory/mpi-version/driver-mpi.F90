module basicmovement

  use cudafor
  use nvtx_mod
  use omp_lib

  integer :: samples =10
  integer, parameter :: gigabyte = 1024*1024*1024/8
  integer, parameter :: megabyte = 1024*1024/8
  integer :: N != 8*gigabyte
  character(len=30) :: str
  CHARACTER(LEN=*), PARAMETER :: fmt = "(2X, A, T30, G8.3, T40, F8.3 )"
  integer :: color

contains




  subroutine omp_HtoD(h_dummy,mem)

    implicit none

    real(kind=8), allocatable, intent(in) :: h_dummy(:)
    real(kind=8), intent(in) :: mem

    real(kind=8) :: t1, t2, T  
    integer :: ierr, i
#if defined(__ibmxl__)

    str = "omp device allocation"
    call nvtxStartRange(str)
    T=0
    do i = 1, samples
       !$omp target exit data map(delete:h_dummy)
       ierr = cudaDeviceSynchronize()
       t1 = omp_get_wtime()
       !$omp target enter data map(alloc:h_dummy)
       ierr = cudaDeviceSynchronize()
       t2 = omp_get_wtime()
       T = T + (t2-t1)
    enddo
    call nvtxEndRange
    write(*,fmt) str, T/samples
  
    ! OpenMP memcopy
    str = "omp HtoD"
    color = modulo(color+1,7)
    call nvtxStartRange(str,color)
    t1 = omp_get_wtime()
    do i=1,samples
       !$omp target update to(h_dummy)
       ierr = cudaDeviceSynchronize()
    enddo
    t2 = omp_get_wtime()
    call nvtxEndRange
    write(*,fmt) str, t2-t1, samples*mem/(t2-t1)

    !$omp target exit data map(delete:h_dummy)
    ierr = cudaDeviceSynchronize()

#endif
  end subroutine omp_HtoD


  subroutine omp_DtoH(h_dummy,mem)

    implicit none

    real(kind=8), allocatable, intent(inout) :: h_dummy(:)
    real(kind=8), intent(in) :: mem

    real(kind=8) :: t1, t2, T  
    integer :: ierr, i
#if defined(__ibmxl__)

    str = "omp device allocation"
    call nvtxStartRange(str)
    T=0
    do i = 1, samples
       !$omp target exit data map(delete:h_dummy)
       ierr = cudaDeviceSynchronize()
       t1 = omp_get_wtime()
       !$omp target enter data map(alloc:h_dummy)
       ierr = cudaDeviceSynchronize()
       t2 = omp_get_wtime()
       T = T + (t2-t1)
    enddo
    call nvtxEndRange
    write(*,fmt) str, T/samples
  
    ! OpenMP memcopy
    str = "omp DtoH"
    color = modulo(color+1,7)
    call nvtxStartRange(str,color)
    t1 = omp_get_wtime()
    do i=1,samples
       !$omp target update from(h_dummy)
       ierr = cudaDeviceSynchronize()
    enddo
    t2 = omp_get_wtime()
    call nvtxEndRange
    write(*,fmt) str, t2-t1, samples*mem/(t2-t1)

    !$omp target exit data map(delete:h_dummy)
    ierr = cudaDeviceSynchronize()

#endif
  end subroutine omp_DtoH


  subroutine acc_HtoD(h_dummy,mem)

    implicit none

    real(kind=8), allocatable, intent(inout) :: h_dummy(:)
    real(kind=8), intent(in) :: mem

    real(kind=8) :: t1, t2, T  
    integer :: ierr, i

#if defined(__PGI)

    str = "acc device allocation"
    call nvtxStartRange(str)
    T=0
    do i = 1, samples
       if(i .gt. 1) then
          !$acc exit data delete(h_dummy)
          ierr = cudaDeviceSynchronize()
       endif
       t1 = omp_get_wtime()
       !$acc enter data create(h_dummy)
       ierr = cudaDeviceSynchronize()
       t2 = omp_get_wtime()
       T = T + (t2-t1)
    enddo
    call nvtxEndRange
    write(*,fmt) str, T/samples
  
    ! OpenACC memcopy
    str = "acc HtoD"
    color = modulo(color+1,7)
    call nvtxStartRange(str,color)
    t1 = omp_get_wtime()
    do i=1,samples
       !$acc update self(h_dummy)
       ierr = cudaDeviceSynchronize()
    enddo
    t2 = omp_get_wtime()
    call nvtxEndRange
    write(*,fmt) str, t2-t1, samples*mem/(t2-t1)

    !$acc exit data delete(h_dummy)
    ierr = cudaDeviceSynchronize()
#endif
  end subroutine acc_HtoD



  subroutine acc_DtoH(h_dummy,mem)

    implicit none

    real(kind=8), allocatable, intent(inout) :: h_dummy(:)
    real(kind=8), intent(in) :: mem

    real(kind=8) :: t1, t2, T  
    integer :: ierr, i

#if defined(__PGI)

    str = "acc device allocation"
    call nvtxStartRange(str)
    T=0
    do i = 1, samples
       if(i .gt. 1) then
          !$acc exit data delete(h_dummy)
          ierr = cudaDeviceSynchronize()
       endif
       t1 = omp_get_wtime()
       !$acc enter data create(h_dummy)
       ierr = cudaDeviceSynchronize()
       t2 = omp_get_wtime()
       T = T + (t2-t1)
    enddo
    call nvtxEndRange
    write(*,fmt) str, T/samples
  
    ! OpenACC memcopy
    str = "acc DtoH"
    color = modulo(color+1,7)
    call nvtxStartRange(str,color)
    t1 = omp_get_wtime()
    do i=1,samples
       !$acc update host(h_dummy)
       ierr = cudaDeviceSynchronize()
    enddo
    t2 = omp_get_wtime()
    call nvtxEndRange
    write(*,fmt) str, t2-t1, samples*mem/(t2-t1)

    !$acc exit data delete(h_dummy)
    ierr = cudaDeviceSynchronize()
#endif
  end subroutine acc_DtoH



  subroutine implicitCUDA_HtoD(d_dummy,h_dummy,mem)
    
    implicit none

    real(kind=8), allocatable, intent(in) :: h_dummy(:)
    real(kind=8), device, allocatable, intent(inout) :: d_dummy(:)
    real(kind=8), intent(in) :: mem

    real(kind=8) :: t1, t2, T  
    integer :: ierr, i

    ! Implicit CUDA way
    str = "Implicit CUDA HtoD"
    color = modulo(color+1,7)
    call nvtxStartRange(str,color)
    t1 = omp_get_wtime()
    do i=1,samples
       d_dummy = h_dummy
       ierr = cudaDeviceSynchronize()
    enddo
    t2 = omp_get_wtime()
    call nvtxEndRange
    write(*,fmt) str, t2-t1, samples*mem/(t2-t1)

  end subroutine implicitCUDA_HtoD


  subroutine explicitCUDA_HtoD(d_dummy,h_dummy,mem)
    
    implicit none

    real(kind=8), allocatable, intent(in) :: h_dummy(:)
    real(kind=8), device, allocatable, intent(inout) :: d_dummy(:)
    real(kind=8), intent(in) :: mem

    real(kind=8) :: t1, t2, T  
    integer :: ierr, i

    ! Explicit CUDA memcopy
    str = "Explicit CUDA HtoD"
    color = modulo(color+1,7)
    call nvtxStartRange(str,color)
    t1 = omp_get_wtime()
    do i=1,samples
       ierr = cudaMemcpy(d_dummy,h_dummy,size(h_dummy))
       ierr = cudaDeviceSynchronize()
    enddo
    t2 = omp_get_wtime()
    call nvtxEndRange
    write(*,fmt) str, t2-t1, samples*mem/(t2-t1)

  end subroutine explicitCUDA_HtoD


  subroutine explicitCUDA_DtoH(d_dummy,h_dummy,mem)
    
    implicit none

    real(kind=8), allocatable, intent(inout) :: h_dummy(:)
    real(kind=8), device, allocatable, intent(inout) :: d_dummy(:)
    real(kind=8), intent(in) :: mem

    real(kind=8) :: t1, t2, T  
    integer :: ierr, i

    ! Explicit CUDA memcopy
    str = "Explicit CUDA DtoH"
    color = modulo(color+1,7)
    call nvtxStartRange(str,color)
    t1 = omp_get_wtime()
    do i=1,samples
       ierr = cudaMemcpy(h_dummy,d_dummy,size(h_dummy))
       ierr = cudaDeviceSynchronize()
    enddo
    t2 = omp_get_wtime()
    call nvtxEndRange
    write(*,fmt) str, t2-t1, samples*mem/(t2-t1)

  end subroutine explicitCUDA_DtoH



end module basicmovement


program main
  use cudafor
  use nvtx_mod
  use omp_lib
  use basicmovement

  implicit none

  include "mpif.h"

  real(kind=8), allocatable :: A(:)
  real(kind=8), pinned, allocatable :: p_A(:)
  real(kind=8), device, allocatable :: d_A(:)

  real(kind=8) :: t1, t2, T, mem

  integer :: ierr, i
  integer :: numtasks,taskid
  character(len=20) :: arg


  call MPI_INIT (ierr)
  
  call MPI_COMM_SIZE (MPI_COMM_WORLD,numtasks,ierr)
  call MPI_COMM_RANK (MPI_COMM_WORLD,taskid,ierr)

  
  !mem = real(8*N,kind=8)/1D9

  call get_command_argument(1, arg)
  
  if ( len_trim(arg) == 0) arg="1024"
  read(arg,"(I6)") i
  N=i*megabyte

  mem = 8*real(N,kind=8)/real(1024*1024*1024,kind=8)


  !write(*,fmt) "Array size: ", real(8*N*1D-9), " GB"
  print*, "****Pinned vs non-pinned memory transfer comparison***"
  write(*,'(A,F8.4,A)') "Array size: ", mem , " GB "
  write(*,'(A,I6)')  "Number of samples: ", samples
  write(*,"(2X,A, T30, A, T40, A )") "Test", "sec", "Bandwidth GB/s"

  write(*,*) "Allocation Timings:"


  call nvtxStartRange("regular host allocation")
  T = 0
  do i = 1, samples
     if (allocated ( A ) ) deallocate( A )
     t1 = omp_get_wtime()
     allocate( A(N) )
     t2 = omp_get_wtime()
     T = T + (t2-t1)
  enddo
  call nvtxEndRange
  write(*,fmt) "regular host allocation: ", T/samples



  call nvtxStartRange("pinned host allocation")
  T=0
  do i = 1, samples
     if (allocated ( p_A ) ) deallocate( p_A )
     t1 = omp_get_wtime()
     allocate( p_A(N) )
     t2 = omp_get_wtime()
     T = T + (t2-t1)
  enddo
  call nvtxEndRange
  write(*,fmt) "pinned host allocation: ", T/samples


  str = "cuda device allocation"
  call nvtxStartRange(str)
  T = 0
  do i = 1, samples
     if (allocated ( d_A ) ) deallocate( d_A )
     ierr = cudaDeviceSynchronize()
     t1 = omp_get_wtime()
     allocate( d_A(N) )
     ierr = cudaDeviceSynchronize()
     t2 = omp_get_wtime()
     T = T + (t2-t1)
  enddo
  call nvtxEndRange  
  write(*,fmt) str, T/samples


  ! REGULAR PAGEABLE MEMORY:
  write(*,*) "REGULAR PAGEABLE MEMORY:"

  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

  ! Implicit CUDA way (similar time to explicit)
  !call implicitCUDA_HtoD(d_A,A,mem)
  
  ! Another CUDA way
  call explicitCUDA_HtoD(d_A,A,mem)


  ! USING PINNED MEMORY
  write(*,*) "USING PINNED MEMORY"
  
  ! Implicit CUDA way
  ! call implicitCUDA_HtoD(d_A,p_A,mem)

  ! Another CUDA way
  call explicitCUDA_HtoD(d_A,p_A,mem)


  deallocate(d_A)
  ierr=cudaDeviceSynchronize()


!  call MPI_BARRIER(MPI_COMM_WORLD,ierr)
!*************************************************

  ! REGULAR PAGEABLE MEMORY:
  write(*,*) "REGULAR PAGEABLE MEMORY:"

  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

  ! omp way
  call omp_HtoD(A,mem)
  call omp_DtoH(A,mem)

  ! openacc way:
  call acc_HtoD(A,mem)
  call acc_DtoH(A,mem)
  
  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

  str = "cuda device allocation"
  call nvtxStartRange(str)
  T = 0
  do i = 1, samples
     if (allocated ( d_A ) ) deallocate( d_A )
     ierr = cudaDeviceSynchronize()
     t1 = omp_get_wtime()
     allocate( d_A(N) )
     ierr = cudaDeviceSynchronize()
     t2 = omp_get_wtime()
     T = T + (t2-t1)
  enddo
  call nvtxEndRange  
  write(*,fmt) str, T/samples

  ! Implicit CUDA way
  !call implicitCUDA_HtoD(d_A,A,mem)
  
  ! Another CUDA way
  call explicitCUDA_HtoD(d_A,A,mem)
  call explicitCUDA_DtoH(d_A,A,mem)

  deallocate(d_A)
  ierr=cudaDeviceSynchronize()


  ! USING PINNED MEMORY
  write(*,*) "USING PINNED MEMORY"

  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

  ! omp way
  call omp_HtoD(p_A,mem)
  call omp_DtoH(p_A,mem)

  ! openacc way:
  call acc_HtoD(p_A,mem)
  call acc_DtoH(p_A,mem)

  call MPI_BARRIER(MPI_COMM_WORLD,ierr)
  
  str = "cuda device allocation"
  call nvtxStartRange(str)
  T = 0
  do i = 1, samples
     if (allocated ( d_A ) ) deallocate( d_A )
     ierr = cudaDeviceSynchronize()
     t1 = omp_get_wtime()
     allocate( d_A(N) )
     ierr = cudaDeviceSynchronize()
     t2 = omp_get_wtime()
     T = T + (t2-t1)
  enddo
  call nvtxEndRange  
  write(*,fmt) str, T/samples

  ! Implicit CUDA way
  !call implicitCUDA_HtoD(d_A,p_A,mem)

  ! Another CUDA way
  call explicitCUDA_HtoD(d_A,p_A,mem)
  call explicitCUDA_DtoH(d_A,p_A,mem)

  deallocate(d_A)
  ierr=cudaDeviceSynchronize()

  print*, "completed"

  call MPI_FINALIZE(ierr)

end program main
