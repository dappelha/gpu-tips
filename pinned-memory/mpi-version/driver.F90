program main
  use cudafor
  use nvtx_mod
  use omp_lib
 
  implicit none

  include "mpif.h"

  integer :: threads, blocks

  integer :: i, N
  real(kind=8), allocatable :: A(:)
  real(kind=8), pinned, allocatable :: p_A(:)
  real(kind=8), device, allocatable :: d_A(:)

  real(kind=8) :: t1, t2, T, mem

  integer :: ierr

  character(len=30) :: str
  CHARACTER(LEN=*), PARAMETER :: fmt = "(2X, A, T30, E8.3, T40, F8.3 )"

  integer :: numtasks,taskid

  call MPI_INIT (ierr)
  
  call MPI_COMM_SIZE (MPI_COMM_WORLD,numtasks,ierr)
  call MPI_COMM_RANK (MPI_COMM_WORLD,taskid,ierr)

  N = 1024*1024*1024/2
  
  !mem = real(8*N,kind=8)/1D9
  mem = 8*real(N,kind=8)/real(1024*1024*1024,kind=8)


  !write(*,fmt) "Array size: ", real(8*N*1D-9), " GB"
  write(*,*) "Array size: ", mem , " GB"


  write(*,*) "Allocation Timings:"

  call nvtxStartRange("regular host allocation")
  t1 = omp_get_wtime()
  allocate( A(N) )
  t2 = omp_get_wtime()
  call nvtxEndRange
  write(*,fmt) "regular host allocation: ", t2-t1 



  call nvtxStartRange("pinned host allocation")
  t1 = omp_get_wtime()
  allocate( p_A(N) )
  t2 = omp_get_wtime()
  call nvtxEndRange

  write(*,fmt) "pinned host allocation: ", t2-t1 



  str = "device allocation"
  call nvtxStartRange(str)
  t1 = omp_get_wtime()
  allocate( d_A(N) )
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange
  
  write(*,fmt) str, t2-t1

  ! REGULAR PAGEABLE MEMORY:
  write(*,*) "REGULAR PAGEABLE MEMORY:"

  ! omp way (FAILS FOR 8 GB ARRAY):

  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

  str = "omp HtoD"
  call nvtxStartRange(str,1)
  t1 = omp_get_wtime()
!  !$omp target enter data map(to:A)
!  !$omp target exit data map(delete:A)
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange
  write(*,fmt) str, t2-t1, mem/(t2-t1)

  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

  ! Implicit CUDA way
  str = "Implicit CUDA HtoD"
  call nvtxStartRange("Implicit CUDA HtoD",2)
  t1 = omp_get_wtime()
  d_A = A
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange
  write(*,fmt) str, t2-t1, mem/(t2-t1)

  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

  ! Another CUDA way
  str = "Explicit CUDA HtoD"
  call nvtxStartRange("Explicit CUDA HtoD",3)
  t1 = omp_get_wtime()
  ierr = cudaMemcpy(d_A,A,size(A))  !<------ this seems to not be executing
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange  
  write(*,fmt) str, t2-t1, mem/(t2-t1)

  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

  ! USING PINNED MEMORY
  write(*,*) "USING PINNED MEMORY"


  ! omp way (WORKS FOR 8 GB ARRAY):
  str = "omp HtoD"
  call nvtxStartRange("omp HtoD",1)
  t1 = omp_get_wtime()
!  !$omp target enter data map(to:p_A)
!  !$omp target exit data map(delete:p_A)
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange
  write(*,fmt) str, t2-t1, mem/(t2-t1)

  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

  ! Implicit CUDA way
  str = "Implicit CUDA HtoD"
  call nvtxStartRange("Implicit CUDA HtoD",2)
  t1 = omp_get_wtime()
  d_A = p_A
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange
  write(*,fmt) str, t2-t1, mem/(t2-t1)

  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

  ! Another CUDA way
  str = "Explicit CUDA HtoD"
  call nvtxStartRange("Explicit CUDA HtoD",3)
  t1 = omp_get_wtime()
  ierr = cudaMemcpy(d_A,p_A,size(p_A))
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange  
  write(*,fmt) str, t2-t1, mem/(t2-t1)

  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

  ! REGULAR PAGEABLE MEMORY:
  write(*,*) "REGULAR PAGEABLE MEMORY:"

  ! omp way (FAILS FOR 8 GB ARRAY):

  str = "omp HtoD"
  call nvtxStartRange(str,1)
  t1 = omp_get_wtime()
!  !$omp target enter data map(to:A)
!  !$omp target exit data map(delete:A)
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange
  write(*,fmt) str, t2-t1, mem/(t2-t1)

  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

  ! Implicit CUDA way
  str = "Implicit CUDA HtoD"
  call nvtxStartRange("Implicit CUDA HtoD",2)
  t1 = omp_get_wtime()
  d_A = A
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange
  write(*,fmt) str, t2-t1, mem/(t2-t1)

  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

  ! Another CUDA way
  str = "Explicit CUDA HtoD"
  call nvtxStartRange("Explicit CUDA HtoD",3)
  t1 = omp_get_wtime()
  ierr = cudaMemcpy(d_A,A,size(A))  !<------ this seems to not be executing
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange  
  write(*,fmt) str, t2-t1, mem/(t2-t1)

  call MPI_BARRIER(MPI_COMM_WORLD,ierr)

  print*, "completed"

  call MPI_FINALIZE(ierr)

end program main
