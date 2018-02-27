module basicmovement

  use cudafor
  use nvtx_mod
  use omp_lib

  character(len=30) :: str
  CHARACTER(LEN=*), PARAMETER :: fmt = "(2X, A, T30, E8.3, T40, F8.3 )"
  integer :: color
  integer :: samples =1
contains


  subroutine omp_HtoD(h_dummy,mem)
    
    implicit none

    real(kind=8), allocatable, intent(in) :: h_dummy(:)
    real(kind=8), intent(in) :: mem

    real(kind=8) :: t1, t2, T  
    integer :: ierr, i



    ! omp way (pageable memory FAILS FOR 8 GB ARRAY):
  
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

  end subroutine omp_HtoD


  subroutine implicitCUDA_HtoD(d_dummy,h_dummy,mem)
    
    implicit none

    real(kind=8), allocatable, intent(in) :: h_dummy(:)
    real(kind=8), device, allocatable, intent(inout) :: d_dummy(:)
    real(kind=8), intent(in) :: mem

    real(kind=8) :: t1, t2, T  
    integer :: ierr, i

    ! Implicit CUDA way
    str = "Implicit CUDA HtoD"
    color = 1!modulo(color+1,7)
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
    color = 1!modulo(color+1,7)
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



end module basicmovement


program main
  use cudafor
  use nvtx_mod
  use omp_lib
  use basicmovement

  implicit none

  integer :: threads, blocks

  integer :: i, N
  real(kind=8), allocatable :: A(:)
  real(kind=8), pinned, allocatable :: p_A(:)
  real(kind=8), device, allocatable :: d_A(:)

  real(kind=8) :: t1, t2, T, mem

  integer :: ierr

  !character(len=30) :: str
  !CHARACTER(LEN=*), PARAMETER :: fmt = "(2X, A, T30, E8.3, T40, F8.3 )"

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



  str = "cuda device allocation"
  call nvtxStartRange(str)
  t1 = omp_get_wtime()
  allocate( d_A(N) )
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange  
  write(*,fmt) str, t2-t1


  str = "omp device allocation (A)"
  call nvtxStartRange(str)
  t1 = omp_get_wtime()
  !$omp target data map(alloc:A)
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange
  write(*,fmt) str, t2-t1

  str = "omp device allocation (p_A)"
  call nvtxStartRange(str)
  t1 = omp_get_wtime()
  !$omp target data map(alloc:p_A)
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange
  write(*,fmt) str, t2-t1
  



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

  ! openacc way:
  str = "openacc HtoD"
  call nvtxStartRange(str,5)
  t1 = omp_get_wtime()
  !$acc enter data copyin(A)
  !$acc exit data delete(A)
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange
  write(*,fmt) str, t2-t1, mem/(t2-t1)



  ! Implicit CUDA way
  call implicitCUDA_HtoD(d_A,A,mem)

  ! str = "Implicit CUDA HtoD"
  ! call nvtxStartRange("Implicit CUDA HtoD",2)
  ! t1 = omp_get_wtime()
  ! d_A = A
  ! ierr = cudaDeviceSynchronize()
  ! t2 = omp_get_wtime()
  ! call nvtxEndRange
  ! write(*,fmt) str, t2-t1, mem/(t2-t1)


  ! Another CUDA way
  str = "Explicit CUDA HtoD"
  call nvtxStartRange("Explicit CUDA HtoD",3)
  t1 = omp_get_wtime()
  ierr = cudaMemcpy(d_A,A,size(A))  !<------ this seems to not be executing
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange  
  write(*,fmt) str, t2-t1, mem/(t2-t1)



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

  ! openacc way:
  str = "openacc HtoD"
  call nvtxStartRange(str,5)
  t1 = omp_get_wtime()
  !$acc enter data copyin(p_A)
  !$acc exit data delete(p_A)
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange
  write(*,fmt) str, t2-t1, mem/(t2-t1)

  ! Implicit CUDA way

  call implicitCUDA_HtoD(d_A,p_A,mem)

  ! str = "Implicit CUDA HtoD"
  ! call nvtxStartRange("Implicit CUDA HtoD",2)
  ! t1 = omp_get_wtime()
  ! d_A = p_A
  ! ierr = cudaDeviceSynchronize()
  ! t2 = omp_get_wtime()
  ! call nvtxEndRange
  ! write(*,fmt) str, t2-t1, mem/(t2-t1)

  ! Another CUDA way
  str = "Explicit CUDA HtoD"
  call nvtxStartRange("Explicit CUDA HtoD",3)
  t1 = omp_get_wtime()
  ierr = cudaMemcpy(d_A,p_A,size(p_A))
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange  
  write(*,fmt) str, t2-t1, mem/(t2-t1)


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

  ! openacc way:
  str = "openacc HtoD"
  call nvtxStartRange(str,5)
  t1 = omp_get_wtime()
  !$acc enter data copyin(A)
  !$acc exit data delete(A)
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange
  write(*,fmt) str, t2-t1, mem/(t2-t1)

  ! Implicit CUDA way
  str = "Implicit CUDA HtoD"
  call nvtxStartRange("Implicit CUDA HtoD",2)
  t1 = omp_get_wtime()
  d_A = A
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange
  write(*,fmt) str, t2-t1, mem/(t2-t1)


  ! Another CUDA way
  str = "Explicit CUDA HtoD"
  call nvtxStartRange("Explicit CUDA HtoD",3)
  t1 = omp_get_wtime()
  ierr = cudaMemcpy(d_A,A,size(A))  !<------ this seems to not be executing
  ierr = cudaDeviceSynchronize()
  t2 = omp_get_wtime()
  call nvtxEndRange  
  write(*,fmt) str, t2-t1, mem/(t2-t1)


  !$omp target exit data map(delete:A)
  !$omp target exit data map(delete:p_A)
  print*, "completed"


end program main
