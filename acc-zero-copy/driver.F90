module transposes

  use cudafor


contains
  !-----------------------------------------------------------------------
  subroutine transpose_zero_copy_DtoH(A,At,nx,ny) 

    ! example of strided writes to an array that lives on the host

    ! variable declarations
    implicit none

    ! passed variables
    integer, intent(in) :: nx, ny
    real(kind=8), intent(in) :: A(nx,ny)
    real(kind=8), intent(out) :: At(ny,nx)    

    integer :: i,j

    !$acc parallel loop gang vector deviceptr(At) present(A)
    do j=1,ny
       do i=1,nx
          At(j,i) = A(i,j) ! uncoallesced read, coalesced write
       enddo
    enddo    

    return

  end subroutine transpose_zero_copy_DtoH
  

  subroutine transpose_zero_copy_HtoD(A,At,nx,ny) 

    ! example of strided writes to an array that lives on the host

    ! variable declarations
    implicit none

    ! passed variables
    integer, intent(in) :: nx, ny
    real(kind=8), intent(in) :: A(nx,ny)
    real(kind=8), intent(out) :: At(ny,nx)    

    integer :: i,j

    !$acc parallel loop gang vector deviceptr(A) present(At)
    do j=1,ny
       do i=1,nx
          At(j,i) = A(i,j) ! uncoallesced read, coalesced write
       enddo
    enddo    

    return

  end subroutine transpose_zero_copy_HtoD

  
  subroutine transpose(A,At,nx,ny) 

    ! example of strided writes to an array on the gpu that then has to be moved
    ! back to host.

    ! variable declarations
    implicit none

    ! passed variables
    real(kind=8), intent(in) :: A(:,:)
    real(kind=8), intent(out) :: At(:,:)
    
    integer, intent(in) :: nx, ny
    integer :: i,j

    !$acc kernels present(A,At)
    do j=1,ny
       do i=1,nx
          At(j,i) = A(i,j) ! coallesced read, uncoalesced write
       enddo
    enddo
    !$acc end kernels
    return

  end subroutine transpose

end module transposes



program main
  use cudafor
  use omp_lib
  use transposes
  implicit none
  CHARACTER(LEN=*), PARAMETER :: fmt = "(2X, I8, T15, F9.4 )"

  integer :: n,nx,ny, samples
  integer :: ierr,i, testnum, q
  real(kind=8), pinned, allocatable :: A(:,:),At(:,:)
  real(kind=8) :: t1, t2, T, mem
  real(kind=8),allocatable :: table(:,:)
  ! CUDA specific
  integer acc_clear_freelists

  character(len=255) :: matrixsize

  samples = 1

  !call get_environment_variable("MY_MATRIXSIZE",matrixsize)

  call get_command_argument(1,matrixsize)
  if(len_trim(matrixsize)==0) then
     write(0,*) "missing command line arg, ./a.out matrixsize"
     matrixsize="2048"
  endif

  ! string to integer conversion
  read(matrixsize,"(I8)") n

  print *, "Transposing matrix of size", n

  nx = n; ny = n;

  print*, "allocating"

  
  allocate( table(12*1024/128,3) )
  
  open (unit = 22, file = "/dev/stdout")
  write(22,*) "Matrix Transpose Benchmarking"
 
  
  write(22,"(A,T15,3(2X,A7,I2) )") "R+W Bytes (MB)", ("test",i,i=1,3)
  ! Total size of the data
  do n = 256, 12*1024, 128

     q = q + 1

     mem = 8*real(n*n,kind=8)/real(1000*1000*1000,kind=8)

     allocate( A(nx,ny), At(ny,nx) )

     ! The following dummy kernel establishes the GPU context, which would otherwise
     ! be done on the first time running the kernels that we are timing (skewing the timings).
     !$acc kernels
     !do nothing, dummy kernel. 
     !$acc end kernels  


     testnum=1
     !$acc wait
     ! compute the transpose in the typical way:
     t1 = omp_get_wtime()
     !$acc enter data create(A,At)

     do i=1, samples
        !$acc update device(A)
        call transpose(A,At,nx,ny) 
        !$acc update host(At)
        !$acc wait
     enddo
     !$acc exit data delete(A,At)
     !$acc wait
     
     t2 = omp_get_wtime()
     T = t2-t1  
     table(q,testnum)=2*mem/T


     ! needed to force OpenACC to really free the memory in the pool allocator.
     ! Without it CUDA Fortran allocations + OpenACC pool can run out of memory.
     ! Alternative is to export PGI_ACC_MEM_MANAGE=0 in env.
     ierr = acc_clear_freelists()     


     testnum=2
     !$acc wait
     ! do the DtoH zero copy way
     t1 = omp_get_wtime()
     !$acc enter data create(A)
     ! don't create device mirror version of At, just tell gpu it is valid memory on the GPU
     ! and use it with deviceptr(At) clause on !$acc kernels region.
     do i=1, samples
        !$acc update device(A)
        call transpose_zero_copy_DtoH(A,At,nx,ny)
        ! the above kernel also writes the data out to pinned host buffer At
        !$acc wait
     enddo
     !$acc exit data delete(A)
     !$acc wait
     
     t2 = omp_get_wtime()
     T = t2-t1
     table(q,testnum)=2*mem/T
     
     ierr = acc_clear_freelists()     

     testnum=3
     !$acc wait
     ! do the HtoD zero copy way
     t1 = omp_get_wtime()
     !$acc enter data create(At)
     do i=1, samples        
        call transpose_zero_copy_HtoD(A,At,nx,ny)
        !$acc update host(At)
        ! the above kernel reads from pinned host buffer A (zero copy of A is on the device)
        !$acc wait
     enddo
     !$acc exit data delete(At)
     !$acc wait
     
     t2 = omp_get_wtime()
     T = t2-t1
     table(q,testnum)=2*mem/T

     deallocate(A,At)
     
     ierr = acc_clear_freelists()     

     
     write(22,"(f9.4,T15,3(2X,f9.2))") 2*mem*1000, table(q,:)
     
  enddo
  

  
  print*, "completed"


end program main
