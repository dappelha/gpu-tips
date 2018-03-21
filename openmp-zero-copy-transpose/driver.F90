module transposes

  use cudafor


contains
  !-----------------------------------------------------------------------
  subroutine transpose_zero_copy(A,At,nx,ny) 

    ! example of strided writes to an array that lives on the host

    ! variable declarations
    implicit none

    ! passed variables
    real(kind=8), intent(in) :: A(:,:)
    real(kind=8), intent(out) :: At(:,:)
    
    integer, intent(in) :: nx, ny
    integer :: i,j
    !$omp target teams distribute parallel do is_device_ptr(At) num_teams(2*56) thread_limit(1024)
    do j=1,ny
       do i=1,nx
          At(j,i) = A(i,j)
       enddo
    enddo

    return

  end subroutine transpose_zero_copy


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

    !$omp target teams distribute parallel do num_teams(2*56) thread_limit(1024)
    do j=1,ny
       do i=1,nx
          At(j,i) = A(i,j) ! coallesced write, uncoalesced read
       enddo
    enddo

    return

  end subroutine transpose



  subroutine transpose_zero_copy_split(A,At,nx,ny) 

    ! example of strided writes to an array that lives on the host

    ! variable declarations
    implicit none

    ! passed variables
    real(kind=8), intent(in) :: A(:,:)
    real(kind=8), intent(out) :: At(:,:)
    
    integer, intent(in) :: nx, ny
    integer :: i,j
    !$omp target teams distribute is_device_ptr(At) num_teams(2*56) thread_limit(1024)
    do j=1,ny
       !$omp parallel do
       do i=1,nx
          At(j,i) = A(i,j)
       enddo
    enddo

    return

  end subroutine transpose_zero_copy_split



  subroutine transpose_split(A,At,nx,ny) 

    ! example of strided writes to an array on the gpu that then has to be moved
    ! back to host.

    ! variable declarations
    implicit none

    ! passed variables
    real(kind=8), intent(in) :: A(:,:)
    real(kind=8), intent(out) :: At(:,:)
    
    integer, intent(in) :: nx, ny
    integer :: i,j

    !$omp target teams distribute num_teams(2*56) thread_limit(1024)
    do j=1,ny
       !$omp parallel do 
       do i=1,nx
          At(j,i) = A(i,j) ! uncoalesced write, coalesced read
       enddo
    enddo

    return

  end subroutine transpose_split


end module transposes



program main
  use cudafor
  use transposes
  implicit none

  integer :: threads, blocks

  integer :: n,nx,ny, i, j, samples
  real(kind=8), pinned, allocatable :: A(:,:),At(:,:)

  integer :: ierr

  character(len=255) :: matrixsize

  samples = 5

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

  allocate( A(nx,ny), At(ny,nx) )

  ! compute the transpose in the typical way:

  !$omp target enter data map(alloc:A,At)
  do i=1, samples
     call transpose(A,At,nx,ny)
  
     !$omp target update from(At)
  enddo
  !$omp target exit data map(delete:At)

  ierr = cudaDeviceSynchronize()

  ! doing the other way
  do i=1, samples
     call transpose_zero_copy(A,At,nx,ny)
  enddo

  print*, "completed"


end program main
