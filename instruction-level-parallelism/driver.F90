module bandwidthtest

! This code test the bandwidth achieved
! under conditions of insuficient thread 
! level parallelism to saturate the GPU
! and how instruction level parallelism 
! can help and what is that relationship? 
! 

! Consider reading data from array A
! and writing data into B when A and B are 
! sized not big enough use all threads in a GPU.

  use cudafor
  use nvtx_mod
  use omp_lib

!  CHARACTER(LEN=*), PARAMETER :: fmt = "(2X, T30, I8, T40, G8.3 )"
  CHARACTER(LEN=*), PARAMETER :: fmt = "(2X,I8, T15, I8, T30, I8, T45, I8, T60, I8, T75, F8.4 )"

contains

  

  subroutine copyAtoB(ilp,unroll)

    implicit none
    integer,intent(in) :: ilp
    logical, intent(in) :: unroll
    

    real(kind=8), pinned, allocatable :: A(:,:), B(:,:)
    real(kind=8) :: t1, t2, T, mem

    integer :: ierr, n, blocks, threads

    n = 100*80*1024/ilp

    allocate(A(n,ilp),B(n,ilp))

    !$acc enter data create(A,B)

    !$acc kernels
    A(:,:) = 1.0d+0
    !$acc end kernels


    if(unroll .eq. .true.) then 
       select case (ilp)

       case default
          call copyAtoB_nounroll(A,B,n,ilp)
       case(2)
          call copyAtoB_unroll2(A,B,n)
       case(3)
          call copyAtoB_unroll3(A,B,n)
       case(4)
          call copyAtoB_unroll4(A,B,n)
       case(5)
          call copyAtoB_unroll5(A,B,n)
       case(8)
          call copyAtoB_unroll8(A,B,n)
       end select

    else
       call copyAtoB_nounroll(A,B,n,ilp)
    endif


    deallocate(A,B)

  end subroutine copyAtoB

  subroutine copyAtoB_guts(A,B,n,ilp)
    implicit none
    integer :: ilp
    integer :: n
    real(kind=8),intent(inout) :: A(n,*), B(n,*)

    integer :: i,j, blocks, threads, tot_threads
    real(kind=8) :: t1, t2, T, mem


    tot_threads=80*1024 ! this is as many blocks*threads as we need to examine.

    !do threads = 32, 512, 512-32
    !do blocks = 0, tot_threads/threads, tot_threads/threads/160
          threads = 32
          blocks = 320
    !      if (blocks .eq. 0) cycle
          t1 = omp_get_wtime()
          !$acc parallel loop gang vector num_gangs(blocks) vector_length(threads)
          do i = 1, n
             do j = 1, ilp ! This loop will be unrolled if ilp is compile time constant.
                B(i,j) = A(i,j)
             enddo
          enddo
          !$acc end parallel
          t2 = omp_get_wtime()
          T = t2-t1

          !mem = 8*real(n,kind=8)/real(1024*1024*1024,kind=8)
          mem = 8*ilp*real(n,kind=8)/real(1000*1000*1000,kind=8)



          !write(22,fmt) n*ilp, blocks, threads, blocks*threads, ilp, 2*mem/T
          write(22,fmt) blocks, threads, blocks*threads, ilp, ilp*blocks*threads, 2*mem/T

    !   enddo
    !enddo

  end subroutine copyAtoB_guts

  subroutine copyAtoB_nounroll(A,B,n,ilp)
    implicit none
    integer, intent(in) :: ilp
    include "copyAtoB_guts.F90"
  end subroutine copyAtoB_nounroll


  subroutine copyAtoB_unroll2(A,B,n)
    implicit none
    integer, parameter :: ilp=2
    include "copyAtoB_guts.F90"

  end subroutine copyAtoB_unroll2


  subroutine copyAtoB_unroll3(A,B,n)
    implicit none
    integer, parameter :: ilp=3
    include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll3

  subroutine copyAtoB_unroll4(A,B,n)
    implicit none
    integer, parameter :: ilp=4
    include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll4

  subroutine copyAtoB_unroll5(A,B,n)
    implicit none
    integer, parameter :: ilp=5
    include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll5


  subroutine copyAtoB_unroll8(A,B,n)
    implicit none
    integer, parameter :: ilp=8
    include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll8

end module bandwidthtest


program main

  use cudafor
  use nvtx_mod
  use bandwidthtest
  implicit none
  character(len=20) :: buffer
  integer :: ilp, ilp_start, ilp_end

  !open (unit = 22, file = "data.txt")
  open (unit = 22, file = "/dev/stdout")
  !write(*,fmt) "Array size: ", real(8*N*1D-9), " GB"
  print*, "****Memory copy bandwidth***"
  write(22,"(2X,A, T15, A, T30, A, T45, A, T60, A, T75, A )") "blocks", "threads", "tot threads", "ilp", "ilp*tot_threads", "Bandwidth GB/s"

  call get_environment_variable("ilpstart", buffer)
  read( buffer, '(I8)' )  ilp_start
  call get_environment_variable("ilpend", buffer)
  read( buffer, '(I8)' )  ilp_end

  print*, "****with compile time parameter***"

  ! measure bandwidth with unrolling
  do ilp=ilp_start,ilp_end
     call copyAtoB(ilp,unroll=.true.)
  enddo

  print*, "****Without compile time parameter***"

  ! measure without unrolling
  do ilp=ilp_start,ilp_end
     call copyAtoB(ilp,unroll=.false.)
  enddo


  !call copyAtoB(ilp=5)
  !call copyAtoB(ilp=8)


  close(22)

  print*, "completed, wrote output to data.txt"


end program main
