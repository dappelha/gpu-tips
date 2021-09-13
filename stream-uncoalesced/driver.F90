module bandwidthtest

  ! This code test the bandwidth achieved
  ! when memory accesses by consecutive threads
  ! are not coalesced into the same cache line
  ! Always move the same amount of data total data

! Consider reading data from array A
! and writing data into B with a varying stride
  use cudafor
  use nvtx_mod
  use omp_lib

!  CHARACTER(LEN=*), PARAMETER :: fmt = "(2X, T30, I8, T40, G8.3 )"
  CHARACTER(LEN=*), PARAMETER :: fmt = "(2X, I8, T15, F8.4 )"

contains

  

  subroutine copyAtoB(stride,unroll)

    implicit none
    integer,intent(in) :: stride
    logical, intent(in) :: unroll
    

    real(kind=8), pinned, allocatable :: A(:,:), B(:,:)
    real(kind=8) :: t1, t2, T, mem

    integer :: ierr, n, blocks, threads
    logical, save :: firsttime=.true.
    
    ! Total size of the data
    n = 128*1024*1024/stride

    allocate(A(stride,n),B(stride,n))

    !$acc enter data create(A,B)



    if(unroll .eq. .true.) then 
       select case (stride)

       case default
          call copyAtoB_nounroll(A,B,n,stride)
       case(1)
          call copyAtoB_unroll1(A,B,n)
       case(2)
          call copyAtoB_unroll2(A,B,n)
       case(3)
          call copyAtoB_unroll3(A,B,n)
       case(4)
          call copyAtoB_unroll4(A,B,n)
       case(5)
          call copyAtoB_unroll5(A,B,n)
       case(6)
          call copyAtoB_unroll6(A,B,n)
       case(7)
          call copyAtoB_unroll7(A,B,n)
       case(8)
          call copyAtoB_unroll8(A,B,n)
       case(9)
          call copyAtoB_unroll9(A,B,n)
       case(10)
          call copyAtoB_unroll10(A,B,n)
       case(11)
          call copyAtoB_unroll11(A,B,n)
       case(12)
          call copyAtoB_unroll12(A,B,n)
       end select

    else
       call copyAtoB_nounroll(A,B,n,stride)
    endif

    !$acc exit data delete(A,B)
    
    deallocate(A,B)

  end subroutine copyAtoB

  subroutine copyAtoB_nounroll(A,B,n,stride)
    implicit none
    integer, intent(in) :: stride
    #include "copyAtoB_guts.F90"
  end subroutine copyAtoB_nounroll

  subroutine copyAtoB_unroll1(A,B,n)
    implicit none
    integer, parameter :: stride=1
    #include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll1
  
  subroutine copyAtoB_unroll2(A,B,n)
    implicit none
    integer, parameter :: stride=2
    #include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll2

  subroutine copyAtoB_unroll3(A,B,n)
    implicit none
    integer, parameter :: stride=3
    #include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll3

  subroutine copyAtoB_unroll4(A,B,n)
    implicit none
    integer, parameter :: stride=4
    #include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll4

  subroutine copyAtoB_unroll5(A,B,n)
    implicit none
    integer, parameter :: stride=5
    #include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll5

  subroutine copyAtoB_unroll6(A,B,n)
    implicit none
    integer, parameter :: stride=6
    #include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll6

  subroutine copyAtoB_unroll7(A,B,n)
    implicit none
    integer, parameter :: stride=7
    #include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll7

  subroutine copyAtoB_unroll8(A,B,n)
    implicit none
    integer, parameter :: stride=8
    #include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll8

  subroutine copyAtoB_unroll9(A,B,n)
    implicit none
    integer, parameter :: stride=9
    #include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll9

  subroutine copyAtoB_unroll10(A,B,n)
    implicit none
    integer, parameter :: stride=10
    #include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll10

  subroutine copyAtoB_unroll11(A,B,n)
    implicit none
    integer, parameter :: stride=11
    #include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll11

  subroutine copyAtoB_unroll12(A,B,n)
    implicit none
    integer, parameter :: stride=12
    #include "copyAtoB_guts.F90"
  end subroutine copyAtoB_unroll12

  
end module bandwidthtest


program main

  use cudafor
  use nvtx_mod
  use bandwidthtest
  implicit none
  character(len=20) :: buffer
  integer :: stride, stride_start, stride_end, istat

  ! first dummy kernel acuires cuda context, so that doesn't affect the later timings. 
  !$acc kernels
  ! nothing
  !$acc end kernels
  
  !open (unit = 22, file = "data.txt")
  open (unit = 22, file = "/dev/stdout")
  !write(*,fmt) "Array size: ", real(8*N*1D-9), " GB"
  print*, "****Memory copy bandwidth test***"


  

  ! stride start and stop are runtime
  call get_environment_variable("stridestart", buffer, status=istat)
  if (istat .ge. 1) then
     print*, "Setting stridestart env variable failed,"
     print*, "using a default stridestart of 1."
     stride_start=1
  else
     read( buffer, '(I8)' )  stride_start
     print*, "stride_start = ", stride_start
  endif
     
  call get_environment_variable("strideend", buffer, status=istat)
  if (istat .ge. 1) then
     print*, "Setting strideend env variable failed,"
     print*, "using a default strideend of 12."
     stride_end=12
  else
     read( buffer, '(I8)' )  stride_end
     print*, "stride_end = ", stride_end
  endif



  
  print*, "****with compile time parameter (compiler should unroll)***"
  write(22,"(2X,A, T15, A, T30)") "stride",  "Bandwidth GB/s"
  ! measure bandwidth with unrolling
  do stride=stride_start,stride_end
     call copyAtoB(stride,unroll=.true.)
  enddo

  print*, "****Without compile time parameter***"
  write(22,"(2X,A, T15, A, T30)") "stride",  "Bandwidth GB/s"
  ! measure without unrolling
  do stride=stride_start,stride_end
     call copyAtoB(stride,unroll=.false.)
  enddo


  !call copyAtoB(stride=5)
  !call copyAtoB(stride=8)


  close(22)

  print*, "completed, wrote output to data.txt"


end program main
