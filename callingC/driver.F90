program main
  use iso_c_binding
  use cudafor
  !use nvtx_mod

  implicit none

  interface 
     ! interface to c function which gives calls a C cuda kernel for a = a + b
     subroutine calling_routine_c(n,d_a,d_b) bind(c)
       use iso_c_binding
       use cudafor
       integer ( c_int ), value :: n
       integer ( c_int ),device :: d_a(*)
       integer ( c_int ),device :: d_b(*)
     end subroutine calling_routine_c
  end interface



  integer :: i,n
  integer, allocatable :: a(:,:), b(:,:)
  integer, allocatable, device :: d_a(:,:), d_b(:,:)

  integer :: istat, q
  integer, parameter :: nstreams=2
  integer(kind=cuda_stream_kind) :: streamid(nstreams), old_stream

  n = 1024*16

  ! host allocations
  allocate(a(n, nstreams), b(n, nstreams))
  ! device allocations
  allocate(d_a(n,nstreams), d_b(n, nstreams)) 

  !Set the values:
  a = 1
  b = 2
  
  do q=1,nstreams
     ! create the streams ahead of time.
     istat = cudaStreamCreate(streamid(q))
  enddo

  ! save the default stream
  old_stream = cudaforGetDefaultStream()       ! save the current default stream

  do q=1,nstreams
  !   istat = cudaforSetDefaultStream(streamid(q))   ! Set the default stream to streamid
     ! move data in the default stream (now non-blocking)
     istat = cudaMemCpy(d_a(1,q),a(1,q),n)
     istat = cudaMemCpy(d_b(1,q),b(1,q),n)
     call calling_routine_c(n, d_a(:,q), d_b(:,q) )
     print*, cudaGetErrorString( cudaGetLastError() )
     ! move the data back. a = a+b
     istat = cudaMemCpy(a(1,q),d_a(1,q),n)     
  enddo
  
  istat = cudaDeviceSynchronize()
  istat = cudaforSetDefaultStream(old_stream) ! restore the original default stream
  
  ! Check the data:
  do q=1,nstreams
     do i=1,n
        if(a(i,q) .ne. 3) then
           print *, "incorrect results at"
           print *, "a(",i,",",q,") = ", a(i,q)
           exit
        endif
     enddo
  enddo

  deallocate(d_a)
  deallocate(a)
  deallocate(d_b)
  deallocate(b)
  

  print*, "completed"


end program main
