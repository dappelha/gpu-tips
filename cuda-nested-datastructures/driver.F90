module datastructures

  use cudafor
  use nvtx_mod

  ! this is data structure you start with
  type, public :: element_type
     integer                           :: Nnodes ! need on gpu
     real(kind=8)                      :: volume ! not needed on gpu
     real(kind=8), allocatable, pinned :: x(:)   ! need on gpu
     real(kind=8), allocatable, pinned :: y(:)   ! need on gpu
     real(kind=8), allocatable, pinned :: val(:)
     real(kind=8), allocatable         :: old(:) ! not needed on gpu
  end type element_type

  ! need to create a duplicate version for the GPU where
  ! members are device variables. Can also "trim" parts
  ! of the data structure that will not be used on the GPU.
  ! For example suppose volume is not used in GPU calculations.
  type, public :: GPUelement_type
     integer :: Nnodes
     real(kind=8), device, pointer, contiguous :: x(:) !could use d_x here
     real(kind=8), device, pointer, contiguous :: y(:)
     real(kind=8), device, allocatable :: val(:)
  end type GPUelement_type

  ! original needed element
  !type(element_type), allocatable :: element(:)

  ! This is the host structure with host members
  !type(element_type), pinned, allocatable :: element(:)
  type(element_type), pointer :: element(:)
  ! To copy these members from within a GPU kernel, we need a 
  ! device version of the struct whos members are still the same ones in host memory
  type(element_type), device, allocatable :: d_element(:) ! <-- needs updating? should be pointer?

  ! Here is the copy of the structure that lives on the GPU
  type(GPUelement_type), device, allocatable :: d_GPUelement(:)
  ! And here is a CPU valid version for using it while on the host:
  type(GPUelement_type), pinned, allocatable :: GPUelement(:)

  ! can also do this with managed memory:
  type(GPUelement_type), managed, allocatable:: m_GPUelement(:)


contains
  !-----------------------------------------------------------------------
  subroutine construct_and_set_elements(element,id,Nnodes, volume, x, y, val) 

    ! initialize the number of nodes in an element,
    ! and the location and values of these nodes.

    ! variable declarations
    implicit none

    ! passed variables
    type(element_type), intent(inout) :: element(:)
    integer, intent(in)      :: id ! element index
    integer, intent(in)      :: Nnodes ! Number of nodes in this element
    real(kind=8), intent(in) :: volume 
    real(kind=8), intent(in) :: x(Nnodes), y(Nnodes)
    real(kind=8), intent(in) :: val(Nnodes)

    element(id)%Nnodes = Nnodes
    element(id)%volume = volume
    call nvtxStartRange("Host allocate")
    allocate(element(id)% x(Nnodes) )
    allocate(element(id)% y(Nnodes) )    
    allocate(element(id)% val(Nnodes) )
    call nvtxEndRange
    allocate(element(id)% old(Nnodes) )

    call nvtxStartRange("Host Populate")
    element(id)% x = x
    element(id)% y = y
    element(id)% val = val
    call nvtxEndRange

    return

  end subroutine construct_and_set_elements

  subroutine construct_and_set_GPUelements(element,id,Nnodes,Nelements, x, y, val) 

    ! initialize the number of nodes in an element,
    ! and the location and values of these nodes.

    ! variable declarations
    implicit none

    ! passed variables
    integer, intent(in) :: Nelements
    !type(GPUelement_type), intent(inout) :: element(:)  !<--- only changed type in this line
    type(GPUelement_type), intent(inout) :: element(:)  !<--- only changed type in this line
    integer, intent(in)      :: id ! element index
    integer, intent(in)      :: Nnodes ! Number of nodes in this element    
    real(kind=8), intent(in) :: x(Nnodes), y(Nnodes)
    real(kind=8), intent(in) :: val(Nnodes)

    
    ! allocate
    allocate(element(id)% x(Nnodes) )
    allocate(element(id)% y(Nnodes) )
    allocate(element(id)% val(Nnodes) )

    ! populate
    element(id)%Nnodes = Nnodes
    element(id)% x(:) = x(:)
    element(id)% y(:) = y(:)
    element(id)% val(:) = val(:)

    return

  end subroutine construct_and_set_GPUelements


  subroutine construct_GPUelements(element,id,Nnodes)

    ! Only allocate the nodes in an GPUelement

    ! variable declarations
    implicit none

    ! passed variables
    type(GPUelement_type), intent(inout) :: element(:)
    integer, intent(in)      :: id ! element index
    integer, intent(in)      :: Nnodes ! Number of nodes in this element

    allocate(element(id)% x(Nnodes) )
    allocate(element(id)% y(Nnodes) )
    allocate(element(id)% val(Nnodes) )

    return

  end subroutine construct_GPUelements


  attributes(global) subroutine set_elements_kernel(GPUelement,element, Nelements)
    implicit none
    ! kernel that uses zero copy to popluate the GPUelement structure.

    ! notice that I have dropped the d_ prefix for convenience on these dummy variables
    ! they are in fact device valid structures.
    type(GPUelement_type), device, intent(inout) :: GPUelement(Nelements) ! destination, members are device memory
    type(element_type), device, intent(in):: element(Nelements) ! use for zero copy, members are pinned host
    integer, value, intent(in) :: Nelements

    ! local variables
    integer :: id, Nnodes, node
       
    do id=blockIdx%x,Nelements, gridDim%x
       Nnodes = element(id)%Nnodes
       GPUelement(id)%Nnodes = Nnodes
       do node = threadIdx%x, Nnodes, blockDim%x
          GPUelement(id)%x(node) = element(id)%x(node)
          GPUelement(id)%y(node) = element(id)%y(node)
          GPUelement(id)%val(node) = element(id)%val(node)
       enddo
    enddo
    
  end subroutine set_elements_kernel

  subroutine check_correctness(GPUelement, element, Nelements)
    implicit none
    type(GPUelement_type), intent(in) :: GPUelement(Nelements)
    type(element_type), intent(in) :: element(Nelements)
    integer, intent(in) :: Nelements

    ! local arrays for holding device version to compare to host
    real(kind=8), allocatable :: x(:)
    real(kind=8), allocatable :: val(:)
    integer :: i, id, Nnodes

    do id = 1, Nelements

       Nnodes = element(id)%Nnodes

       allocate(x(Nnodes))

       !x = GPUelement(id)%x
       x(:) = GPUelement(id)%x(:)
       if(any(x .ne.  element(id)%x) ) then
          print *, "error in x, id = ", id
          write(*,"(A6,A6)") "GPU", "CPU"
          do i = 1, Nnodes
             write(*, "(f6.2,f6.2)") x(i), element(id)%x(i)
          enddo
       endif
       
       deallocate(x)

       allocate(val(Nnodes))

       !val = GPUelement(id)%val
       val(:) = GPUelement(id)%val(:)
       if(any(val .ne.  element(id)%val) ) then
          print *, "error in val, id = ", id
          write(*,"(A6,A6)") "GPU", "CPU"
          do i = 1, Nnodes
             write(*, "(f6.2,f6.2)") val(i), element(id)%val(i)
          enddo
       endif
       
       deallocate(val)

       
       !GPUelement(id)%y(node) = element(id)%y(node)
       !GPUelement(id)%val(node) = element(id)%val(node)
    enddo

  end subroutine check_correctness


  subroutine destruct_GPUelements(element,id)
    implicit none
    
    type(GPUelement_type), intent(inout) :: element(:)
    integer, intent(in) :: id
    
    deallocate(element(id)%x)
    deallocate(element(id)%y)
    deallocate(element(id)%val)

  end subroutine destruct_GPUelements

end module datastructures



program main
  use cudafor
  use datastructures
  implicit none

  integer :: threads, blocks
  integer(kind=cuda_stream_kind) old_stream, streamid
  integer :: istat

  integer :: Nelements, Nnodes, NnodesMax, id, i
  real(kind=8) :: volume
  real(kind=8), allocatable :: x(:),y(:)
  real(kind=8), allocatable  :: val(:)
  character(len=255) :: char_elements, char_nodes

  call get_command_argument(1,char_elements)
  if(len_trim(char_elements)==0) then
     write(0,*) "missing command line arg, ./a.out Nelements Nnodes"
     char_elements="10000"
  endif

  call get_command_argument(2,char_nodes)
  if(len_trim(char_nodes)==0) then
     write(0,*) "missing command line arg, ./a.out Nelements Nnodes"
     char_nodes="8"
  endif

  ! string to integer conversion
  read(char_elements,"(I16)") Nelements
  read(char_nodes,"(I16)") NnodesMax

  print *, "running example with ", Nelements, "elements and ", NnodesMax, "nodes"

  ! regular host initialization of elements
  allocate( element(Nelements) )

  
  ! **Manually maintain host/device structures:
  ! allocate a "skinny" structure with device components (will be reused below)
  allocate( GPUelement(Nelements) )
  ! allocate device structure that will point to the same device components as above
  allocate( d_GPUelement(Nelements) )
  ! **


  ! *Managed Memory Style:
  allocate( m_GPUelement(Nelements) )
  ! *


  ! example arrays
  allocate( x(NnodesMax), y(NnodesMax), val(NnodesMax) )
  ! In practice this data would be meaningful and different for each element,
  ! so here we fill values unique to an element and a node:

  do id=1,Nelements
     do i=1,NnodesMax
        x(i)   = id+i
        y(i)   = id+i
        val(i) = id+i
     enddo
     volume = 5
     Nnodes = NnodesMax
     ! In practice Nnodes in each element could be different
     call nvtxStartRange("const_and_set_host_elements",8)
     call construct_and_set_elements(element,id,Nnodes, volume, x, y, val)
     call nvtxEndRange
  enddo
  



  ! STYLE 1: allocate and set in same host routine:
  ! One option that is sometimes possible is allocate and construct at the same
  ! time. This allows allocation and setting to be qued up on the device,
  ! but it is still not very fast.

  call nvtxStartRange("Style 1",1)
  do id=1,Nelements
     do i=1,NnodesMax
        x(i)   = id+i
        y(i)   = id+i
        val(i) = id+i
     enddo
     volume = 5
     Nnodes = NnodesMax

     call construct_and_set_GPUelements(GPUelement,id,Nnodes,Nelements, x, y, val)
     
  enddo

  istat = cudaDeviceSynchronize()
     
  call nvtxEndRange


  call check_correctness(GPUelement, element, Nelements)


  do id=1,Nelements
     call destruct_GPUelements(GPUelement,id)
  enddo

  print*, "completed style 1"

  !************************************************************************

  ! Usually allocation happens in one place in the code and later the values are populated.

  ! allocate GPUelement%members which would usually take place apart from setting data.
  call nvtxStartRange("allocate GPUelement%members")
  do id=1, Nelements
     call construct_GPUelements(GPUelement,id,Nnodes)
  enddo
  call nvtxEndRange

  
  ! STYLE 2: Naively set the values from the host version

  call nvtxStartRange("Style 2",2)
  ! set the values from the host version of the data structure:
  do id=1, Nelements
     GPUelement(id)%Nnodes = element(id)%Nnodes !host to host copy
     GPUelement(id)%x = element(id)%x           !implicit HtoD
     GPUelement(id)%y = element(id)%y
     GPUelement(id)%val = element(id)%val
  enddo

  istat = cudaDeviceSynchronize()

  call nvtxEndRange

  call check_correctness(GPUelement, element, Nelements)

  print*, "completed style 2"


  ! STYLE 3
  ! Improve style 2 by queing up async movements

  ! pin element as it was a pageable pointer before
  istat = cudaHostRegister(C_LOC(element), sizeof(element), cudaHostRegisterMapped)

  call nvtxStartRange("Style 3",3)

  ! use async memcopy command to que up data transfers. 
  ! using default stream makes GPU movement request concurrent with
  ! continuing CPU code, but GPU requests are still serial
  ! with respect to eachother. 
  streamid = cudaforGetDefaultStream()
  !!!!$omp parallel do private(id) schedule(static,1)
  do id=1, Nelements
     GPUelement(id)%Nnodes = element(id)%Nnodes ! cpu to cpu copy
     istat=cudaMemcpyAsync(GPUelement(id)%x, element(id)%x, size(element(id)%x), streamid)
     istat=cudaMemcpyAsync(GPUelement(id)%y, element(id)%y, size(element(id)%y), streamid)
     istat=cudaMemcpyAsync(GPUelement(id)%val, element(id)%val, size(element(id)%val), streamid)
  enddo

  istat = cudaDeviceSynchronize()

  call nvtxEndRange

  call check_correctness(GPUelement, element, Nelements)

  print*, "completed style 3"





  ! this part is really a separate topic

  call nvtxStartRange("MemcpyAsync d_GPUelement")
  ! Really want something that is accessible on the device, so need to move base structure to device:
  istat=cudaMemcpyAsync(d_GPUelement, GPUelement, size(GPUelement), 0)
  call nvtxEndRange

  ! If using Unified Memory can prefetch to the device
  istat=cudaMemPrefetchAsync(m_GPUelement,size(m_GPUelement),device=0, stream=0)

  istat = cudaDeviceSynchronize()


  

  ! STYLE 4
  ! Can also do the above in different streams--any difference?
  call nvtxStartRange("Style 4", 4)

  old_stream = cudaforGetDefaultStream()       ! save the current default stream
  !!!!!$omp parallel private(streamid, id)
  istat = cudaStreamCreate(streamid)
  istat = cudaforSetDefaultStream(streamid)   ! Set the default stream to streamid
  !!!!$omp do schedule(static,1)
  do id=1, Nelements
     GPUelement(id)%Nnodes = element(id)%Nnodes ! cpu to cpu copy
     GPUelement(id)%x = element(id)%x           ! implicit cudaMemcpyAsync on streamid
     GPUelement(id)%y = element(id)%y           ! implicit cudaMemcpyAsync on streamid
     GPUelement(id)%val = element(id)%val       ! implicit cudaMemcpyAsync on streamid
  enddo
  !!!!$omp end do
  !!!!$omp end parallel
  istat = cudaforSetDefaultStream(old_stream) ! restore the original default stream
  istat = cudaDeviceSynchronize()
  call nvtxEndRange


  print*, "completed style 4"


  ! STYLE 5
  ! Create a CUDA kernel that uses zero copy to pull the host data
  ! into the device structure.

  call nvtxStartRange("Style 5",5)

  ! Now we need to be able to refer to the data structure from within a device kernel,
  ! so we need the following device valid structures:
  

  call nvtxStartRange("Cost to setup zero copy of element")
  ! d_element -- a device valid variable which points to the pinned host memory locations of element
  allocate( d_element(Nelements) )

  ! to use element on the device, we have to make a copy in the device valid d_element:
  #if defined(__ibmxl__)
     istat=cudaMemcpyAsync(d_element, element, size(element), 0)
  #else
     istat=cudaMemcpyAsync(C_DEVLOC(d_element), C_LOC(element), sizeof(element), 0)
  #endif
  istat = cudaDeviceSynchronize()
  call nvtxEndRange


  ! Now we can use these in a CUDA kernel to zero copy the member data from d_element into d_GPUelement
  
  threads = min(NnodesMax,1024)
  blocks = Nelements

  call set_elements_kernel<<<blocks,threads>>>(d_GPUelement,d_element,Nelements)

  istat = cudaDeviceSynchronize()

  call nvtxEndRange


  call check_correctness(GPUelement, element, Nelements)

  do id=1,Nelements
     call destruct_GPUelements(GPUelement,id)
  enddo
  
  istat = cudaDeviceSynchronize()


  print*, "completed style 5"



end program main
