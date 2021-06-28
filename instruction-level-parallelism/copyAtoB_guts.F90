    integer :: n
    real(kind=8),intent(inout) :: A(n,*), B(n,*)

    integer :: i,j, blocks, threads, tot_threads
    real(kind=8) :: t1, t2, T, mem


    tot_threads=80*1024 ! this is as many blocks*threads as we need to examine.
    
    threads = 32
    ! do blocks = 0, tot_threads/threads, tot_threads/threads/160
    !    if (blocks .eq. 0) cycle
    blocks = 320
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

    !enddo
