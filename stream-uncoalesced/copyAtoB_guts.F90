    integer :: n
    real(kind=8),intent(inout) :: A(:,:), B(:,:)

    integer :: i,j, blocks, threads, tot_threads
    real(kind=8) :: t1, t2, T, mem


    tot_threads=80*1024 ! this is as many blocks*threads as we need to examine.

    !do threads = 32, 512, 512-32
    !do blocks = 0, tot_threads/threads, tot_threads/threads/160
          threads = 32
          blocks = 320
    !      if (blocks .eq. 0) cycle
          t1 = omp_get_wtime()
          
          !$acc parallel loop gang vector present(A,B)
          do j = 1, n
             do i = 1, stride ! This loop will be unrolled if stride is compile time constant.
                B(i,j) = A(i,j)
             enddo
          enddo
          !$acc end parallel
          t2 = omp_get_wtime()
          T = t2-t1

          !mem = 8*real(n,kind=8)/real(1024*1024*1024,kind=8)
          mem = 8*stride*real(n,kind=8)/real(1000*1000*1000,kind=8)



          !write(22,fmt) n*ilp, blocks, threads, blocks*threads, ilp, 2*mem/T
          write(22,fmt) stride, 2*mem/T


          
