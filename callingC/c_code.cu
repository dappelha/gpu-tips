#include<stdio.h>
#include<iostream>


extern "C" {
  __global__ void GPU_add(
			    int  n,
			    int* d_a,
			    int* d_b
			  );

  void calling_routine_c (
			  int  n,
			  int* d_a,
			  int* d_b
			  ) 
  {
    

    //printf("cuda c stream = %lld\n",streamid);

    
    // Call the cuda kernel:
    GPU_add<<<1,1024>>>(
			n,
			d_a,
			d_b
			);





    printf("Completed an add kernel\n");

  } // end calling routine

} // extern "C"
