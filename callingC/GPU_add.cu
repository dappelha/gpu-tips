#include <stdio.h>


extern "C"
{

  __global__ void GPU_add(
			    int  n,
			    int* d_a,
			    int* d_b
			  )
  {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
	 i < n; 
	 i += blockDim.x * gridDim.x) 
    {
      d_a[i] += d_b[i];
    }
  }

}

