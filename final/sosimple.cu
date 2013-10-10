#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

__global__ void kernel(void) {
}

int kernel_test()
{
  size_t available, total;
  cudaMemGetInfo(&available, &total);
  printf("Memory available: %ld, Total: %ld\n", available, total);

  float *dev_values;
  size_t size = num_values * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  kernel<<<1,1>>>();
  printf("finished \n");
  return 0;
}

