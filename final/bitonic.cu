/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc -arch=sm_11 bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "config.h"

#define THREADS 512 // 2^9
#define BLOCKS 32768 // 2^15
#define GPU_NUM_VALS THREADS*BLOCKS

/**
 * Show elapsed time between two points in time
 */
void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

/**
 * Built-in random number generator for testing purposes only
 */
float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

/**
 * Write <length> elements of an array to stdout.
 */
void array_print(float *arr, int length)
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

/**
 * Fill an array with random floats.
 * For testing purposes only.
 */
void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}


/**
 * Round number up to the next given multiple
 */
int roundUp(int numToRound, int multiple) {
 if(multiple == 0) {
  return numToRound;
 }
 int remainder = numToRound % multiple;
 return numToRound + multiple - remainder;
}

/**
 * Perform one step of bitonic sort
 */
__global__ void bitonic_sort_step(float *dev_values, int j, int k, int n)
{
  unsigned int i, ixj, stride; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  stride = blockDim.x * gridDim.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) { // avoid unnecessary threads -> in each iteration half iterations does nothing
    while(i < n){
      if ((ixj)>i) {
        if ((i&k)==0) {
          /* Sort ascending */
          if (dev_values[i]>dev_values[ixj]) {
            /* exchange(i,ixj); */
            float temp = dev_values[i];
            dev_values[i] = dev_values[ixj];
            dev_values[ixj] = temp;
          }
        }
        if ((i&k)!=0) {
          /* Sort descending */
          if (dev_values[i]<dev_values[ixj]) {
            /* exchange(i,ixj); */
            float temp = dev_values[i];
            dev_values[i] = dev_values[ixj];
            dev_values[ixj] = temp;
          }
        }
      }
      i = i + stride;
      ixj = i ^ j;
    }
  }
}

/**
 * Sort <values> with GPU
 */
float *sort_gpu_bitonic(float *values, size_t num_values, int rank)
{
  DEBUG_PRINT("Processor %d: Bitonic sort on GPU\n", rank);

  clock_t start, stop;

  size_t available, total;
  cudaMemGetInfo(&available, &total);
  DEBUG_PRINT("Processor %d: Memory available: %ld, Total: %ld\n", rank, available, total);

  float *dev_values;
  size_t size = num_values * sizeof(float);

  start = clock();
  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  // Number of threads should be equal to num_values, 1/2 * num_values, 1/4 *num_values, etc
  dim3 blocks(BLOCKS/4,1);    // Number of blocks
  dim3 threads(THREADS/2,1);  // Number of threads

  size_t j, k;
  // Major step
  for (k = 2; k <= num_values; k <<= 1) {
    // Minor step
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k,num_values);
    }
  }
  // Send result to host
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  // Free space on GPU device
  cudaFree(dev_values);
  stop = clock();

  // Show a few values to check if everything is ok
  // array_print(values, 200);
  print_elapsed(start, stop);
  return values;
}
