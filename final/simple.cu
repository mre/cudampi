/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc -arch=sm_11 bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define THREADS 512 // 2^9
#define BLOCKS 32768 // 2^15
#define BUCKET_NUM_VALS THREADS*BLOCKS

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length)
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

__global__ void bitonic_sort_step(float *dev_values, int j, int k, int n)
{
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values, size_t num_values)
{
  float *dev_values;
  size_t size = num_values * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  //number of threads should be equals of num_values, 1/2 * num_values, 1/4 *num_values, etc
  dim3 blocks(BLOCKS/4,1);    /* Number of blocks   */
  dim3 threads(THREADS/2,1);  /* Number of threads  */

  bitonic_sort_step<<<blocks, threads>>>(dev_values, 0, 0,num_values);

  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);
}

int kernel_test(void)
{
  clock_t start, stop;

  int i = 64;

  size_t data_size_mb = (i * sizeof(float)) >> 20;
  printf("Sorting %ld MB of data\n", data_size_mb);

  size_t available, total;
  cudaMemGetInfo(&available, &total);
  printf("Memory available: %ld, Total: %ld\n", available, total);

  float *values = (float*) malloc(i * sizeof(float));
  array_fill(values, i);

  start = clock();
  bitonic_sort(values, i); /* Inplace */
  stop = clock();

  // Show a few values to check if everything is ok
  array_print(values, 200);
  print_elapsed(start, stop);
  return 0;
}

