/*
 * Parallel bitonic sort using CUDA.
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 4
#define BLOCKS 4096
#define NUM_VALS THREADS*BLOCKS

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

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* inline exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* inline exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  // Flat memory layout:
  // [T,T,T,T...,T] [T,T,T,T...,T] [T,T,T,T...,T] [T,T,T,T...,T]
  // \___________/  \___________/  \___________/  \___________/
  //    Block 1        Block 2        Block 3        Block 4

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */

  int j, k;
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      //printf("Major step: k=%d, Minor step: j=%d\n", k, j);
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);
}


double get_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  return elapsed;
}

int read_elements() {
  printf("Number of elements (must be a power of 2)? ");
  int elem;
  scanf("%d", &elem);
  return elem;
}

int nextPow2(int n)
{
    if ( n <= 1 ) return n;
    double d = n-1;
    return 1 << ((((int*)&d)[1]>>20)-1022);
}

int main(void)
{
  clock_t start, stop;
  start = clock();

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);

  bitonic_sort(values);
  stop = clock();

  double elapsed = get_elapsed(start, stop);
  array_print(values, NUM_VALS);
  //printf("Elements: %d (%lu MB)\t\tElapsed: %.6fs\n", NUM_VALS, (NUM_VALS * sizeof(float))/(1024*1024),elapsed);

  int padding = nextPow2(NUM_VALS) - NUM_VALS;
  //printf("Elements: %d\n", NUM_VALS);
  //printf("Padding: %d\n", padding);
}
