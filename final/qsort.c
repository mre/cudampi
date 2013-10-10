/*
 * Standard quicksort implementation for CPU
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "config.h"

double get_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  return elapsed;
}

int floatcomp(const void* elem1, const void* elem2)
{
    if(*(const float*)elem1 < *(const float*)elem2)
        return -1;
    return *(const float*)elem1 > *(const float*)elem2;
}

float *sort_cpu_qsort(float *buf, size_t num_values, int rank)
{
  DEBUG_PRINT("Processor %d: Quicksort on CPU\n", rank);

  clock_t start, stop;

  start = clock();
  qsort (buf, num_values, sizeof(float), floatcomp);
  stop = clock();

  double elapsed = get_elapsed(start, stop);
  printf("Elements: %ld\t\tElapsed: %.3fs\n", num_values, elapsed);
  return buf;
}
