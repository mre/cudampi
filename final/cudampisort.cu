/*
 * Parallel sorting using CUDA and MPI.
 * Note that this program is only useful for big data
 * (Gigabytes of floating point numbers).
 *
 * Compile with
 * nvcc -lmpi cudampisort.cu sort.cu # on Mac
 * nvcc -I$CUDA_SDK/shared/inc -I$MPI_HOME/include -L$MPI_HOME/lib64/ -lmpi -arch sm_13 sort.cu cudampisort.cu
 *
 * Copyright (C) 2013 Matthias Endler
 *
 */

#include <stdio.h>
#include <mpi.h>
#include "config.h"
#include "generator.h"
#include "bucket.h"
#include "sort.h"

int main(int argc, char *argv[])
{
  int rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
      printf("Error starting MPI. Aborting\n");
      MPI_Abort(MPI_COMM_WORLD, rc);
  }


  size_t num_vals = NUM_VALS;
  if (argc > 1) {
    num_vals = atoi(argv[1]);
  }

  int rank, size; /* Rank of each processor and number of processors in total */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int len;
  char name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(name, &len);

  DEBUG_PRINT("Processor %d: Starting on %s. Poolsize: %d\n", rank, name, size);

  //generate(num_vals, UNSORTED_FILENAME, rank);
  bucket(num_vals, UNSORTED_FILENAME, rank, size);
  sort(rank, size);
  MPI_Finalize();
  return 0;
}
