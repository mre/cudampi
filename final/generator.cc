#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "config.h"
#include "randgen.h"

/**
 * Generate a large amount of random numbers and write them into an MPI binary file.
 * Since I/O is much slower than random number generation on the processor, only one
 * processor is needed to write the file (more processors would just mean more overhead).
 */
int generate(size_t num_vals, char *filename, int rank) {
    MPI_File data;

    // Remove old file if it already exists.
    MPI_File_delete(filename, MPI_INFO_NULL);

    int rc = MPI_File_open(MPI_COMM_WORLD, filename, (MPI_MODE_WRONLY | MPI_MODE_CREATE), MPI_INFO_NULL, &data);
    if (rc != MPI_SUCCESS) {
      return -1;
    }


    if (rank == 0) {

      // Since there's only one processor involved, we don't need to set a special view on the file.
      // Anyway, the syntax would be as follows.
      /*
      MPI_File_set_view (data, rank*sizeof(MPI_FLOAT), MPI_FLOAT, MPI_Datatype filetype,
                         char *datarep, MPI_Info info);
      */

      float *buf = (float *) malloc(BUFSIZE * sizeof(float)); /* Buffer for random number chunks */
      srand(time(0));

      double bursts = num_vals / BUFSIZE; /* Number of writes from buffer to file */

      DEBUG_PRINT("Writing %lu MB of generated data in %.0f steps with blocksize %lu MB...\n",
          (num_vals*sizeof(float) >> 20), bursts, (BUFSIZE*sizeof(float) >> 20));

      int i,j;
      for (i = 0; i < bursts; ++i ) {
        for (j = 0; j < BUFSIZE; ++j) {
          // Method 1 - Elaborate, uniform random number generator
          buf[j] = randrange(MIN_NUM, MAX_NUM);

          // Method 2 - Simle random number generator
          // int x = random();
          // buf[j] = (float) x / RAND_MAX - 0.000001;
        }
        MPI_Status status;
        MPI_File_write(data, buf, BUFSIZE, MPI_FLOAT, &status);
      }
    }
    // Writing the numbers is done. Close the file.
    MPI_File_close(&data);

    if (rank == 0) {
      DEBUG_PRINT("Generate done.\n");
    }

    return 0; // No errors
}
