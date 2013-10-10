#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "bucket.h"
#include "config.h"

/**
 * Get number of floating point values inside of a bucket file
 */
size_t bucket_file_num_floats(char *filename, int EXCLUSIVE_ACCESS_FLAG) {
    MPI_File fh;

    if (EXCLUSIVE_ACCESS_FLAG == MPI_IO_EXCLUSIVE_ACCESS) {
      MPI_File_open(MPI_COMM_SELF, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    } else {
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    }

    // Get the size of the file
    MPI_Offset filesize;
    MPI_File_get_size(fh, &filesize);
    MPI_File_close(&fh);
    // Calculate how many elements that is
    return filesize/sizeof(float);
}

/**
 * Read floating point numbers from file.
 * filename   Name of file on filesystem
 * start      Starting postion to read (in bytes)
 * nfloats    Number of floats to read from file
 *            If nfloats <= 0, read entire file.
 */
float *read_floats(int rank, char *filename, int start, int nfloats, int EXCLUSIVE_ACCESS_FLAG) {
    MPI_File fh;
    float *buf;

    if (EXCLUSIVE_ACCESS_FLAG == MPI_IO_EXCLUSIVE_ACCESS) {
      // Open file exclusively
      MPI_File_open(MPI_COMM_SELF, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    } else {
      // Share access to the file.
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    }

    if (nfloats <= 0) {
      // Read entire file.
      MPI_Offset filesize;
      // Get the size of the file
      MPI_File_get_size(fh, &filesize);
      // Calculate how many elements that is
      nfloats = filesize/sizeof(float);
      // Allocate buffer
      buf = (float*) malloc(nfloats * sizeof(float));
    } else {
      // Read only parts of the file.
      // Allocate buffer
      buf = (float *) malloc(nfloats * sizeof(float));
      MPI_File_seek(fh, start, MPI_SEEK_SET);
    }
    // DEBUG_PRINT("Processor %d: Reading %d floats, %ld Bytes\n", rank, nfloats, nfloats * sizeof(float));

    MPI_Status status;
    MPI_File_read(fh, buf, nfloats, MPI_FLOAT, &status);
    MPI_File_close(&fh);
    return buf;
}

/**
 * As an intial sorting step, each processor generates a number of buckets (which are files) which
 * contain numbers of a given interval.
 * These files have the following naming scheme:
 * b_p where
 *  b is the number of the current bucket (ranging from 0 to num_buckets)
 *  p is the processor number
 * The bucket numbers are zero padded (e.g. 0020 when there are 1000 buckets in total).
 */
char *get_bucket_filename(int curr_bucket, int num_buckets, int rank, int size) {
    int len_buckets = floor(log10(abs(num_buckets)));
    int len_size = floor(log10(abs(size))) + 1;
    int len_bucket_dir = strlen(BUCKET_DIR);

    char *bucket_file = (char*) malloc((len_bucket_dir + len_buckets + len_size + 2)*sizeof(char));

    // The first part of the filename is the directory for the bucket files
    strcpy(bucket_file, BUCKET_DIR);
    strcat(bucket_file, "/"); // Directory separator

    // The second part of the filename is the bucket number which gets
    // padded with zeros.
    char *bucket_name = (char*) malloc((len_buckets + 1)*sizeof(char));
    sprintf(bucket_name, "%0*d", len_buckets, curr_bucket);
    strcat(bucket_name, "_"); // Separator
    strcat(bucket_file, bucket_name);

    // The third part of the filename is the processor rank (not
    // padded)
    char str_rank[len_size];
    sprintf(str_rank,"%d",rank);
    strcat(bucket_file, str_rank);  // Processor rank
    return bucket_file;
}

/**
 * Open <bucket_file> for a single processor and return a handle
 */
MPI_File open_handle(char *bucket_file) {
  MPI_File_delete(bucket_file, MPI_INFO_NULL);
  MPI_File handle;
  // A process can open a file independently of other processes by using the MPI_COMM_SELF communicator.
  int status_code = MPI_File_open(MPI_COMM_SELF, bucket_file, (MPI_MODE_WRONLY | MPI_MODE_CREATE), MPI_INFO_NULL, &handle);

  // Error handling
  if (status_code != MPI_SUCCESS) {
    int resultlen = 0;
    char status_string[MPI_MAX_ERROR_STRING];
    MPI_Error_string(status_code, status_string, &resultlen);
    perror(status_string);
    exit(EXIT_FAILURE);
  }

  return handle;
}

/**
 * Open <num_buckets> files that serve as buckets for a specific processor
 */
MPI_File *open_handles(int num_buckets, int rank, int size) {
    // Since we are opening a lot of files, we create an array to keep
    // all the file handles.
    MPI_File *handles = (MPI_File *) malloc(num_buckets*sizeof(MPI_File));

    int curr_bucket;
    for (curr_bucket = 0; curr_bucket < num_buckets; ++curr_bucket) {
      char *bucket_file = get_bucket_filename(curr_bucket, num_buckets, rank, size);
      DEBUG_PRINT("Processor %d: Opening file %s\n", rank, bucket_file);
      handles[curr_bucket] = open_handle(bucket_file);
      free(bucket_file);
    }
    return handles; // Needed to close the files later on.
}

/**
 * Close open files that serve as buckets for a specific processor
 */
void close_handles(MPI_File *handles, int num_buckets) {
    int curr_bucket;
    for (curr_bucket = 0; curr_bucket < num_buckets; ++curr_bucket) {
      DEBUG_PRINT("Closing file for bucket %d\n", curr_bucket);
      MPI_File_close(&handles[curr_bucket]);
    }
}

/**
 * Add an element to the buffer
 */
int buf_add(bucket_buffer* b, float val) {
  if (b->num_elements < BUFSIZE) {
      b->buf[b->num_elements] = val;
      b->num_elements = b->num_elements + 1;
      return 1;
  } else {
      return 0;
  }
}

/**
 * Write current content of buffer to file.
 */
int buf_flush(bucket_buffer *b, MPI_File *handle) {
  MPI_Status status;
  int status_code = MPI_File_write(*handle, b, b->num_elements, MPI_FLOAT, &status);

  // Error handling
  if (status_code != MPI_SUCCESS) {
    int resultlen = 0;
    char status_string[MPI_MAX_ERROR_STRING];
    MPI_Error_string(status_code, status_string, &resultlen);
    perror(status_string);
    exit(EXIT_FAILURE);
  }

  DEBUG_PRINT("Swapping to disk. MPI status: %s\n", status_string);
  b->num_elements = 0;
  return status_code;
}

/**
 * Write an element to the buffer.
 * If the buffer is full, write to disk and
 * clear buffer.
 */
int buf_write(bucket_buffer* b, MPI_File *handle, float val) {
  if (! buf_add(b, val)) {
    // Buffer full. Write to disk and clear buffer.
    int status_code = buf_flush(b, handle);
    // Write value into empty buffer
    buf_add(b, val);
    return status_code;
  }
  return 1;
}

/**
 * Write buckets to disk.
 * Each processor creates its own buckets which will later be
 * merged by the master processor.
 * Assume that each processor puts about the same amount of
 * data into each bucket.
 */
void write_buckets(size_t num_vals, float *buf, int bufsize, int rank, int size) {
    // How many bucket files must each processor create in order to
    // fill them with a maximum of MAX_BUCKETSIZE_BYTES.
    size_t data_size = num_vals * sizeof(float);
    size_t num_buckets = ceil(data_size / (double) MAX_BUCKETSIZE_BYTES);
    double interval = MAX_NUM - MIN_NUM;

    if (rank == 0) {
      DEBUG_PRINT("Number of buckets: %ld\n", num_buckets);
      DEBUG_PRINT("Number interval: %d to %d\n", MIN_NUM, MAX_NUM);
    }

    // Create buffers
    bucket_buffer *buffers = (bucket_buffer *) malloc(num_buckets * sizeof(bucket_buffer));

    // Create handles
    MPI_File *handles = open_handles(num_buckets, rank, size);

    int i;
    for (i = 0; i < bufsize; ++i) {
      // Determine the correct bucket for the current value
      int bucket = floor((buf[i]/interval) * num_buckets);
      // DEBUG_PRINT("i: %d buf[i]: %f bucket: %d\n", i, buf[i], bucket);
      // Optimize write performance by using a write-buffer
      int status_code = buf_write(&buffers[bucket], &handles[bucket], buf[i]);
      if (status_code) {
          DEBUG_PRINT("Processor %d: Error while writing to bucket %d\n", rank, bucket);
      }

      // Alternatively, write each float on its own.
      // (Slow! For debugging purposes only)
      // MPI_Status status;
      // MPI_File_write(handles[bucket], &buf[i], 1, MPI_FLOAT, &status);
    }

    // Flush remaining values of all buffers to disk
    for (i = 0; i < num_buckets; ++i) {
        buf_flush(&buffers[i], &handles[i]);
    }

    close_handles(handles, num_buckets);
    free(handles);
    free(buffers);
}

/**
 * Load the sorting data from a file and distribute it
 * evenly amongst all slave processes.
 *
 * filename Unsorted values are stored in this file
 * rank     MPI processor rank
 * size     Number of MPI processors in total
 */
void bucket(size_t num_vals, char *filename, int rank, int size) {
  // Wait for other processors
  MPI_Barrier(MPI_COMM_WORLD);

  int nums_per_processor = num_vals / size;
  int start = rank * nums_per_processor * sizeof(float);

  float *buf;
  buf = read_floats(rank, filename, start, nums_per_processor, MPI_IO_SHARED_ACCESS);
  DEBUG_PRINT("Processor %d read %d floats\n", rank, nums_per_processor);
  DEBUG_PRINT("Processor %d: %f %f %f\n", rank, buf[0], buf[1], buf[2]);
  DEBUG_PRINT("Processor %d: Creating sort buckets...\n", rank);
  write_buckets(num_vals, buf, nums_per_processor, rank, size);
  free(buf);
  return;
}
