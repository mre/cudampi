#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <mpi.h>

#include "bitonic.cu"
#include "qsort.h"
#include "bucket.h"
#include "config.h"

/**
 * Get the id from a bucket file of processor 0.
 * (filename: bucketid + FILENAME_SEPARATOR + pid)
 */
char *get_bucket_id(char *file) {

  // Check if filename begins with a bucket id
  char *non_numeric_chars_pos;
  int bucket_id = strtol(file, &non_numeric_chars_pos, 10);

  // Find filename separator
  char *found;
  found = strchr(file, FILENAME_SEPARATOR);

  if (found == file || found != non_numeric_chars_pos) {
    // FILENAME_SEPARATOR at beginning of filename or
    // FILENAME_SEPARATOR does not immediately
    // follow after bucket id. Something's fishy.
    return NULL;
  }

  // Get length of bucket id substring in filename
  size_t len_bucket_id = found - file;
  size_t len_separator = 1;

  // Is there a processor id behind the filename separator?
  if (strlen(file) <= len_bucket_id + len_separator) {
      return NULL; // No processor id. Error.
  }

  // Read processor id
  int pid = file[len_bucket_id + 1] - '0';
  if (pid != 0) return NULL; // Not a bucket file of Processor 0

  // Found a valid file. Return bucket id
  char *bucket = (char *) malloc(len_bucket_id);
  strncpy(bucket, file, len_bucket_id);
  bucket[len_bucket_id] = '\0';

  return bucket;
}

/**
 * Get the next free bucket which needs to be sorted
 */
char* next_bucket_id() {
  DIR* myDirectory;
  char* fullpath;
  char path[MAXPATHLEN];

  if (getcwd(path, sizeof(path)) == NULL) {
    exit(-1);
  }

  strcat(path,"/");
  strcat(path,BUCKET_DIR);
  strcat(path,"/");
  myDirectory=opendir(path);

  struct stat fileProperties;
  struct dirent* directory;

  do {
    directory=readdir(myDirectory);
    if (directory != NULL) {

      fullpath = (char*) malloc((strlen(path)+strlen(directory->d_name)+2 ) * sizeof(char));
      strcpy(fullpath,path);
      strcat(fullpath,directory->d_name);

      lstat(fullpath,&fileProperties);

      if ((fileProperties.st_mode & S_IFMT) == S_IFREG) {
        // File found. Get bucket id from filename
        char *bucket_id = get_bucket_id(directory->d_name);
        if (bucket_id) {
          closedir(myDirectory);
          free(fullpath);
          return bucket_id;
        }
      }
    }
  } while(directory!=NULL);
  return NULL; // No remaining buckets found
}

/**
 * Given a bucket name (i.e. 0_0), return the full path to the bucket
 * file
 */
char *get_bucket_path(char *dir, char *bucket_name, int rank, int size) {
    size_t dir_len = strlen(dir);
    int len_size = floor(log10(abs(size))) + 1;

    // Calculate space for bucket filename
    // Extra padding for:
    //  - Path separator
    //  - Filename separator
    //  - Terminating \0 character
    //  Example:              buckets/1233_0\0
    int len_path = dir_len +  strlen(bucket_name) + len_size + 3;

    char *bucket_path = (char*) malloc(len_path);
    strcpy(bucket_path, dir);
    strcat(bucket_path, "/");
    strcat(bucket_path, bucket_name);
    bucket_path[dir_len + strlen(bucket_name) + 1] = FILENAME_SEPARATOR;
    bucket_path[dir_len + strlen(bucket_name) + 2] = '\0';

    char str_rank[len_size];
    sprintf(str_rank,"%d",rank);
    strcat(bucket_path, str_rank);
    return bucket_path;
}

/**
 * Signal other processors that we are working on a bucket
 * by moving the corresponding files into a temporary folder
 */
void enqueue_bucket(char *bucket_name, int rank, int size) {
  // Move all files of all processes belonging to one bucket into the working dir
  int current_rank;
  for (current_rank = 0; current_rank < size; ++current_rank) {

    char* old_bucket_path = get_bucket_path(BUCKET_DIR, bucket_name, current_rank, size);
    char* new_bucket_path = get_bucket_path(WORKING_DIR, bucket_name, current_rank, size);
    DEBUG_PRINT("Processor %d: %s -> %s\n", rank, old_bucket_path, new_bucket_path);
    rename(old_bucket_path, new_bucket_path);

    free(old_bucket_path);
    free(new_bucket_path);
  }
}

/**
 * Find the next power of two for a given number
 * Explanation: http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
 */
size_t next_pow2(size_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

float* hybrid_sort(float *buf, size_t kernel_values, int rank) {
  if (rank % 2 == 0) {
    return sort_gpu_bitonic(buf, kernel_values, rank);
  } else {
    return sort_cpu_qsort(buf, kernel_values, rank);
  }
}

/**
 * Prepare data and call sorting function
 */
void sort_bucket(char *bucket_name, int rank, int size) {
  DEBUG_PRINT("Processor %d: Read bucket values of each processor for bucket %s\n", rank, bucket_name);

  // Allocate a sorting kernel buffer large enough to keep all float values for all
  // files of the current bucket. Use MAX_BUCKETSIZE_MB as an upper limit for
  // the total buffer size.

  size_t kernel_values = next_pow2(MAX_BUCKETSIZE_BYTES / sizeof(float));
  DEBUG_PRINT("Processor %d: Max kernel values: %ld\n", rank, kernel_values);
  float *buf = (float *) malloc(kernel_values * sizeof(float));

  // Count total number of floats that need to be sent to the sorting
  // kernel
  size_t total_num_values = 0;
  int current_rank;
  for (current_rank = 0; current_rank < size; ++current_rank) {
    char* new_bucket_path = get_bucket_path(WORKING_DIR, bucket_name, current_rank, size);
    // Read entire file
    DEBUG_PRINT("Processor %d: Reading bucket file of processor %d\n", rank, current_rank);
    float *tmp_buf = read_floats(rank, new_bucket_path, 0, 0, MPI_IO_EXCLUSIVE_ACCESS);
    size_t num_floats = bucket_file_num_floats(new_bucket_path, MPI_IO_EXCLUSIVE_ACCESS);
    DEBUG_PRINT("Processor %d: Read %ld\n", rank, num_floats);
    memcpy(&buf[total_num_values], tmp_buf, num_floats*sizeof(float));
    total_num_values += num_floats;
    free(tmp_buf);
  }

  // Bitonic sort only works if the number of values is a power of two.
  // Check if padding is needed.
  int i;
  for (i = total_num_values; i < kernel_values; ++i) {
    buf[i] = (float) MIN_NUM - 1.0;
  }

  // Even processors use the GPU to sort,
  // the uneven ones use the CPU.

  if (SORT_WITH_GPU_ONLY) {
    buf = sort_gpu_bitonic(buf, kernel_values, rank);
  } else if (SORT_WITH_CPU_ONLY) {
    buf = sort_cpu_qsort(buf, kernel_values, rank);
  } else {
    buf = hybrid_sort(buf, kernel_values, rank);
  }


  // Write result
  char* final_bucket_path = get_bucket_path(FINAL_DIR, bucket_name, 0, 1);
  DEBUG_PRINT("Processor %d: Writing file %s\n", rank, final_bucket_path);
  MPI_File handle = open_handle(final_bucket_path);
  MPI_Status status;
  MPI_File_write(handle, buf, total_num_values, MPI_FLOAT, &status);
  MPI_File_close(&handle);

  // Cleanup
  free(buf);
  // Delete buckets
  for (current_rank = 0; current_rank < size; ++current_rank) {
    char* remove_bucket_path = get_bucket_path(WORKING_DIR, bucket_name, current_rank, size);
    DEBUG_PRINT("Processor %d: Removing file %s\n", rank, remove_bucket_path);
    MPI_File_delete(remove_bucket_path, MPI_INFO_NULL);
  }
}

/**
 * Global sorting function which fetches a bucket of numbers and
 * sorts them using GPU and CPU (depending on what is currently idle).
 */
void sort(int rank, int size) {
  // Wait for other processors
  MPI_Barrier(MPI_COMM_WORLD);

  char *bucket_id;
  bucket_id = next_bucket_id();
  while (bucket_id) {
    DEBUG_PRINT("Processor %d: Sorting bucket %s\n", rank, bucket_id);
    enqueue_bucket(bucket_id, rank, size);
    sort_bucket(bucket_id, rank, size);
    bucket_id = next_bucket_id();
  }
  DEBUG_PRINT("Processor %d: No pending work. Shutdown.\n", rank);
}
