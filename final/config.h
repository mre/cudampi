#ifndef _CONFIG_H_
#define _CONFIG_H_

#define TAG 1                           // Communication tag
#define BUFSIZE (1*1024*1024UL)         // Write data in chunks with sizeof(float)*BUFSIZE to file
#define NUM_VALS (127*1024*1024UL)      /* Default value of total size of merged file chunks
                                         * (buckets) of all processors. NUM_VALS * sizeof(float).
                                         * Amount of floating point numbers to create.
                                         * Must be multiple of BUFSIZE */
#define MAX_BUCKETSIZE_BYTES (64*1024*1024UL) // Amount of data in each bucket file
#define MAX_NUM 1                       // Maximal number that occurs in data set
#define MIN_NUM 0                       // Minimal number that occurs in data set

//#define UNSORTED_FILENAME "unsorted"    // Storage for unsorted numbers
//#define BUCKET_DIR "buckets"            // Place for unsorted buckets
//#define WORKING_DIR "pending"           // Place for currently used buckets

#define UNSORTED_FILENAME "/tmp-neu/unsorted"    // Storage for unsorted numbers
#define BUCKET_DIR "/tmp-neu/buckets"            // Place for unsorted buckets
#define WORKING_DIR "/tmp-neu/pending"           // Place for currently used buckets

#define FINAL_DIR "sorted"              // Place for sorted output
#define MAXPATHLEN 255                  // Length of path to a file
#define FILENAME_SEPARATOR '_'          // Must be char

// Gain shared/exclusive access to a file using the correct MPI_Communicator
#define MPI_IO_EXCLUSIVE_ACCESS 1 // MPI_COMM_SELF
#define MPI_IO_SHARED_ACCESS 0    // MPI_COMM_WORLD

// Sorting devices used
// Set one of the following flags to use either CPU or GPU exclusively
#define SORT_WITH_GPU_ONLY 0
#define SORT_WITH_CPU_ONLY 1

#define DEBUG
#ifdef DEBUG
  #define DEBUG_PRINT printf
#else
  #define DEBUG_PRINT
#endif

#endif
