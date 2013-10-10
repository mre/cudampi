#ifndef _BUCKET_H_
#define _BUCKET_H_

#include "config.h"

typedef struct bucket_buffer_t {
    float buf[BUFSIZE];
    size_t num_elements;
} bucket_buffer;

void bucket(size_t nums, char *filename, int rank, int size);
size_t bucket_file_num_floats(char *filename, int EXCLUSIVE_ACCESS_FLAG);
float *read_floats(int rank, char *filename, int start, int nfloats, int EXCLUSIVE_ACCESS_FLAG);
MPI_File open_handle(char *bucket_file);

#endif
