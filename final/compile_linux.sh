#!/bin/bash

nvcc -O3 -g -c cudampisort.cu -m64
nvcc -O3 -g -c sort.cu -m64
mpicxx -O3 -g -c randgen.c -m64
mpicxx -O3 -g -c generator.cc -m64
mpicxx -O3 -g -c bucket.cc -m64
mpicxx -O3 -g -c qsort.c -m64
mpicxx -O3 -g -o cudampisort *.o "-L/tmp-netz/software/cuda/5.0.0/lib64" -lcudart
