CUDAMPI
=======

A large hybrid CPU/GPU sorting network using CUDA and MPI.
The sorting network uses a standard Quicksort for CPUs and a custom
Bitonic Sort for GPUs. These two algorithms were the fastest
in a number of prior benchmarks.

We execute the first step of a bucketsort algorithm to presort the data.
A bucket only contains numbers in a given range.
We put each number into its corresponding bucket. This can be done in parallel.
Now each bucket can be sorted on either a CPU or a GPU.

The sorting network uses the filesystem as a process management solution.
Therefore no explicit locks are required.
MPI/IO is used to write to the filesystem in parallel.
The result is a number of sorted files.
