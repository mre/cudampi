#!/bin/bash

# Run script for cudampisort

blocksize=512 # MB
datatypesize=4 # Float
buffer=1 # Variation security
log="log" # logfile
repeats=3 # Repeat each test

for i in {1..8}
do
  # Repeat test
  for j in {1..$repeats}
  do
    let amount=$blocksize*$i
    let vals=($amount/$datatypesize-$buffer)*1024*1024

    echo "Starting script with blocksize $blocksize ($vals values)..."
    date                            >> $log
    uname -a                        >> $log
    echo "Size [MiB]: $blocksize"   >> $log
    time ./cudampisort $vals        >> $log
    echo "------------------------" >> $log
    echo "Cleanup..."
    ./cleanup.sh all
    echo
  done
done
