#!/bin/bash

# $RANDOM returns a different random integer at each invocation.
# Nominal range: 0 - 32767 (signed 16-bit integer).

MAXCOUNT=$1
FILE="$1.txt"
count=1

echo "$MAXCOUNT random numbers"
cat /dev/null > $FILE # Clear file contents
while [ "$count" -le $MAXCOUNT ]      # Generate 10 ($MAXCOUNT) random integers.
do
  number=$RANDOM
  echo -n $number >> $FILE
  echo -n " " >> $FILE
  let "count += 1"  # Increment count.
done
exit 0
