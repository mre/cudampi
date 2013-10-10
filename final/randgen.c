/*
 * Random-number generator by Allen B. Downey
 */
#include <sys/time.h>
#include <stdlib.h>
#include "randgen.h"

/*
 * GET_BIT: returns a random bit. For efficiency, bits are generated 31
 * at a time using the C library function random()
 */
int get_bit() {
    int bit;
    static int bits = 0;
    static int x;

    if (bits == 0) {
      x = random();
      bits = 31;
    }
    bit = x & 1;
    x = x >> 1;
    bits--;
    return bit;
}

/*
 * RANDF: returns a random floating-point
 * number in the range (min,max),
 * including min, subnormals, and max
 */
float randf(float min, float max) {
  int x;
  int mant, exp, high_exp, low_exp;
  Box low, high, ans;

  low.f = min;
  high.f = max;

  /* extract the exponent fields from low and high */
  low_exp = (low.i >> 23) & 0xFF;
  high_exp = (high.i >> 23) & 0xff;

  /* choose random bits and decrement exp until a 1 appears. */
  for (exp = high_exp-1; exp > low_exp; exp--) {
      if (get_bit()) break;
  }

  /* choose random 23-bit mantissa */
  mant = random() & 0x7FFFFF;

  /* if the mantissa is zero, half the time we should move to the next
   * exponent range */
  if (mant == 0 && get_bit()) exp++;

  /* combine exponent and mantissa */
  ans.i = (exp << 23) | mant;

  //return ans.f - 1.175494351e-38F;
  return ans.f;
}

/**
 * Generate a random floating point number x
 * with x element [min, max[
 */
float randrange(float min, float max) {
  float r = randf(min, max);
  // Check if outside of interval
  if (r >= max) {
      return randrange(min, max);
  }
  return r;
}
