#ifndef _RANDGEN_H_
#define _RANDGEN_H_

/* BOX: this union is used to access the bits of floating-point values */
typedef union box {
    float f;
    int i;
} Box;

int get_bit();
float randrange(float min, float max);

#endif
