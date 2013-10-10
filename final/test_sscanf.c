#include <stdio.h>
#include <stdlib.h>

main() {
    unsigned long long val;
    char test[100] = "12312323234234224398";
    sscanf(test, "%llu", &val);
    printf("%llu\n", val);
}
