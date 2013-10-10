#include <stdio.h> 
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main() 
{ 
	FILE *file; 
	file = fopen("file.txt","w"); 

srand(time(0));
	int RANGE = 20;
	int random;
	int numValues = 10;	
	int i;
	for (i = 0; i < numValues; ++i) {
		//random = floor((rand()/((double) RAND_MAX + 1))* RANGE) - RANGE/2;
		//random = floor(rand()/((double) RAND_MAX + 1));
		//random = (int)( INT_MAX * rand() / ( RAND_MAX + 1.0 ));
		random = rand();
		fprintf(file,"%i ",random);
	}
	fclose(file);
	return 0; 
}
