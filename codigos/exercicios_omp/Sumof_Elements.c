#include<stdio.h>
#include<omp.h>
#include<stdlib.h>

#define array_size 100000000

/* Main Program */

int main()
{
	double  *Array, sum;
	unsigned long int  i;

   /* Dynamic Memory Allocation */
    Array = (double *) malloc(sizeof(double) * array_size);

	/* Array Elements Initialization */

	for (i = 0; i < array_size; i++) {
		Array[i] = 1.0;
	}

	sum = 0.0;

	#pragma omp parallel
	{
		#pragma omp parallel for reduction(+:sum)
		for (i = 0; i < array_size; i++) {
			sum = sum + Array[i];
		}
	}

	/* Freeing Memory */
	free(Array);

	printf("\nThe SumOfElements Of The Array Using OpenMP Directives Is %lf\n", sum);
	
	return 0;
}
