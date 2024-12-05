#include <stdio.h>
#include <omp.h>
#include <math.h>

#define n 10000000
#define chunk_size 2

int main()
{
    int i;
    int a[n], b[n], c[n];


    for (i = 0; i < n; i++)
    {
        a[i]=i;
        b[i]=i;
    }

	#pragma omp parallel num_threads(8)
	{
	      //#pragma omp for schedule(static, chunk_size)
	      //#pragma omp for schedule(dynamic, chunk_size)
	      #pragma omp for schedule(guided, chunk_size)
	      //#pragma omp for schedule(runtime, chunk_size)

	      for (i = 0; i < n; i++)
	      {
	          c[i] = a[i]+b[i];
	          printf("Thread %d \t elemento %d\n",omp_get_thread_num(),i);
	      }
	}

    return 0;
}



