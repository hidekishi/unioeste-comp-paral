
#include <stdio.h>
#include <omp.h>

#define n 8

int main()
{
    int i=0;
    int a=10;
    int id=0;
    int b=888;

	#pragma omp parallel num_threads(4) private(id,a) shared(b)
	{
        id=omp_get_thread_num();
        //a=id;

        #pragma omp for //private(a)
        for (i=0; i<n; i++)
        {
            a = a+1;
            printf("id=%d \t a=%d \t for i=%d\n",id,a,i);
        }
    }

    printf("Qual o valor de a?: %d ",a);

	 return 0;
}
