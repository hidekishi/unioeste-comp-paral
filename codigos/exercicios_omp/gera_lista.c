#include <stdio.h>
#include <stdlib.h>

#define RANGE 255

int main()
{
    unsigned long int i, N;

    N= 1000000000;

    printf("%li\n", N);
    for(i=0;i<N;i++)
        printf("%d\n",rand()%RANGE);

}