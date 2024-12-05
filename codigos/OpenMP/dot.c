#include <stdio.h>

#define n 200

int main()
{
    int i=0;
    int dot=0;
    int a[n];
    int b[n];

    for (i = 0; i < n; i++)
    {
        a[i]=1;
        b[i]=1;
    }

    for (i = 0; i < n; i++)
    {
        dot += a[i]*b[i];
    }

    printf("Dot: %d \n\n", dot);
    return 0;
}
