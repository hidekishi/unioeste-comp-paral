#include<stdio.h>
#include<omp.h>

#define N 1000

int  main ()
{
    int a=0, b=0, c=0;

     #pragma omp parallel 
     {
          int id = omp_get_thread_num();
          #pragma omp sections nowait
          {
               #pragma  omp  section 
               {
                    printf("Section 1 - Thread %d\n", id);
                    a++;
               }

               #pragma  omp  section
               {
                     printf("Aula de Arquitetura - Thread %d\n", id);
                     b=b+a;
               }

               #pragma  omp  section
               printf("Section 3 - Thread %d\n", id);

          } /*end of  sections*/
          
          printf("Fora das sections - Thread %d\n", id);
     }
     return  0;
}
