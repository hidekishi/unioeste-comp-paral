#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define deltaX 0.1
#define deltaT 0.1
#define alpha 0.01
#define itmax 3
#define ciclos 10

float timea, timeb;

//---------------------------------------------------------
void copia_vetor(float *dest, float *orig, int tdom)
{
    int i;

    #pragma omp parallel for default(none) \
            private(i) \
            shared(tdom, dest, orig)
    for(i=0;i<tdom;i++)
    {
        dest[i]=orig[i];
    }
}

void mult_mat_vet(float **matriz, int *offset, float *vetor, float *result, int tdom)
{
	float tmp;
	int i,j;


	#pragma omp parallel for   default(none) \
            private(i,j,tmp) \
            shared(tdom, matriz, vetor,offset, result)
	for(i=0;i<tdom;i++)
	{
	      tmp=0.00;
	      for(j=0;j<5;j++)
	      {
		  if(((i+offset[j])>=0)   &&    (matriz[i][j]!=0))
		     tmp+=matriz[i][j]*vetor[i+offset[j]];
	      }
	      result[i]=tmp;
	}
}

float produto_escalar (float *vetor1, float *vetor2, int tdom)
{
	int i;
	float resposta=0;

	#pragma omp parallel for   default(none) \
            private(i) \
            shared(tdom, vetor1, vetor2)\
            reduction (+:resposta)
	for (i=0;i<tdom;i++)
		resposta+= vetor1[i] * vetor2[i];

	return (resposta);
}


void escalar_vetor (float *vetor, float escalar, float *resposta, int tdom)
{
	int i;

	#pragma omp parallel for   default(none) \
            private(i) \
            shared(tdom, resposta, escalar, vetor)
	for (i=0;i<tdom;i++)
		resposta[i]=escalar * vetor[i];
}

void soma_vetor (float *vetor1, float *vetor2, float *resposta, int tdom)
{
	int i;

	#pragma omp parallel for   default(none) \
            private(i) \
            shared(tdom, resposta, vetor1, vetor2)
	for (i=0;i<tdom;i++)
		resposta[i]=vetor1[i]+vetor2[i];
}

void sub_vetor (float *vetor1, float *vetor2, float *resposta, int tdom)
{
	int i;

	#pragma omp parallel for   default(none) \
            private(i) \
            shared(tdom, resposta, vetor1, vetor2)
	for (i=0;i<tdom;i++)
		resposta[i]=vetor1[i]-vetor2[i];
}


//////////////////////////////////////////////////////
//              Gradiente Conjugado                 //
//////////////////////////////////////////////////////
void GC(float **matriz,
	int *offset,
        float *vet1,
        float *resp,
        int tdom)
{
	int it;
	float *r,*d,*q,*aux,*aux2;
	float alfa, beta, sigma0, sigman, sigmav,den;

	r=(float *)calloc(tdom,sizeof(float));
	d=(float *)calloc(tdom,sizeof(float));
	q=(float *)calloc(tdom,sizeof(float));
	aux=(float *)calloc(tdom,sizeof(float));
	aux2=(float *)calloc(tdom,sizeof(float));

	it=0;
	mult_mat_vet(matriz, offset, resp, r, tdom);
	sub_vetor (vet1,r,r,tdom);
	copia_vetor(d, r,tdom);
	sigman=produto_escalar (r,r,tdom);
	sigma0 = sigman;
	copia_vetor(aux2, resp,tdom);

    do
	{
  		mult_mat_vet(matriz, offset, d, q, tdom);
		den=produto_escalar(d,q,tdom);
		alfa=sigman/den;
		escalar_vetor (d,alfa,aux,tdom);
		soma_vetor (resp,aux,resp,tdom);
		escalar_vetor (q,alfa,aux,tdom);
		sub_vetor (r,aux,r,tdom);
		sigmav=sigman;
		sigman=produto_escalar(r,r,tdom);
		beta=sigman/sigmav;
		escalar_vetor (d,beta,aux,tdom);
		soma_vetor (r,aux,d,tdom);
		it++;
	}
	while(it<itmax);
}

//--------------------------------------------------------
int geramatriz(float **dominio, float **matA, float *vetb, float *vetb_ext, int m, int n)
{
    int k, i, j;
    float C, X;
    int tid;

    k=0;
    C=(float) 1+(4*alpha*deltaT)/(deltaX*deltaX);
    X=(float) -(alpha*deltaT)/(deltaX*deltaX);

    #pragma omp parallel for   default(none) \
            private(i,j,k,tid) \
            shared(vetb,vetb_ext,matA,dominio,X,C,m,n)
    for(i = 1; i < m - 1; i++)
    {
	tid=omp_get_thread_num();
        for(j = 1; j < n - 1; j++)
	{
	    k=(i-1)*(m-2)+(j-1);

	    //valor da 1a diagonal
	    if(i==1)
	    {
	      vetb_ext[k]+=-X*dominio[i-1][j];
	    }
	    else
	      matA[k][0]=X;

	    //valor da 2a diagonal
	    if(j==1)
	    {
	      vetb_ext[k]+=-X*dominio[i][j-1];
	    }
	    else
	    {
	      matA[k][1]=X;
	    }

	    //valor da diagonal central
	    matA[k][2]=C;
	    vetb[k]=dominio[i][j];

	    //valor da 4a diagonal
	    if(j==n-2)
	    {
	      vetb_ext[k]+=-X*dominio[i][j+1];
	    }
	    else
	      matA[k][3]=X;

	    //valor da 5a diagonal
	    if(i==n-2)
	    {
	      vetb_ext[k]+=-X*dominio[i+1][j];
	    }
	    else
	      matA[k][4]=X;
	}
    }

    return (m-2)*(n-2);
}

//---------------------------------------------------------

int main()
{
    FILE *input;

    float **dominio;
    float **matA; int *offset;
    float *vetb;
    float *vetb_ext;
    float *vetx;
    float timea, timeb;

    int m,n,size;
    int i,j;

    input = fopen("entrada.dat", "r");
    if(input == NULL)
    {
        perror("Erro ao abrir arquivo.\n");
        return (EXIT_FAILURE);
    }

// alocação dimensão do domínio
    fscanf(input, "%d", &m);
    fscanf(input, "%d", &n);

    offset = (int*) calloc(5, sizeof(int));
    offset[0]=-(n-2);
    offset[1]=-1;
    offset[2]=0;
    offset[3]=1;
    offset[4]=n-2;

    dominio = (float**) calloc(m, sizeof(float *));
    for(i = 0; i < m; i++)
       dominio[i] = (float*) calloc(n, sizeof(float));

    for(i = 0; i < m; i++)
       for(j = 0; j < n; j++)
          fscanf(input, "%f", &dominio[i][j]);

    fclose(input);

//alocação estrutura de dados da pentadiagonal

    matA = (float**) calloc(((n-2)*(m-2)), sizeof(float *));
    for(i = 0; i < ((n-2)*(m-2)); i++)
	matA[i] = (float*) calloc(5, sizeof(float));

//alocação do vetor dos termos independentes b e b_ext e de resposta x
    vetb = (float*) calloc(((n-2)*(m-2)), sizeof(float));
    vetb_ext = (float*) calloc(((n-2)*(m-2)), sizeof(float));
    vetx = (float*) calloc(((n-2)*(m-2)), sizeof(float));


//geração dos sistemas
    size=geramatriz(dominio, matA, vetb, vetb_ext, m, n);

//solução iterativa - ciclos
timea=omp_get_wtime();
    for(j=0;j<ciclos;j++)
    {
	soma_vetor (vetb,vetb_ext,vetb,size);
	GC(matA, offset, vetb, vetx,  size);
	copia_vetor(vetb, vetx,size);

    }
//saida
// 	for(i = 0; i < size; i++)
// 	{
// 	  printf("%4.5f \t ", vetx[i]);
// 	}
// 	printf("\n ");


timeb=omp_get_wtime();
printf("%f \n ",timeb-timea);

return(0);

}


