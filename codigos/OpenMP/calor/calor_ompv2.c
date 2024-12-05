//difusão de calor - openmp versão 2
//Guilherme Galante
//12/04
//20:05

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define deltaX 0.1
#define deltaT 0.1
#define alpha 0.01
#define itmax 3
#define ciclos 10

//---------------------------------------------------------
void copia_vetor(float *dest, float *orig, int ini, int fim)
{
    int i;
    
    for(i=ini;i<fim;i++)
    {
	dest[i]=orig[i];
    }
}

void mult_mat_vet(float **matriz, int *offset, float *vetor, float *result, int ini, int fim)
{
	float tmp;
	int i,j;
	
	for(i=ini;i<fim;i++)
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

float produto_escalar (float *vetor1, float *vetor2, int ini, int fim)
{
	int i;
	float resposta=0;

	for(i=ini;i<fim;i++)
		resposta+= vetor1[i] * vetor2[i];

	return (resposta);
}


void escalar_vetor (float *vetor, float escalar, float *resposta, int ini, int fim)
{
	int i;
	
	for(i=ini;i<fim;i++)
		resposta[i]=escalar * vetor[i];
}

void soma_vetor (float *vetor1, float *vetor2, float *resposta, int ini, int fim)
{
	int i;

	for(i=ini;i<fim;i++)
		resposta[i]=vetor1[i]+vetor2[i];
}

void sub_vetor (float *vetor1, float *vetor2, float *resposta, int ini, int fim)
{
	int i;
	
	for(i=ini;i<fim;i++)
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
	int it,i;
	float *r,*d,*q,*aux,*pesc;
	float alfa, beta, sigma0, sigman, sigmav,den;
	int ini,fim,tid,tam,resto,nth;
	
	r=(float *)calloc(tdom,sizeof(float));
	d=(float *)calloc(tdom,sizeof(float));
	q=(float *)calloc(tdom,sizeof(float));
	aux=(float *)calloc(tdom,sizeof(float));
	
	it=0;
		
	#pragma omp parallel default(none)\
	    private(ini,fim,tid,i) \
            shared(matriz,offset,resp,r,tdom,vet1,d,sigman,sigma0,sigmav,q,beta,it,den,alfa,aux,tam,nth,resto,pesc)
	 {
	    tid=omp_get_thread_num();
	    nth=omp_get_num_threads();
	    ini=0;
	    fim=0;
	    
	    #pragma omp master
	    {
		tam=(int)tdom/nth;
		resto=tdom%nth;
		pesc=(float *)calloc(nth,sizeof(float));
	    }
	    #pragma omp barrier
	    
	    ini=tid*tam;
	    fim=ini+tam;
	    
	    if(tid==(nth-1))
	      fim=fim+resto;
	    
	    mult_mat_vet(matriz, offset, resp, r, ini,fim);
	    sub_vetor (vet1,r,r,ini,fim);
	    copia_vetor(d, r,ini,fim);
	    
	    pesc[tid]=produto_escalar(r,r,ini,fim);
	    
	    #pragma omp barrier
	    #pragma omp master
	    {
		for(i=0;i<nth;i++)
		  sigman=pesc[i];
	      
		sigma0 = sigman;
	    }
	    #pragma omp barrier
	    
	    do  
	    {
		    mult_mat_vet(matriz, offset, d, q, ini,fim);
		    
		    pesc[tid]=produto_escalar(d,q,ini,fim);
		    #pragma omp barrier
		    #pragma omp master
		    {
		      for(i=0;i<nth;i++)
		      den=pesc[i];
		      alfa=sigman/den;
		    }
		    #pragma omp barrier
		    
		    escalar_vetor (d,alfa,aux,ini,fim);
		    soma_vetor (resp,aux,resp,ini,fim);
		    escalar_vetor (q,alfa,aux,ini,fim);
		    sub_vetor (r,aux,r,ini,fim);
		    
		    #pragma omp master
		    {
		      sigmav=sigman;
		    }
		    
		    pesc[tid]=produto_escalar(r,r,ini,fim);
		    
		    #pragma omp barrier
		    #pragma omp master
		    {
		      for(i=0;i<nth;i++)
			sigman=pesc[i];
		      beta=sigman/sigmav;
		    }
		    #pragma omp barrier
		    
		    escalar_vetor (d,beta,aux,ini,fim);
		    soma_vetor (r,aux,d,ini,fim);
		    
		    #pragma omp master
		    {
			it++;
		    }
		    #pragma omp barrier
	    }
	    while(it<itmax);
	}
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
    
    #pragma omp parallel for default(none) \
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
	    //vetb[k]=-1000;
	    
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
    int i,j,k,t;
    
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

timea=omp_get_wtime(); 
//geração dos sistemas
    size=geramatriz(dominio, matA, vetb, vetb_ext, m, n);
    
//solução iterativa - ciclos
  
    for(j=0;j<ciclos;j++)
    {
	#pragma omp parallel for default(none) \
            private(i) \
            shared(vetb, vetb_ext,size)
	for(i=0;i<size;i++)
	    vetb[i]=vetb[i]+vetb_ext[i];
	
	GC(matA, offset, vetb, vetx,size);
	  
	#pragma omp parallel for default(none) \
            private(i) \
            shared(vetb, vetx,size)
	for(i=0;i<size;i++)
	  vetb[i]=vetx[i];
    }  
//saida    
// 	for(i = 0; i < size; i++)
// 	{
// 	  printf("%4.5f \t ", vetx[i]);
// 	}
// 	printf("\n ");    
   
timeb=omp_get_wtime();
printf("tempo: %f\n ",timeb-timea);
    return(0);
}


