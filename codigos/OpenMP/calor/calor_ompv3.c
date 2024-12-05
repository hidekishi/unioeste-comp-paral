#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define deltaX 0.1
#define deltaT 0.1
#define alpha 0.01
#define itmax 3
#define ciclos 10
#define TROCAS 2

float C,X;
int tid;

//---------------------------------------------------------
void mult_mat_vet(float **matriz, int *offset, float *vetor, float *result, int tdom)
{
	float tmp;
	int y,j;
	
   	for(y=0;y<tdom;y++)
  	{
 	      tmp=0.00;
 	      for(j=0;j<5;j++)
 	      {
		  if(((y+offset[j])>=0)   &&    (matriz[y][j]!=0))
		  {
		     tmp+=matriz[y][j]*vetor[y+offset[j]];
		  }
 	      }
 	      result[y]=tmp;
  	}
}

float produto_escalar (float *vetor1, float *vetor2, int tdom)
{
	int i;
	float resposta=0;

	for (i=0;i<tdom;i++)
		resposta+= vetor1[i] * vetor2[i];

	return (resposta);
}


void escalar_vetor (float *vetor, float escalar, float *resposta, int tdom)
{
	int i;
	for (i=0;i<tdom;i++)
		resposta[i]=escalar * vetor[i];
}

void soma_vetor (float *vetor1, float *vetor2, float *resposta, int tdom)
{
	int i;

	for (i=0;i<tdom;i++)
		resposta[i]=vetor1[i]+vetor2[i];
}

void sub_vetor (float *vetor1, float *vetor2, float *resposta, int tdom)
{
	int i;

	for (i=0;i<tdom;i++)
		resposta[i]=vetor1[i]-vetor2[i];
}


//////////////////////////////////////////////////////
//              Gradiente Conjugado                 //
//////////////////////////////////////////////////////
void GC(float **matriz,
	int *offset,
        float *vet1,
        float *vetx,
        int ini,
	int fim)
{
	int it,tdom,i;
	float *r,*d,*q,*aux;//,*resp;
	float alfa, beta, sigma0, sigman, sigmav,den;

	tdom=fim-ini;
	
	r=(float *)calloc(tdom,sizeof(float));
	d=(float *)calloc(tdom,sizeof(float));
	q=(float *)calloc(tdom,sizeof(float));
	aux=(float *)calloc(tdom,sizeof(float));

	it=0;
	mult_mat_vet(matriz, offset, vetx+ini, r, tdom);
	sub_vetor (vet1,r,r,tdom);
	memcpy(d,r,tdom*sizeof(float));
	sigman=produto_escalar (r,r,tdom);
	sigma0 = sigman;
 
        do  
	{
  		mult_mat_vet(matriz, offset, d, q, tdom);
		den=produto_escalar(d,q,tdom);
		alfa=sigman/den;
		escalar_vetor (d,alfa,aux,tdom);
		soma_vetor (vetx+ini,aux,vetx+ini,tdom);
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
int geramatriz(float **dominio, float **matA, float *vetb, float *vetb_ext, int ini, int fim,int m,int *vetb_extup,int *vetb_extdw)
{
    int k, i, j;

    C=(float) 1+(4*alpha*deltaT)/(deltaX*deltaX);
    X=(float) -(alpha*deltaT)/(deltaX*deltaX);
 
 
    k=0;
    for(i = ini; i < fim; i++)
    {
        for(j = 1; j < m - 1; j++)
	{
	    //valor da 1a diagonal
	    if(i==1)
	    {
		vetb_ext[k]+=-X*dominio[i-1][j];
	    }
	    else
	    if(ini==i)
	    {
	        vetb_extup[j-1]=(i-2)*(m-2)+(j-1);
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
	    if(j==m-2)
	    {
	      vetb_ext[k]+=-X*dominio[i][j+1];
	    }
	    else
	      matA[k][3]=X;
	    
	    if(i==m-2)
	    {
	      vetb_ext[k]+=-X*dominio[i+1][j];
	    }
	    else
	    if(i==fim-1)
	    {
	        vetb_extdw[j-1]=(i)*(m-2)+(j-1);
	    }
	    else
	      matA[k][4]=X;
    
	    k++;
	}
    }
	
    return k;
}

//---------------------------------------------------------

int main()
{
    float **dominio;
    float **matA;
    int *offset;
    float *vetb;
    float *vetb_orig;
    float *vetb_ext;
    float *vetx;
    int *vetb_extup;
    int *vetb_extdw;
   
    FILE *input;
    
    int m,n,size,st;
    int i,j,k;
    int tam,ini,fim,nth,resto;
    
    float timea,timeb;
    
    input = fopen("entrada.dat", "r");
    if(input == NULL){
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
    if (dominio == NULL) 
    {
        printf("** Erro: Memoria Insuficiente **\n");
	return (EXIT_FAILURE);
    }
    
    for(i = 0; i < m; i++)
    {
        dominio[i] = (float*) calloc(n, sizeof(float));
        if (dominio[i] == NULL)
	{
            printf("** Erro: Memoria Insuficiente **\n");
            return (EXIT_FAILURE);
        }
    }

    vetx = (float*) calloc(((m-2)*(m-2)), sizeof(float));

    for(i = 0; i < m; i++)
    for(j = 0; j < n; j++)
       fscanf(input, "%f", &dominio[i][j]);
	
    fclose(input);
    
    k=0;
    for(i = 1; i < m-1; i++)
    {
      for(j = 1; j < m-1; j++)
      {
	  vetx[k]=dominio[i][j];
	  k++;
      }  
    }
    
//alocação estrutura de dados da pentadiagonal
    
    
    #pragma omp parallel default(none)\
	    private(matA,vetb,vetb_ext,vetb_orig,i,j,ini,fim,tid,size,vetb_extup,vetb_extdw,st,k)\
            shared(vetx,dominio,tam,resto,m,nth,X,offset,timea)
    {
	    tid=omp_get_thread_num();
	    ini=1;
	    fim=0;
	    
	    #pragma omp master
	    {
		nth=omp_get_num_threads();
		tam=(int)(m-2)/nth;
		resto=(m-2)%nth;
	    }
	    #pragma omp barrier
	    
	    ini=tid*tam+1;
	    fim=ini+tam;
	    
	    if(tid==(nth-1))
	      fim=fim+resto;
    
	    matA = (float**) calloc(((fim-ini)*(m-2)), sizeof(float *));
	    for(i = 0; i < ((fim-ini)*(m-2)); i++)
	      matA[i] = (float*) calloc(5, sizeof(float));
	    
	    vetb      = (float*) calloc(((fim-ini)*(m-2)), sizeof(float));
	    vetb_orig = (float*) calloc(((fim-ini)*(m-2)), sizeof(float));
	    vetb_ext  = (float*) calloc(((fim-ini)*(m-2)), sizeof(float));
	    vetb_extup = (int*) calloc((m-2), sizeof(int));
	    vetb_extdw = (int*) calloc((m-2), sizeof(int));

	    #pragma omp barrier
	    
#pragma omp master
{
  timea=omp_get_wtime();
}
//geração dos sistemas
	    size=geramatriz(dominio, matA, vetb, vetb_ext, ini, fim,m,vetb_extup,vetb_extdw);
	    
//solução iterativa - ciclos
	    for(j=0;j<ciclos;j++)
	    {
		    for(i=0;i<size;i++)
		      vetb[i]=vetb[i]+vetb_ext[i];
		    
//subciclos - troca de dados das fronteiras		    
		    for(k=0;k<TROCAS;k++)
		    {     
			for(i=0;i<size;i++)
			  vetb_orig[i]=vetb[i];
			
			for(i=0;i<m-2;i++)
			{
			  if(vetb_extup[i]!=0)
			  {  
			    vetb_orig[i]+=-X*vetx[vetb_extup[i]];
			  }
			}

			st=size-(m-2);
			for(i=0;i<m-2;i++)
			{
			  if(vetb_extdw[i]!=0)
			  {  
			    vetb_orig[st+i]=vetb_orig[st+i]-X*vetx[vetb_extdw[i]];
			  }
			}
			#pragma omp barrier
			
			GC(matA, offset, vetb_orig, vetx,((ini-1)*(m-2)),((ini-1)*(m-2))+size);
			
			#pragma omp barrier
 		    }
	    
		    for(i=0;i<size;i++)
			vetb[i]=vetx[i+(ini-1)*(m-2)];
		      
		      #pragma omp barrier
  	    }
//saida - pode ocorrer fora de ordem!  	    
// 		    #pragma omp critical
// 		    {
// 		    for(i = ((ini-1)*(m-2)); i < ((ini-1)*(m-2))+size; i++)
// 		    {
// 			  printf("%d:%4.5f\n ", tid,vetx[i]);
// 		    }
// 		    printf("\n ");
// 		    }

     }
    timeb=omp_get_wtime();
    printf("tempo: %f\n ",timeb-timea);
    return(0);
}


