/*
 Description: Kmeans on c
 up1019182
*/


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 540
#define Nv 53
#define Nc 3
#define THR_KMEANS 0.000001


/////////////////////////////////////////////////////////////
int init_vectors();
void init_centers();
float estimateClass();
void estimateCenter();
float Dist(int V_ind, int C_ind);
void test_set();

/////////////////////////////////////////////////////////////
float Vec[N][Nv];
float Center[Nc][Nv];
int Classes[N];

/////////////////////////////////////////////////////////////
int main()
{
	printf("\n- CLASSIFICATION ALGORITHM USED : K-MEANS -\n\n");
    float prevDist, totDist = 1000;
	float a;
    init_vectors();
    init_centers();
 
    do
    {
        prevDist = totDist;
        totDist = estimateClass();
        estimateCenter();
		
		//absolute percentage is required
		a = (prevDist - totDist) > 0 ? (prevDist - totDist)/ totDist : -(prevDist - totDist)/ totDist;

	}while(a > THR_KMEANS);
	
	test_set();
	return 0;
}


/////////////////////////////////////////////////////////////
int init_vectors()
{
	FILE *fp ;
	char B[850], *p ;
	
	fp = fopen( "preprocessed_clinical.txt", "r" ) ;
	if ( fp == NULL )
			return -1 ;
	
	for( int j = 0 ; j < N ; j++ )
	{
		if ( fgets( B, 850, fp ) != B )
				return -2 ;

		p = strtok(B,";") ;
		if ( p == NULL )
				return -3 ;
			
		p = strtok(NULL,";" ) ;
		Classes[j] = atoi(p) ;

		for( int i = 0 ; i < Nv ; i++ )  
		{
				p = strtok(NULL,";" ) ;
				Vec[j][i] = atof(p) ;
		}
	}
	fclose(fp) ;	
	
	for(int i = 0; i < Nv; i++)
	{
		if(i == 32 || i == 40) continue;
		float max = 0;
		for(int j = 0; j < N; j++)
			max = max > Vec[j][i] ? max : Vec[j][i];
		for(int j = 0; j < N; j++)
			Vec[j][i] /= max;
	}
}


/////////////////////////////////////////////////////////////
void init_centers()
{
	
	int i, j;
    int ind[Nc], count = 1;
    ind[0] = rand() % ((int)(0.7*N));
    
    do
    {
        int New = rand() % ((int)(0.7*N));
        int flag = 0;
        
        for(i = 0; i < count; i++)
            if(ind[i] == New)
                flag = 1;
                
        if(!flag)
            ind[count++] = New;
            
    }while(count < Nc);
    
    
    for(i = 0; i < Nc; i++)
        for(j = 0; j < Nv; j++)
			Center[i][j] = Vec[ind[i]][j];
}


/////////////////////////////////////////////////////////////
float estimateClass()
{
    float min, CurrDist, totDist = 0;
    int classInd = 0;
    
#pragma omp parallel for reduction(min:min) reduction(+:totDist) private(CurrDist, classInd) schedule(static, 30)
    for(int i = 0; i < (int)(0.7*N); i++)
    {
        min = 1000;
        for(int j = 0; j < Nc; j++)
        {
            CurrDist = Dist(i, j);
            totDist += CurrDist;
            if(CurrDist < min)
            {
                min = CurrDist;
                classInd = j;
            }
        }
        Classes[i] = classInd;
    }
    return totDist;
}


/////////////////////////////////////////////////////////////
float Dist(int V_ind, int C_ind)
{
    float sum = 0;
    
#pragma omp simd reduction(+:sum) //αυτό είναι πιο γρήγορο
    for(int i = 0; i < Nv; i++)
	{
		float t = (Vec[V_ind][i] - Center[C_ind][i]);
        sum += t*t;
    }
    return sqrt(sum);
}


/////////////////////////////////////////////////////////////
void estimateCenter()
{
	
#pragma omp parallel for 
    for(int i = 0; i < Nc; i++) 
    {
        float count = 0;
        float temp = 0;

        for(int j = 0; j < (int)(0.7*N); j++)
        {
            if(Classes[j] == i)
            {
                for(int k = 0; k < Nv; k++)
                {
                    //set center to the first vector found for that class
                    if(!count) 
						Center[i][k] = Vec[j][k];  
                    else Center[i][k] += Vec[j][k];
                }
                count++;
            }
        }
		
		count = 1.0/count;
#pragma omp simd 
        for(int j = 0; j < Nv; j++)
            Center[i][j] *= count;
    }
}

void test_set()
{
	float accuracy = 0;
	for(int i = ((int)(0.7*N)); i < N; i++)
	{
		float min = 100000;
		int class;
		for(int j = 0; j < Nc; j++)
		{
			float tmp = Dist(i, j);
			min = tmp > min ? min : tmp;
			if(min == tmp) class = j;
		}
		if(class == Classes[i]) accuracy += 1;
	}
	printf("Training set percentage : 0.70\nTesting set percentage : 0.30\nClassification accuracy : %d out of %d samples\nClassification error : %f\n", (int) accuracy, (int) (0.3*N), 1 - accuracy / (0.3*N) );			
}



