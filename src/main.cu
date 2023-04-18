#include"common.h"
#include"mmio_highlevel.h"
#include"utils.h"
#include"utils_tile.h"
#include"step.h"
#include"mask_bfs.h"
#include <thrust/sort.h>
#include"beidoublas_format_csr2beidou.h"

# define INDEX_DATA_TYPE unsigned char
#define WARMUP_NUM 200
#define WARP_SIZE 32
#define WARP_PER_BLOCK 2

#define num_f 240
#define num_b 15

#define PREFETCH_SMEM_TH 8
#define COO_NNZ_TH 12
#define DEBUG_FORMATCOST 0
#define MAX_TIME 999999

int main(int argc, char ** argv)
{

    Beidou_Tile_Matrix *matrixA_d = (Beidou_Tile_Matrix *)malloc(sizeof(Beidou_Tile_Matrix));
    struct timeval t1, t2;
    int argi = 1;
    int nthreads;
    if(argc > argi)
    {
        nthreads = atoi(argv[argi]);
        argi++;
    }
    omp_set_num_threads(nthreads);
    
    char  *filename;
    filename = argv[2];
    printf("MAT: -------------- %s --------------\n", filename);
    int bfs_source=0;
    bfs_source=atoi(argv[3]);
    
    
    // int use_numeric=0;
    // use_numeric=atoi(argv[4]);  //0 is not calculating values, 1 is calculating values
    
    printf("bfs_source = %d\n",bfs_source);
    // load mtx A data to the csr format
    gettimeofday(&t1, NULL);
    mmio_allinone(&matrixA_d->m, &matrixA_d->n, &matrixA_d->nnz, &matrixA_d->isSymmetric, &matrixA_d->rowpointer, &matrixA_d->columnidx, &matrixA_d->value, filename);
    gettimeofday(&t2, NULL);
    
    double time_loadmat  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("input matrix A: ( %i, %i ) nnz = %i\n loadfile time    = %4.5f sec\n", matrixA_d->m, matrixA_d->n, matrixA_d->nnz, time_loadmat/1000.0);
    if(matrixA_d->isSymmetric==0)
    {
        int *cscColPtr = (int *)malloc((matrixA_d->n + 1) * sizeof(int));
        int *cscRowIdx = (int *)malloc(matrixA_d->nnz * sizeof(int));
        MAT_VAL_TYPE *cscVal = (MAT_VAL_TYPE *)malloc(matrixA_d->nnz * sizeof(MAT_VAL_TYPE));
        matrix_transposition(matrixA_d->m, matrixA_d->n, matrixA_d->nnz, matrixA_d->rowpointer, matrixA_d->columnidx, matrixA_d->value,cscRowIdx, cscColPtr, cscVal);
    
        int *csrRowPtr_AAT = (int *)malloc((matrixA_d->n + 1) * sizeof(int));
        int *csrColIdx_AAT = (int *)malloc(matrixA_d->nnz * sizeof(int)*2); 
        int AATlen=0;
        for(int i=0;i<matrixA_d->m;i++)
        {
            int len1=cscColPtr[i+1]-cscColPtr[i];
            int len2=matrixA_d->rowpointer[i+1]-matrixA_d->rowpointer[i];
            csrRowPtr_AAT[i]=MergeArr(cscRowIdx+cscColPtr[i],len1,matrixA_d->columnidx+matrixA_d->rowpointer[i],len2,csrColIdx_AAT+AATlen);
            AATlen+=csrRowPtr_AAT[i];
        }
        exclusive_scan(csrRowPtr_AAT,matrixA_d->n+1);
        matrixA_d->nnz=0;
        for(int i=0;i<matrixA_d->m;i++)
        {
            matrixA_d->rowpointer[i]=0;
            for(int j=csrRowPtr_AAT[i];j<csrRowPtr_AAT[i+1];j++)
            {
                matrixA_d->rowpointer[i]++;
                matrixA_d->columnidx[matrixA_d->nnz++]=csrColIdx_AAT[j];          
            }
        }
   
        matrixA_d->rowpointer[matrixA_d->m]=0;
        exclusive_scan(matrixA_d->rowpointer,matrixA_d->m+1);
   
        free(cscColPtr);
        free(cscRowIdx);
        free(cscVal);
    
        free(csrRowPtr_AAT);
        free(csrColIdx_AAT);
    }

    int BLOCK_SIZE;
    if(matrixA_d->m > 10000)
    {
        BLOCK_SIZE=64;
        matrixA_d->blocksize_flag=1;
    }
    else
    {
        BLOCK_SIZE=32;
        matrixA_d->blocksize_flag=0;
    }
    if(bfs_source>=matrixA_d->m)
    {
        printf("error:bfs_source is too big!!!\n");
        return;
    }
    for (int i = 0; i < matrixA_d->nnz; i++)
	    matrixA_d->value[i] = i % 10 +1;//i % 10 +1


    matrixA_d->numtile =0;
    matrixA_d->tilem = matrixA_d->m%BLOCK_SIZE==0 ? matrixA_d->m/BLOCK_SIZE : (matrixA_d->m/BLOCK_SIZE)+1 ;
    matrixA_d->tilen = matrixA_d->n%BLOCK_SIZE==0 ? matrixA_d->n/BLOCK_SIZE : (matrixA_d->n/BLOCK_SIZE)+1 ;
    matrixA_d->tile_ptr=(MAT_PTR_TYPE *)malloc((matrixA_d->tilem+1)*sizeof(MAT_PTR_TYPE));
    memset(matrixA_d->tile_ptr, 0, (matrixA_d->tilem+1)*sizeof(MAT_PTR_TYPE));
    
    int *new_coo_rowidx_1;
    int *new_coo_colidx_1;
    MAT_VAL_TYPE *new_coo_value_1;
    int *new_coocount_1;

    format_transform(matrixA_d);
//-------------------balance--------------------------------
    
    balance(matrixA_d,bfs_source);

    unsigned int * d_blkcoostylerowidx;
    int * d_blkcoostylerowidx_colstart;
    int * d_blkcoostylerowidx_colstop;
    int * d_blk_colid;
    int * d_idx_xxx;

    cudaMalloc((void **)&d_blkcoostylerowidx, matrixA_d->rowblkblock * sizeof(unsigned int));
    cudaMalloc((void **)&d_blkcoostylerowidx_colstart, matrixA_d->rowblkblock * sizeof(int));
    cudaMalloc((void **)&d_blkcoostylerowidx_colstop, matrixA_d->rowblkblock * sizeof(int));
    cudaMalloc((void **)&d_blk_colid, (matrixA_d->tilem+1) * sizeof(int));
    cudaMalloc((void **)&d_idx_xxx, (BLOCK_SIZE+2) * sizeof(int));

    cudaMemcpy(d_blkcoostylerowidx, matrixA_d->blkcoostylerowidx, matrixA_d->rowblkblock * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcoostylerowidx_colstart, matrixA_d->blkcoostylerowidx_colstart, matrixA_d->rowblkblock * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcoostylerowidx_colstop, matrixA_d->blkcoostylerowidx_colstop, matrixA_d->rowblkblock * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blk_colid, matrixA_d->blk_colid, (matrixA_d->tilem+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_idx_xxx, 0, (BLOCK_SIZE+2)*sizeof(int) );
    
    MAT_VAL_TYPE *x = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * matrixA_d->n);
    memset(x, 0, sizeof(MAT_VAL_TYPE) * matrixA_d->n);
    x[bfs_source]=1;

    MAT_PTR_TYPE *d_tile_ptr_A;
    cudaMalloc((void **)&d_tile_ptr_A, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1) );
    cudaMemcpy(d_tile_ptr_A, matrixA_d->tile_ptr, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1), cudaMemcpyHostToDevice);
    
    int *d_tile_columnidx;
    cudaMalloc((void **)&d_tile_columnidx, (matrixA_d->numtile ) * sizeof(int) );
    cudaMemcpy(d_tile_columnidx, matrixA_d->tile_columnidx, (matrixA_d->numtile) * sizeof(int), cudaMemcpyHostToDevice);
 
    //------------------------------------------------ mask -------------------------------------------------------
    int direct=matrixA_d->m;
    if(matrixA_d->blocksize_flag==0)
    {
        double time_mask_bfs=0;
        int mask_iter=0;
        for(int iter=0;iter<20;iter++)
        {        
            unsigned int *mask_x = (unsigned int *)malloc((matrixA_d->tilem) * sizeof(unsigned int));
            memset(mask_x, 0, sizeof(unsigned int) * (matrixA_d->tilem));
            unsigned int *d_mask_x;
            cudaMalloc((void **)&d_mask_x, (matrixA_d->tilem)*sizeof(unsigned int));
            unsigned int *d_mask_x_1;
            cudaMalloc((void **)&d_mask_x_1, (matrixA_d->tilem)*sizeof(unsigned int));

            for(int i=0;i<matrixA_d->m;i++)
            {
                if(x[i]!=0)
                {
                    mask_x[i/BLOCK_SIZE] |= (1<<(BLOCK_SIZE-1-i%BLOCK_SIZE)) ;   
                }
            }

            cudaMemcpy(d_mask_x, mask_x, sizeof(unsigned int) * (matrixA_d->tilem), cudaMemcpyHostToDevice); 
            cudaMemcpy(d_mask_x_1, mask_x, sizeof(unsigned int) * (matrixA_d->tilem), cudaMemcpyHostToDevice); 

            unsigned int *mask_y = (unsigned int *)malloc((matrixA_d->tilem) * sizeof(unsigned int));
            memset(mask_y, 0, sizeof(unsigned int) * (matrixA_d->tilem));
            unsigned int *d_mask_y;
            cudaMalloc((void **)&d_mask_y, (matrixA_d->tilem)*sizeof(unsigned int));
            cudaMemcpy(d_mask_y, mask_y, sizeof(unsigned int) * (matrixA_d->tilem), cudaMemcpyHostToDevice); 

            unsigned int *d_mask;
            cudaMalloc((void **)&d_mask, matrixA_d->numtile * BLOCK_SIZE * sizeof(unsigned int));
            cudaMemcpy(d_mask, matrixA_d->mask_32, matrixA_d->numtile * BLOCK_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice);

            unsigned int *mask_flag = (unsigned int *)malloc((matrixA_d->tilem) * sizeof(unsigned int));
            memset(mask_flag, 0, sizeof(unsigned int) * (matrixA_d->tilem));
            mask_flag[bfs_source/BLOCK_SIZE]=1<<(BLOCK_SIZE-1-bfs_source%BLOCK_SIZE);
    
            unsigned int *d_mask_flag;
            cudaMalloc((void **)&d_mask_flag, sizeof(unsigned int) * (matrixA_d->tilem));
            cudaMemcpy(d_mask_flag, mask_flag, sizeof(unsigned int) * (matrixA_d->tilem), cudaMemcpyHostToDevice);
            unsigned int *d_flag_1;
            cudaMalloc((void **)&d_flag_1, sizeof(unsigned int));
            unsigned int *d_flag_2;
            cudaMalloc((void **)&d_flag_2, sizeof(unsigned int));
            cudaMemset(d_flag_1, 0, sizeof(unsigned int) );
            cudaMemset(d_flag_2, 0, sizeof(unsigned int) );
    
            unsigned int *flag_1=(unsigned int *)malloc( sizeof(unsigned int));
            unsigned int *flag_2=(unsigned int *)malloc( sizeof(unsigned int));
   
            flag_2[0]=1;
            flag_1[0]=1;
            cudaMemcpy(d_flag_2, flag_2, sizeof(unsigned int) , cudaMemcpyHostToDevice);
            cudaMemcpy(d_flag_1, flag_1, sizeof(unsigned int) , cudaMemcpyHostToDevice);
    
            unsigned int *d_mask_x_c;
            cudaMalloc((void **)&d_mask_x_c, direct*sizeof(unsigned int));
            cudaMemset(d_mask_x_c, 0, direct*sizeof(unsigned int));
    
            unsigned int *d_mask_x_c_1;
            cudaMalloc((void **)&d_mask_x_c_1, direct*sizeof(unsigned int));
            cudaMemset(d_mask_x_c_1, 0, direct*sizeof(unsigned int));
    
            unsigned int *d_flag_1_c;
            cudaMalloc((void **)&d_flag_1_c, (matrixA_d->m)*sizeof(unsigned int));
            cudaMemset(d_flag_1_c, 0, (matrixA_d->m)*sizeof(unsigned int));
   
            mask_iter=0;
            double time_cuda_spmv_mask=0;
            
            do{ 
    	        int num_threads = WARP_PER_BLOCK * WARP_SIZE;
                int num_blocks = ceil ((double)matrixA_d->rowblkblock / (double)(num_threads / WARP_SIZE));
        
                gettimeofday(&t1, NULL);
                cudaMemset(d_mask_y, 0, sizeof(int) * (matrixA_d->tilem));
                cudaMemset(d_flag_1, 0, sizeof(int) );
        
                if(DEBUG_FORMATCOST)
                {   
                    int num_blocks = ceil ((double)flag_2[0] / (double)(num_threads / WARP_SIZE));
                    if(mask_iter%2==0)
                    {
                        cudaMemset(d_mask_x_1, 0, sizeof(unsigned int ) * (matrixA_d->tilem) );
    	                mask_spmv_32_v1<<< num_blocks, num_threads >>>(matrixA_d->tilem,matrixA_d->tilen,matrixA_d->m,matrixA_d->n,
    	                d_tile_ptr_A,  d_tile_columnidx,
    	                matrixA_d->rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,d_blk_colid,
    	                d_mask_x, d_mask_x_1, d_mask_y,  d_mask ,d_mask_flag,
    	                d_flag_1,d_mask_x_c,d_mask_x_c_1,d_idx_xxx,
    	                d_flag_1_c,d_flag_2);  
    	            }
    	            else
    	            {
    	                cudaMemset(d_mask_x, 0, sizeof(unsigned int ) * (matrixA_d->tilem) );
    	                mask_spmv_32_v1<<< num_blocks, num_threads >>>(matrixA_d->tilem,matrixA_d->tilen,matrixA_d->m,matrixA_d->n,
    	                d_tile_ptr_A,  d_tile_columnidx,
    	                matrixA_d->rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,d_blk_colid,
    	                d_mask_x_1,d_mask_x,  d_mask_y,  d_mask ,d_mask_flag,d_flag_1,d_mask_x_c_1,d_mask_x_c,
    	                d_idx_xxx,d_flag_1_c,d_flag_2);
                    }   
                }
                else
                {
                    if(mask_iter%2==0)
                    {
                        cudaMemset(d_mask_x_1, 0, sizeof(unsigned int ) * (matrixA_d->tilem) );
    	                mask_spmv_32_v2<<< num_blocks, num_threads >>>(matrixA_d->tilem,matrixA_d->tilen,matrixA_d->m,matrixA_d->n,
    	                d_tile_ptr_A,  d_tile_columnidx,
    	                matrixA_d->rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
    	                d_mask_x, d_mask_x_1, d_mask_y,  d_mask ,d_mask_flag,d_flag_1);//d_mask_tile_x,d_mask_tile_y,  
    	            }
    	            else
    	            {
    	                cudaMemset(d_mask_x, 0, sizeof(unsigned int ) * (matrixA_d->tilem) );
    	                mask_spmv_32_v2<<< num_blocks, num_threads >>>(matrixA_d->tilem,matrixA_d->tilen,matrixA_d->m,matrixA_d->n,
    	                d_tile_ptr_A,  d_tile_columnidx,
    	                matrixA_d->rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
    	                d_mask_x_1,d_mask_x,  d_mask_y,  d_mask ,d_mask_flag,d_flag_1);
    	            }
    	        }
    	        cudaMemcpy(flag_1, d_flag_1, sizeof(int) , cudaMemcpyDeviceToHost);
    
    	        gettimeofday(&t2, NULL);
                time_cuda_spmv_mask += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
                mask_iter++;
  
            }while(flag_1[0]==1);

            //    printf("iter %d : time_cuda_spmv_mask=%f   mask_iter=%d\n",iter,time_cuda_spmv_mask,mask_iter);
    
           
            time_mask_bfs+=time_cuda_spmv_mask;
    
            free(mask_x);
            free(mask_y);
            cudaFree(d_mask);
            free(mask_flag);
            cudaFree(d_mask_x);
            cudaFree(d_mask_y);
            cudaFree(d_mask);
            cudaFree(d_mask_flag);
            cudaFree(d_mask_x_c);
            cudaFree(d_mask_x_1);
            cudaFree(d_flag_1);
            cudaFree(d_flag_2);
            cudaFree(d_flag_1_c);
            cudaFree(d_mask_x_c_1);
        }
        time_mask_bfs /= 20;
        printf("time_mask_bfs= %f ms  mask_iter=%d\n",time_mask_bfs,mask_iter);
        FILE *fout=fopen("results.csv","a");
        if(fout==NULL) printf("writing results falis!\n");
        fprintf(fout,"%s,%d,%d,%d,%d,%f\n",filename,matrixA_d->m,matrixA_d->n,matrixA_d->nnz,mask_iter,time_mask_bfs);
        fclose(fout);
    }
    else
    {
        double time_mask_bfs=0;
        int mask_iter=0;
        for(int iter=0;iter<20;iter++)
        {
            unsigned long long int  *mask_x = (unsigned long long int  *)malloc((matrixA_d->tilem) * sizeof(unsigned long long int ));
            memset(mask_x, 0, sizeof(unsigned long long int ) * (matrixA_d->tilem));
            unsigned long long int  *d_mask_x;
            cudaMalloc((void **)&d_mask_x, (matrixA_d->tilem)*sizeof(unsigned long long int ));
            unsigned long long int  *d_mask_x_1;
            cudaMalloc((void **)&d_mask_x_1, (matrixA_d->tilem)*sizeof(unsigned long long int ));
   
            for(int i=0;i<matrixA_d->m;i++)
            {
                if(x[i]!=0)
                {
                    mask_x[i/BLOCK_SIZE] |= ((unsigned long long int )1<<(BLOCK_SIZE-1-i%BLOCK_SIZE)) ;
                }
            }
   
            cudaMemcpy(d_mask_x, mask_x, sizeof(unsigned long long int ) * (matrixA_d->tilem), cudaMemcpyHostToDevice); 
            cudaMemcpy(d_mask_x_1, mask_x, sizeof(unsigned long long int ) * (matrixA_d->tilem), cudaMemcpyHostToDevice); 
  
            unsigned long long int  *mask_y = (unsigned long long int  *)malloc((matrixA_d->tilem) * sizeof(unsigned long long int ));
            memset(mask_y, 0, sizeof(unsigned long long int ) * (matrixA_d->tilem));
            unsigned long long int *d_mask_y;
            cudaMalloc((void **)&d_mask_y, (matrixA_d->tilem)*sizeof(unsigned long long int ));
            cudaMemcpy(d_mask_y, mask_y, sizeof(unsigned long long int ) * (matrixA_d->tilem), cudaMemcpyHostToDevice); 
 
            unsigned long long int  *d_mask;
            cudaMalloc((void **)&d_mask, matrixA_d->numtile * BLOCK_SIZE * sizeof(unsigned long long int ));
            cudaMemcpy(d_mask, matrixA_d->mask_64, matrixA_d->numtile * BLOCK_SIZE * sizeof(unsigned long long int ), cudaMemcpyHostToDevice);
 
            unsigned long long int  *mask_flag = (unsigned long long int  *)malloc((matrixA_d->tilem) * sizeof(unsigned long long int ));
            memset(mask_flag, 0, sizeof(unsigned long long int ) * (matrixA_d->tilem));
            mask_flag[bfs_source/BLOCK_SIZE]=((unsigned long long int )1)<<(BLOCK_SIZE-1-bfs_source%BLOCK_SIZE);
    
            unsigned long long int *d_mask_flag;
            cudaMalloc((void **)&d_mask_flag, sizeof(unsigned long long int) * (matrixA_d->tilem));
            cudaMemcpy(d_mask_flag, mask_flag, sizeof(unsigned long long int) * (matrixA_d->tilem), cudaMemcpyHostToDevice);

            unsigned long long int *d_mask_flag_1;
            cudaMalloc((void **)&d_mask_flag_1, sizeof(unsigned long long int) * (matrixA_d->tilem));
            unsigned long long int *mask_flag_1=(unsigned long long int  *)malloc((matrixA_d->tilem) * sizeof(unsigned long long int ));
            
            unsigned long long int *d_flag_1;
            cudaMalloc((void **)&d_flag_1, sizeof(unsigned long long int));
            unsigned long long int *d_flag_2;
            cudaMalloc((void **)&d_flag_2, sizeof(unsigned long long int));
            cudaMemset(d_flag_1, 0, sizeof(unsigned long long int) );
            cudaMemset(d_flag_2, 0, sizeof(unsigned long long int) );
    
            unsigned long long int *flag_1=(unsigned long long int *)malloc( sizeof(unsigned long long int));
            unsigned long long int *flag_2=(unsigned long long int *)malloc( sizeof(unsigned long long int));

            flag_2[0]=1;
            flag_1[0]=1;
            cudaMemcpy(d_flag_2, flag_2, sizeof(unsigned long long int) , cudaMemcpyHostToDevice);
            cudaMemcpy(d_flag_1, flag_1, sizeof(unsigned long long int) , cudaMemcpyHostToDevice);
    
            int *d_mask_x_c;
            cudaMalloc((void **)&d_mask_x_c, direct*sizeof(int));
            cudaMemset(d_mask_x_c, 0, direct*sizeof(int));
    
            int *d_mask_x_c_1;
            cudaMalloc((void **)&d_mask_x_c_1, direct*sizeof(int));
            cudaMemset(d_mask_x_c_1, 0, direct*sizeof(int));
    
            unsigned long long int *d_flag_1_c;
            cudaMalloc((void **)&d_flag_1_c, (matrixA_d->m)*sizeof(unsigned long long int));
            cudaMemset(d_flag_1_c, 0, (matrixA_d->m)*sizeof(unsigned long long int) );

            mask_iter=0;
            double time_cuda_spmv_mask=0;
            int num_threads = WARP_PER_BLOCK * WARP_SIZE;
            int sub_num_x=matrixA_d->m;
            int pull_flag=0;
            int pull_num_blocks=0;
            int pull_num_x=0;
            do{
       
                double ratio=(1.0*flag_2[0])/matrixA_d->m;
    	        int ratio_flag=0;
                sub_num_x-=flag_2[0];

                gettimeofday(&t1, NULL);
                cudaMemset(d_mask_y, 0, sizeof(unsigned long long int ) * (matrixA_d->tilem));
                cudaMemset(d_flag_1, 0, sizeof(unsigned long long int) );
                
                if(pull_flag==1 || sub_num_x<=0.5*flag_2[0])
                {
                    ratio_flag=2;
                    if(pull_flag==0)
                    {
                        int num_blocks = ceil (((double)matrixA_d->tilem) / (double)(num_threads));

                        mask_pull_flag<<< num_blocks, num_threads >>>(d_mask_x_c,d_mask_x_c_1,d_mask_flag,d_flag_1,matrixA_d->tilem,matrixA_d->m,d_flag_1_c);
                        pull_flag=1;
                        cudaMemcpy(flag_2, d_flag_1, sizeof(unsigned long long int) , cudaMemcpyDeviceToHost);
                        pull_num_blocks=ceil ((double)flag_2[0] / (double)(num_threads / WARP_SIZE));
                        pull_num_x=flag_2[0];
                        cudaMemcpy(d_mask_flag_1, d_mask_flag, sizeof(unsigned long long int) * (matrixA_d->tilem), cudaMemcpyDeviceToDevice);
                    }
                    cudaMemset(d_flag_1, 0, sizeof(unsigned long long int) );
                    cudaMemcpy(mask_flag_1, d_mask_flag_1, sizeof(unsigned long long int) * (matrixA_d->tilem), cudaMemcpyDeviceToHost);
    	            
                    mask_spmv_64_pull_v1<<< pull_num_blocks, num_threads >>>(matrixA_d->tilem,matrixA_d->tilen,matrixA_d->m,matrixA_d->n,
    	            d_tile_ptr_A,  d_tile_columnidx,
    	            matrixA_d->rowblkblock, d_blkcoostylerowidx,d_blkcoostylerowidx_colstart,d_blkcoostylerowidx_colstop,
    	            d_blk_colid,d_mask_x, d_mask_x_1, d_mask_y,  d_mask ,d_mask_flag,d_mask_flag_1,
    	            d_flag_1,d_mask_x_c,d_mask_x_c_1,d_idx_xxx,d_flag_1_c,d_flag_2,pull_num_x); 
    	            
                    cudaMemcpy(d_mask_flag, d_mask_flag_1, sizeof(unsigned long long int) * (matrixA_d->tilem), cudaMemcpyDeviceToDevice);
                }
                else if(ratio<0.01)
                {
                    ratio_flag=0;
                    int num_blocks = ceil ((double)flag_2[0] / (double)(num_threads / WARP_SIZE));
        
                    if(mask_iter%2==0)
                    {
                        cudaMemset(d_mask_x_1, 0, sizeof(unsigned long long int ) * (matrixA_d->tilem) );
    	                
                        mask_spmv_64_v1<<< num_blocks, num_threads >>>(matrixA_d->tilem,matrixA_d->tilen,matrixA_d->m,matrixA_d->n,
    	                d_tile_ptr_A,  d_tile_columnidx,
    	                matrixA_d->rowblkblock, d_blkcoostylerowidx,d_blkcoostylerowidx_colstart,d_blkcoostylerowidx_colstop,
    	                d_blk_colid,d_mask_x, d_mask_x_1, d_mask_y,  d_mask ,d_mask_flag,
    	                d_flag_1,d_mask_x_c,d_mask_x_c_1,d_idx_xxx,d_flag_1_c,d_flag_2); 
    	            }
    	            else
    	            {
    	                cudaMemset(d_mask_x, 0, sizeof(unsigned long long int ) * (matrixA_d->tilem) );
    	            
                        mask_spmv_64_v1<<< num_blocks, num_threads >>>(matrixA_d->tilem,matrixA_d->tilen,matrixA_d->m,matrixA_d->n,
    	                d_tile_ptr_A,  d_tile_columnidx,
    	                matrixA_d->rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
    	                d_blk_colid,d_mask_x_1,d_mask_x,  d_mask_y, d_mask ,d_mask_flag,
    	                d_flag_1,d_mask_x_c_1,d_mask_x_c,d_idx_xxx,d_flag_1_c,d_flag_2);
                    }
                }
                else
                {
                    ratio_flag=1;
                    int num_blocks = ceil ((double)matrixA_d->rowblkblock / (double)(num_threads / WARP_SIZE));
             
                    if(mask_iter%2==0)
                    {
                        cudaMemset(d_mask_x_1, 0, sizeof(unsigned long long int ) * (matrixA_d->tilem) );
    	            
                        mask_spmv_64_v2<<< num_blocks, num_threads >>>(matrixA_d->tilem,matrixA_d->tilen,matrixA_d->m,matrixA_d->n,
    	                d_tile_ptr_A,  d_tile_columnidx,
    	                matrixA_d->rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
    	                d_mask_x,d_mask_x_1,  d_mask_y, d_mask ,d_mask_flag,
    	                d_flag_1,d_mask_x_c_1,d_mask_x_c,d_idx_xxx,d_flag_1_c,d_flag_2
    	                );//d_mask_tile_x,d_mask_tile_y,
    	            }
    	            else
    	            {
    	                cudaMemset(d_mask_x, 0, sizeof(unsigned long long int ) * (matrixA_d->tilem) );
    	            
                        mask_spmv_64_v2<<< num_blocks, num_threads >>>(matrixA_d->tilem,matrixA_d->tilen,matrixA_d->m,matrixA_d->n,
    	                d_tile_ptr_A,  d_tile_columnidx,
    	                matrixA_d->rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
    	                d_mask_x_1,d_mask_x,  d_mask_y,  d_mask ,d_mask_flag,
    	                d_flag_1,d_mask_x_c,d_mask_x_c_1,d_idx_xxx,d_flag_1_c,d_flag_2
    	                );
    	            }
    	 
    	        }
    	 
    	        cudaMemcpy(flag_2, d_flag_1, sizeof(unsigned long long int) , cudaMemcpyDeviceToHost);
    	        cudaMemcpy(d_flag_2, d_flag_1, sizeof(unsigned long long int) , cudaMemcpyDeviceToDevice);
    	        cudaMemset(d_flag_1, 0, sizeof(unsigned long long int) );
    
    	        gettimeofday(&t2, NULL);
                double time_iter =(t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
                time_cuda_spmv_mask += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
                mask_iter++;
        
            }while(flag_2[0]!=0);//
   
            
            time_mask_bfs+=time_cuda_spmv_mask;
    
            free(mask_x);
            free(mask_y);
            cudaFree(d_mask);
            free(mask_flag);
            cudaFree(d_mask_x);
            cudaFree(d_mask_y);
            cudaFree(d_mask);
            cudaFree(d_mask_flag);
            cudaFree(d_mask_flag_1);
            cudaFree(d_mask_x_c);
            cudaFree(d_mask_x_1);
            cudaFree(d_flag_1);
            cudaFree(d_flag_2);
            cudaFree(d_flag_1_c);
            cudaFree(d_mask_x_c_1);
        }
        time_mask_bfs /= 20;
        printf("time_mask_bfs= %f ms   mask_iter=%d \n",time_mask_bfs,mask_iter);

        FILE *fout=fopen("results.csv","a");
        if(fout==NULL) printf("writing results falis!\n");
        fprintf(fout,"%s,%d,%d,%d,%d,%f\n",filename,matrixA_d->m,matrixA_d->n,matrixA_d->nnz,mask_iter,time_mask_bfs);
        fclose(fout);
    
    }
    cudaFree(d_tile_ptr_A);
    cudaFree(d_tile_columnidx);
    cudaFree(d_blkcoostylerowidx);
    cudaFree(d_blkcoostylerowidx_colstart);
    cudaFree(d_blkcoostylerowidx_colstop);

    Tile_destroy(matrixA_d);
   
    return 0;
}
