#include<thrust/scan.h>
__global__
void cuda_step1_kernel_new(int rowA, int colA, int *d_rowpointerA, int *d_columnindexA, 
    int tilemA, int tilenA, int *d_tile_ptr_A, unsigned int *d_flag_t, int row_start_idx, int row_end_idx,int num_threads,int n_tile,int BLOCK_SIZE)
{

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_id<num_threads)
    {

    int row=row_start_idx;
    for(int i=row_start_idx;i<=row_end_idx;i++)
    {
        if(d_rowpointerA[row_start_idx]+global_id<d_rowpointerA[i+1])
        {
            row=i;
            break;
        }
    }
    int col=d_columnindexA[d_rowpointerA[row_start_idx]+global_id];
    int idx=((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE;
   // int y=32;
    int flag_idx=(idx>>5); //>>5=/32  &31=%32
    unsigned int flag_val=0;
    flag_val|=(1<<(idx%32));
   // x|=(1<<y)
   if((atomicOr(&d_flag_t[flag_idx],flag_val)&flag_val)==0)
   {
       atomicAdd(&d_tile_ptr_A[row/BLOCK_SIZE],1);
   }


}
    __syncthreads();

}

__global__
void cuda_step2_kernel_new_sort_1(int rowA, int colA, int *d_rowpointerA, int *d_columnindexA, 
                  int tilemA, int tilenA,int num_threads,int row_start_idx, int row_end_idx, MAT_PTR_TYPE *d_tile_ptr_A, int *d_tile_columnidx, 
                  int *d_tile_nnz, int *d_tile_csr_ptr, int numtileA,int *d_j_col,int n_tile,unsigned int *d_flag_t, int *d_j_num_t,int BLOCK_SIZE)
{

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_id<num_threads)
    {
    int row=row_start_idx;
    for(int i=row_start_idx;i<=row_end_idx;i++)
    {
        if(d_rowpointerA[row_start_idx]+global_id<d_rowpointerA[i+1])
        {
            row=i;
            break;
        }
    }
     int col=d_columnindexA[d_rowpointerA[row_start_idx]+global_id];
    int tileptr=d_tile_ptr_A[row/BLOCK_SIZE];
    int j=0;
    
    int idx=((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE;
    int flag_idx=(idx>>5); //>>5=/32  &31=%32
    unsigned int flag_val=0;
    flag_val|=(1<<(idx%32));
   if((atomicOr(&d_flag_t[flag_idx],flag_val)&flag_val)==0)
    {
        d_j_num_t[idx]=atomicAdd(&d_j_col[(row/BLOCK_SIZE)%n_tile],1);
        j=d_j_num_t[idx];
        d_tile_columnidx[tileptr+j]=col/BLOCK_SIZE;
    }
}
__syncthreads();
   
}


__global__
void cuda_step2_2_kernel_new_sort_2(int rowA, int colA, int *d_rowpointerA, int *d_columnindexA, 
                  int tilemA, int tilenA,int num_threads,int row_start_idx, int row_end_idx, MAT_PTR_TYPE *d_tile_ptr_A, int *d_tile_columnidx, 
                  int *d_tile_nnz, int *d_tile_csr_ptr, int numtileA,int *d_j_col,int n_tile,unsigned int *d_flag_t, int *d_j_num_t,int BLOCK_SIZE)
{

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_id<num_threads)
    {
    int row=row_start_idx;
    for(int i=row_start_idx;i<=row_end_idx;i++)
    {
        if(d_rowpointerA[row_start_idx]+global_id<d_rowpointerA[i+1])
        {
            row=i;
            break;
        }
    }
    int col=d_columnindexA[d_rowpointerA[row_start_idx]+global_id];
    int tileptr=d_tile_ptr_A[row/BLOCK_SIZE];
    int j=d_j_num_t[((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE];
    atomicAdd(&d_tile_nnz[tileptr+j],1);
    atomicAdd(&d_tile_csr_ptr[(tileptr+j)*BLOCK_SIZE+row%BLOCK_SIZE],1);

}
__syncthreads();
   
}



__inline__ __device__
void swap_key_cuda(int *a , int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}
__inline__ __device__
int partition_key_cuda(int *key, int length, int pivot_index , int *nnz, int *csr_ptr)
{
    int i  = 0 ;
    int small_length = pivot_index;

 
 
    return small_length;
}

__inline__ __device__
void bubbleSort(int *src, int n,int *nnz,int *csr_ptr,int BLOCK_SIZE)
{
	int i = 0, j = 0, tmp = 0;
	for(i = n - 1; i > 0; --i){
		for(j = 0; j < i; ++j){
			if(src[j] > src[j + 1]){
				tmp = src[j];
				src[j] = src[j + 1];
				src[j + 1] = tmp;     //数据交换
				swap_key_cuda(&nnz[j],&nnz[j+1]);
				for(int k=0;k<BLOCK_SIZE;k++) 
                              {
                                    swap_key_cuda(&csr_ptr[(j)*BLOCK_SIZE+k], &csr_ptr[BLOCK_SIZE*(j+1)+k]);
                              }
			}
		}
	}
}


__global__
void quick_sort_cuda(int *d_tile_columnidx,int *d_tile_nnz,int *d_tile_csr_ptr,int end,int blkj,int *d_tile_ptr_A,int BLOCK_SIZE)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(global_id<end)
    {
    int blki=global_id+blkj;
 

bubbleSort(d_tile_columnidx + d_tile_ptr_A[blki],d_tile_ptr_A[blki+1] - d_tile_ptr_A[blki],d_tile_nnz + d_tile_ptr_A[blki], d_tile_csr_ptr+d_tile_ptr_A[blki]*BLOCK_SIZE,BLOCK_SIZE);
     }
}

__global__
void cuda_step2_kernel_new(int rowA, int colA, int *d_rowpointerA, int *d_columnindexA, 
                  int tilemA, int tilenA,int num_threads,int row_start_idx, int row_end_idx, MAT_PTR_TYPE *d_tile_ptr_A, int *d_tile_columnidx, 
                  int *d_tile_nnz, int *d_tile_csr_ptr, int numtileA,int *d_j_col,int n_tile,unsigned int *d_flag_t, int *d_j_num_t,int BLOCK_SIZE)
{

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_id<num_threads)
    {
    int row=row_start_idx;
    for(int i=row_start_idx;i<=row_end_idx;i++)
    {
        if(d_rowpointerA[row_start_idx]+global_id<d_rowpointerA[i+1])
        {
            row=i;
            break;
        }
    }
    int col=d_columnindexA[d_rowpointerA[row_start_idx]+global_id];
    int idx=((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE;
    d_j_num_t[idx]=1;
    
    int flag_idx=(idx>>5); //>>5=/32  &31=%32
    int flag_val= (1<<(idx%32));
    atomicOr(&d_flag_t[flag_idx],flag_val);
}
__syncthreads();
   
}


__global__
void cuda_step2_2_kernel_new(int rowA, int colA, int *d_rowpointerA, int *d_columnindexA, 
                  int tilemA, int tilenA,int num_threads,int row_start_idx, int row_end_idx, MAT_PTR_TYPE *d_tile_ptr_A, int *d_tile_columnidx, 
                  int *d_tile_nnz, int *d_tile_csr_ptr, int numtileA,int *d_j_col,int n_tile,unsigned int *d_flag_t, int *d_j_num_t,int BLOCK_SIZE)
{

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_id<num_threads)
    {
    int row=row_start_idx;
    for(int i=row_start_idx;i<=row_end_idx;i++)
    {
        if(d_rowpointerA[row_start_idx]+global_id<d_rowpointerA[i+1])
        {
            row=i;
            break;
        }
    }
    int col=d_columnindexA[d_rowpointerA[row_start_idx]+global_id];
    int tileptr=d_tile_ptr_A[row_start_idx/BLOCK_SIZE];
    int idx=((row/BLOCK_SIZE)%n_tile)*tilenA+col/BLOCK_SIZE;
int j=0;
    for(int i=0;i<idx;i++)
    {
      int k=((d_flag_t[i>>5])>>(i%32)&1);
        if(k==1)
        {
            j++;
        }
    }
    d_tile_columnidx[tileptr+j]=col/BLOCK_SIZE;
    atomicAdd(&d_tile_nnz[tileptr+j],1);
    atomicAdd(&d_tile_csr_ptr[(tileptr+j)*BLOCK_SIZE+row%BLOCK_SIZE],1);

}
__syncthreads();
   
}

 

__global__
void cuda_step3_kernel(int rowA, int colA, int *rowpointerA, int *columnindexA, 
                  int tilemA, int tilenA, int numtileA, MAT_PTR_TYPE *tile_ptr_A, 
                  int *tile_columnidx, int *tile_nnz,int *blknnz,
                  int *csr_offset, int *csrptr_offset,
                  int blki_1,int num_tile_row,int BLOCK_SIZE)
{

    if(threadIdx.x==1)
    {
    int begin_blki=blki_1;
    int blockid=blockIdx.x;    
    int blki;
    for(int i=0;i<num_tile_row&&i+begin_blki<tilemA;i++)
    {
        if(tile_ptr_A[begin_blki]+blockid<tile_ptr_A[begin_blki+i+1])
        {
            blki=i+begin_blki;
            break;
        }
    }
    if(blki==tilemA)
       blki--;
   
    int rowlen= blki==tilemA-1 ? rowA-(tilemA-1)*BLOCK_SIZE : BLOCK_SIZE ;
      
    int collen = tile_columnidx[tile_ptr_A[begin_blki]+blockid] == tilenA-1 ? colA - (tilenA-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
    int tile_id = tile_ptr_A[begin_blki]+blockid;
    int tilennz = tile_nnz[tile_id +1] - tile_nnz[tile_id];
    int nnzthreshold = rowlen * collen * 0.5 ;

 //   Format[tile_id] =0 ;
    blknnz[tile_id] = tilennz ;
    csr_offset[tile_id] = tilennz;
    csrptr_offset[tile_id] = BLOCK_SIZE;
                
         /*       Format[tile_id] = 4 ;
               // printf("Format=%d\n",Format[blki]);
               // printf("444444444444  blockid=%d\n",blockid);
                blknnz[tile_id] = rowlen * collen;
                dns_offset[tile_id] = rowlen * collen;*/
                return;             
    }
__syncthreads();
}


__inline__ __device__
void exclusive_scan_cu(MAT_PTR_TYPE *input, int length)
{
    if(length == 0 || length == 1)
        return;
    
    MAT_PTR_TYPE old_val, new_val;
    
    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}

__global__
void cuda_step4_kernel_1(int rowA, int colA, int *rowpointerA, int *columnindexA, MAT_VAL_TYPE *valueA,
                  int tilemA, int tilenA, int numtileA, MAT_PTR_TYPE *tile_ptr_A, int *tile_columnidx, int *tile_nnz, 
                  int *blknnz, int *csr_ptr,  int nnz_temp,  int tile_count_temp,
                  unsigned char  *csr_colidx_temp_g,MAT_VAL_TYPE *csr_val_temp_g,int *tile_count_g,int blki_1,int num_tile_row,int BLOCK_SIZE)
{

   if(threadIdx.x==1)
    {
        int begin_blki=blki_1;
        int blockid=blockIdx.x;    
        int blki=begin_blki+blockid;
   
        int thread_id = blockIdx.x;
        unsigned char  *csr_colidx_temp = csr_colidx_temp_g + thread_id * nnz_temp;
        MAT_VAL_TYPE *csr_val_temp = csr_val_temp_g + thread_id * nnz_temp;
        int *tile_count = tile_count_g + thread_id * tile_count_temp;
     
        int tilenum_per_row=tile_ptr_A[blki+1]-tile_ptr_A[blki];
        int rowlen= blki==tilemA-1 ? rowA-(tilemA-1)*BLOCK_SIZE : BLOCK_SIZE ;
        int start = blki*BLOCK_SIZE;
        int end = blki==tilemA-1 ?  rowA : (blki+1)*BLOCK_SIZE ;
        
       for (int blkj = rowpointerA[start]; blkj < rowpointerA[end]; blkj ++)
        {
           int jc_temp = columnindexA[blkj]/BLOCK_SIZE;
             for (int bi = 0; bi < tilenum_per_row; bi ++)
            {
                int tile_id = tile_ptr_A[blki]+bi;
                int jc = tile_columnidx[tile_id];
                int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr_A[blki]];
               
                 if (jc == jc_temp)
                {
                    csr_val_temp[pre_nnz + tile_count[bi]] = valueA[blkj];
                    csr_colidx_temp[pre_nnz + tile_count[bi]] = columnindexA[blkj] - jc * BLOCK_SIZE;
                    
                    tile_count[bi] ++;  //woshicuode
                    break;
                      
                }
            }
        }
    }
__syncthreads();
}




__global__
void cuda_step4_kernel_2_64(int rowA, int colA, int *rowpointerA, int *columnindexA, MAT_VAL_TYPE *valueA,
                  int tilemA, int tilenA, int numtileA, MAT_PTR_TYPE *tile_ptr_A, int *tile_columnidx, int *tile_nnz, 
                  int *blknnz, int *csr_ptr,  int nnz_temp,  int tile_count_temp,
                  unsigned char  *csr_colidx_temp_g,MAT_VAL_TYPE *csr_val_temp_g,int *tile_count_g,
                 MAT_VAL_TYPE *Tile_csr_Val, unsigned char  *Tile_csr_Col, unsigned char  *Tile_csr_Ptr, int *csr_offset, int *csrptr_offset,
       
                 int blki_1, int num_tile_row,unsigned long long int *mask,int BLOCK_SIZE)
{
    if(threadIdx.x==1)
    {
        int begin_blki=blki_1;
        int blockid=blockIdx.x;    
        int blki;
        for(int i=0;i<num_tile_row&&i+begin_blki<tilemA;i++)
        {
            if(tile_ptr_A[begin_blki]+blockid<tile_ptr_A[begin_blki+i+1])
            {
                blki=i+begin_blki;
                break;
            }
        }
        if(blki==tilemA)
            blki--;

        MAT_VAL_TYPE *csr_val_temp = csr_val_temp_g + (blki%num_tile_row) * nnz_temp;
        unsigned char  *csr_colidx_temp = csr_colidx_temp_g + (blki%num_tile_row) * nnz_temp;
        
        int rowlen= blki==tilemA-1 ? rowA-(tilemA-1)*BLOCK_SIZE : BLOCK_SIZE ;
        int bi = blockIdx.x;
            int tile_id = tile_ptr_A[begin_blki]+bi;
            int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr_A[blki]];
            int tilennz = tile_nnz[tile_id +1] - tile_nnz[tile_id];
            int collen = tile_columnidx[tile_id] == tilenA-1 ? colA - (tilenA-1 ) * BLOCK_SIZE : BLOCK_SIZE ;              
                    int offset = csr_offset[tile_id];
                    int ptr_offset = csrptr_offset[tile_id];
                    int *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                  
                    exclusive_scan_cu(ptr_temp, BLOCK_SIZE);
                    int k1=0;
                    for (int ri = 0; ri < BLOCK_SIZE; ri++)
                {
                    int start = ptr_temp[ri];
                    int stop = ri == BLOCK_SIZE - 1 ? tilennz : ptr_temp[ri + 1];
                   
                    for (int k = start; k < stop; k++)
                    {
                        unsigned char colidx = csr_colidx_temp[pre_nnz + k];
                        Tile_csr_Val[offset + k] = csr_val_temp[pre_nnz + k];
                        Tile_csr_Col[offset + k] = csr_colidx_temp[pre_nnz + k];
                        mask[tile_id * BLOCK_SIZE + ri] |= (((unsigned long long int )0x1) << (BLOCK_SIZE - colidx - 1));
                     
                    }
                    Tile_csr_Ptr[ptr_offset + ri] = ptr_temp[ri];
                }
        }
__syncthreads();

    }

__global__
void cuda_step4_kernel_2_32(int rowA, int colA, int *rowpointerA, int *columnindexA, MAT_VAL_TYPE *valueA,
                  int tilemA, int tilenA, int numtileA, MAT_PTR_TYPE *tile_ptr_A, int *tile_columnidx, int *tile_nnz,  
                  int *blknnz, int *csr_ptr,  int nnz_temp,  int tile_count_temp,
                  unsigned char  *csr_colidx_temp_g,MAT_VAL_TYPE *csr_val_temp_g,int *tile_count_g,
                 MAT_VAL_TYPE *Tile_csr_Val, unsigned char  *Tile_csr_Col, 
                 unsigned char  *Tile_csr_Ptr, int *csr_offset, int *csrptr_offset,           
                 int blki_1, int num_tile_row,unsigned int *mask,int BLOCK_SIZE)
{
    if(threadIdx.x==1)
    {
        int begin_blki=blki_1;
        int blockid=blockIdx.x;    
        int blki;
        for(int i=0;i<num_tile_row&&i+begin_blki<tilemA;i++)
        {
            if(tile_ptr_A[begin_blki]+blockid<tile_ptr_A[begin_blki+i+1])
            {
                blki=i+begin_blki;
                break;
            }
        }
        if(blki==tilemA)
            blki--;
        
        MAT_VAL_TYPE *csr_val_temp = csr_val_temp_g + (blki%num_tile_row) * nnz_temp;
        unsigned char  *csr_colidx_temp = csr_colidx_temp_g + (blki%num_tile_row) * nnz_temp;
        
        int rowlen= blki==tilemA-1 ? rowA-(tilemA-1)*BLOCK_SIZE : BLOCK_SIZE ;
        int bi = blockIdx.x;
            int tile_id = tile_ptr_A[begin_blki]+bi;
            int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr_A[blki]];
            int tilennz = tile_nnz[tile_id +1] - tile_nnz[tile_id];
            int collen = tile_columnidx[tile_id] == tilenA-1 ? colA - (tilenA-1 ) * BLOCK_SIZE : BLOCK_SIZE ;
                    int offset = csr_offset[tile_id];
                    int ptr_offset = csrptr_offset[tile_id];
                
                    int *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                  
                    exclusive_scan_cu(ptr_temp, BLOCK_SIZE);
                    int k1=0;

                    for (int ri = 0; ri < BLOCK_SIZE; ri++)
                {
                    int start = ptr_temp[ri];
                    int stop = ri == BLOCK_SIZE - 1 ? tilennz : ptr_temp[ri + 1];
                   
                    for (int k = start; k < stop; k++)
                    {
                        unsigned char colidx = csr_colidx_temp[pre_nnz + k];
                        Tile_csr_Val[offset + k] = csr_val_temp[pre_nnz + k];
                        Tile_csr_Col[offset + k] = csr_colidx_temp[pre_nnz + k];
                        mask[tile_id * BLOCK_SIZE + ri] |= (unsigned int)(0x1 << (BLOCK_SIZE - colidx - 1));
                     
                    }
                    Tile_csr_Ptr[ptr_offset + ri] = ptr_temp[ri];
                }

        }
__syncthreads();

    }
   


__inline__ __device__
int d_binary_search_right_boundary_kernel(const int *row_pointer,
                                        const int  key_input,
                                        const int  size)
{
    int start = 0;
    int stop  = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}

__global__
void cuda_bal_step2(int stridennz, int nnz, int nthreads, int rowblkblock, int *flag_tilerow_start, int *group_ptr,int BLOCK_SIZE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<nthreads)
    {
        // compute partition boundaries by partition of size stride
	int boundary = tid * stridennz;
	// clamp partition boundaries to [0, nnzR]
	boundary = boundary > nnz ? nnz : boundary;
	// binary search
	flag_tilerow_start[tid] = d_binary_search_right_boundary_kernel(group_ptr, boundary,  rowblkblock+ 1) - 1;
    }
__syncthreads();
}

void d_format_transform(Beidou_Tile_Matrix *matrixA_d,  MAT_VAL_TYPE **new_coo_value_temp, int **new_coo_colidx_temp, int **new_coo_rowidx_temp, int **new_coocount_temp,int BLOCK_SIZE)
{ 
    size_t limit=4096;
    cudaDeviceSetLimit(cudaLimitStackSize,limit);
    struct timeval t1, t2,t3,t4;
    int num_threads, num_blocks;
    num_blocks = matrixA_d->tilem*matrixA_d->tilen; 
    MAT_PTR_TYPE *d_rowpointerA;
    MAT_PTR_TYPE *d_columnindexA;

    MAT_PTR_TYPE *d_tile_ptr_A;
    unsigned int *d_flag_t;
    int n_tile=64;
    unsigned int *flag_t=(unsigned int *)malloc(n_tile*matrixA_d->tilen * sizeof(unsigned int)/32+32);///32+32
    memset(flag_t,0,n_tile*matrixA_d->tilen * sizeof(unsigned int)/32+32);

    cudaMalloc((void **)&d_flag_t, n_tile*matrixA_d->tilen * sizeof(unsigned int)/32+32 );

    cudaMalloc((void **)&d_rowpointerA, sizeof(MAT_PTR_TYPE) *(matrixA_d->m+1) );
    cudaMemcpy(d_rowpointerA, matrixA_d->rowpointer, sizeof(MAT_PTR_TYPE) *(matrixA_d->m+1), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_columnindexA, sizeof(MAT_PTR_TYPE) *(matrixA_d->nnz+1) );
    cudaMemcpy(d_columnindexA, matrixA_d->columnidx, sizeof(MAT_PTR_TYPE) *(matrixA_d->nnz+1), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_tile_ptr_A, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1) );
    cudaMemcpy(d_tile_ptr_A, matrixA_d->tile_ptr, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1), cudaMemcpyHostToDevice);

    double time_cuda_step1 = 0;
    double time_cuda_step1_sum = 0;
    gettimeofday(&t3, NULL);
    for(int i=0;i<matrixA_d->tilem;i+=n_tile)
    {   
        cudaMemset(d_flag_t,0,n_tile*matrixA_d->tilen * sizeof(unsigned int)/32+32);
        int row_start_idx=i*BLOCK_SIZE;
        int row_end_idx=i*BLOCK_SIZE+n_tile*BLOCK_SIZE>matrixA_d->m-1 ? row_end_idx=matrixA_d->m-1 : row_end_idx=i*BLOCK_SIZE+n_tile*BLOCK_SIZE-1;
        
        num_threads= matrixA_d->rowpointer[row_end_idx+1]-matrixA_d->rowpointer[row_start_idx];
        num_blocks=num_threads/64+1;
       
        gettimeofday(&t1, NULL);

        
        cuda_step1_kernel_new<<< num_blocks, 64 >>>(matrixA_d->m, matrixA_d->n, d_rowpointerA, d_columnindexA, matrixA_d->tilem, matrixA_d->tilen, d_tile_ptr_A,d_flag_t,row_start_idx,row_end_idx,num_threads,n_tile,BLOCK_SIZE);

        gettimeofday(&t2, NULL);
        
        time_cuda_step1 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    }
    gettimeofday(&t4, NULL);
    printf("transform_step1(cuda) runtime    = %4.5f ms\n", time_cuda_step1);
    
    time_cuda_step1_sum = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
  
 cudaMemcpy(matrixA_d->tile_ptr, d_tile_ptr_A, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1), cudaMemcpyDeviceToHost);
    //cuda-step1-end
    exclusive_scan(matrixA_d->tile_ptr, matrixA_d->tilem+1);
  
    matrixA_d->numtile = matrixA_d->tile_ptr[matrixA_d->tilem];
    printf("matrixA_d->numtile=%d\n",matrixA_d->numtile);
//cuda-step2
    cudaMemcpy(d_tile_ptr_A, matrixA_d->tile_ptr, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1), cudaMemcpyHostToDevice);

    int *d_tile_columnidx;
  
    matrixA_d->tile_columnidx=(int *)malloc((matrixA_d->numtile+1)*sizeof(int));
    memset(matrixA_d->tile_columnidx, 0, (matrixA_d->numtile+1)*sizeof(int));
    cudaMalloc((void **)&d_tile_columnidx, (matrixA_d->numtile+1) * sizeof(int) );
    cudaMemcpy(d_tile_columnidx, matrixA_d->tile_columnidx, (matrixA_d->numtile+1) * sizeof(int), cudaMemcpyHostToDevice);

   
    matrixA_d->tile_nnz=(int *)malloc((matrixA_d->numtile + 1)* sizeof(int));
    int *d_tile_nnz;
    memset(matrixA_d->tile_nnz,0,(matrixA_d->numtile + 1) * sizeof(int));
    cudaMalloc((void **)&d_tile_nnz, (matrixA_d->numtile+1) * sizeof(int) );
    cudaMemcpy(d_tile_nnz, matrixA_d->tile_nnz, (matrixA_d->numtile+1) * sizeof(int), cudaMemcpyHostToDevice);

    int *d_tile_csr_ptr;
    matrixA_d->csr_ptr_1 = (int *)malloc(((matrixA_d->numtile + 1) * BLOCK_SIZE) * sizeof(int));
   
    memset (matrixA_d->csr_ptr_1, 0, ((matrixA_d->numtile + 1) * BLOCK_SIZE) * sizeof(int));
    cudaMalloc((void **)&d_tile_csr_ptr, ((matrixA_d->numtile + 1) * BLOCK_SIZE) * sizeof(int) );
    cudaMemcpy(d_tile_csr_ptr, matrixA_d->csr_ptr_1, ((matrixA_d->numtile + 1) * BLOCK_SIZE) * sizeof(int), cudaMemcpyHostToDevice);

      int *d_j_col;
    int *j_col=(int *)malloc((n_tile+1)*sizeof(int));
    memset(j_col, 0, (n_tile+1)*sizeof(int));
    cudaMalloc((void **)&d_j_col, (n_tile+1) * sizeof(int) );

   int *d_j_num_t_1;
    int *j_num_t_1=(int *)malloc(n_tile*matrixA_d->tilen * sizeof(int));
    memset(j_num_t_1,0,n_tile*matrixA_d->tilen * sizeof(int));
   cudaMalloc((void **)&d_j_num_t_1, n_tile*matrixA_d->tilen * sizeof(int) );
   
      int *d_flag_idx;
    int *flag_idx=(int *)malloc(n_tile * sizeof(int));
    memset(flag_idx,0,n_tile * sizeof(int));
   cudaMalloc((void **)&d_flag_idx, n_tile * sizeof(int) );
 
printf("matrixA_d->tilem=%d\n",matrixA_d->tilem);
n_tile=4;


    double time_cuda_step2 = 0;
    double time_cuda_step2_sum = 0;
    double time_cuda_step2_2 = 0;
    gettimeofday(&t3, NULL);
    if(matrixA_d->m<100000){
    for(int i=0;i<matrixA_d->tilem;i+=n_tile)//
    {   

        cudaMemset(d_flag_t,0,n_tile*matrixA_d->tilen * sizeof(unsigned int)/32+32);
        cudaMemset(d_j_num_t_1,0,n_tile*matrixA_d->tilen * sizeof(int));
        int row_start_idx=i*BLOCK_SIZE;
        int row_end_idx=i*BLOCK_SIZE+n_tile*BLOCK_SIZE>matrixA_d->m-1 ? row_end_idx=matrixA_d->m-1 : row_end_idx=i*BLOCK_SIZE+n_tile*BLOCK_SIZE-1;
        
        num_threads= matrixA_d->rowpointer[row_end_idx+1]-matrixA_d->rowpointer[row_start_idx];
        num_blocks=num_threads/64+1;
       
        gettimeofday(&t1, NULL);
       
        cuda_step2_kernel_new<<< num_blocks, 64 >>>(matrixA_d->m, matrixA_d->n, d_rowpointerA, d_columnindexA, matrixA_d->tilem, matrixA_d->tilen, num_threads, row_start_idx, row_end_idx, d_tile_ptr_A,d_tile_columnidx,d_tile_nnz,d_tile_csr_ptr, matrixA_d->numtile,d_j_col,n_tile, d_flag_t,d_j_num_t_1,BLOCK_SIZE);
        gettimeofday(&t2, NULL);
        time_cuda_step2 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        
      
     gettimeofday(&t1, NULL);
  
     gettimeofday(&t2, NULL);
     time_cuda_step2 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
       
        gettimeofday(&t1, NULL);
cuda_step2_2_kernel_new<<< num_blocks, 64 >>>(matrixA_d->m, matrixA_d->n, d_rowpointerA, d_columnindexA, matrixA_d->tilem, matrixA_d->tilen, num_threads, row_start_idx, row_end_idx, d_tile_ptr_A,d_tile_columnidx,d_tile_nnz,d_tile_csr_ptr, matrixA_d->numtile,d_j_col,n_tile, d_flag_t,d_j_num_t_1,BLOCK_SIZE);
        
        
        gettimeofday(&t2, NULL);
      
        time_cuda_step2 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    }
    }
    else
    {
   
        for(int i=0;i<matrixA_d->tilem;i+=n_tile)//
        {   
          
            cudaMemset(d_j_col,0,(n_tile+1) * sizeof(int));
          
            cudaMemset(d_flag_t,0,n_tile*matrixA_d->tilen * sizeof(unsigned int)/32+32);
            cudaMemset(d_j_num_t_1,0,n_tile*matrixA_d->tilen * sizeof(int));
        
            int row_start_idx=i*BLOCK_SIZE;
            int row_end_idx=i*BLOCK_SIZE+n_tile*BLOCK_SIZE>matrixA_d->m-1 ? row_end_idx=matrixA_d->m-1 : row_end_idx=i*BLOCK_SIZE+n_tile*BLOCK_SIZE-1;
       
            num_threads= matrixA_d->rowpointer[row_end_idx+1]-matrixA_d->rowpointer[row_start_idx];
            num_blocks=num_threads/64+1;
       
            gettimeofday(&t1, NULL);
        
            cuda_step2_kernel_new_sort_1<<< num_blocks, 64 >>>(matrixA_d->m, matrixA_d->n, d_rowpointerA, d_columnindexA, matrixA_d->tilem, matrixA_d->tilen, num_threads, row_start_idx, row_end_idx, d_tile_ptr_A,d_tile_columnidx,d_tile_nnz,d_tile_csr_ptr, matrixA_d->numtile,d_j_col,n_tile, d_flag_t,d_j_num_t_1,BLOCK_SIZE);
            gettimeofday(&t2, NULL);
            time_cuda_step2 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
     
            gettimeofday(&t1, NULL);
            cuda_step2_2_kernel_new_sort_2<<< num_blocks, 64 >>>(matrixA_d->m, matrixA_d->n, d_rowpointerA, d_columnindexA, matrixA_d->tilem, matrixA_d->tilen, num_threads, row_start_idx, row_end_idx, d_tile_ptr_A,d_tile_columnidx,d_tile_nnz,d_tile_csr_ptr, matrixA_d->numtile,d_j_col,n_tile, d_flag_t,d_j_num_t_1,BLOCK_SIZE);
        
            gettimeofday(&t2, NULL);
            time_cuda_step2 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

        }
        int blki_n=4096;
       gettimeofday(&t1, NULL);
        for(int blki =0;blki < matrixA_d->tilem ;blki +=blki_n)
        {
            int num_blocks=blki_n/64;
           
            int end= blki+blki_n <= matrixA_d->tilem ? end=blki_n : end=matrixA_d->tilem-blki;
            quick_sort_cuda<<< num_blocks, 64 >>>(d_tile_columnidx,d_tile_nnz,d_tile_csr_ptr,end,blki,d_tile_ptr_A,BLOCK_SIZE);     
        }
        gettimeofday(&t2, NULL);
            time_cuda_step2 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    }
    gettimeofday(&t4, NULL);
    cudaMemcpy(matrixA_d->tile_columnidx, d_tile_columnidx, matrixA_d->numtile*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->tile_nnz, d_tile_nnz, matrixA_d->numtile*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(matrixA_d->csr_ptr_1, d_tile_csr_ptr, BLOCK_SIZE*(matrixA_d->numtile + 1)*sizeof(int), cudaMemcpyDeviceToHost);
    

    printf("transform_step2(cuda) runtime    = %4.5f ms\n", time_cuda_step2);
    time_cuda_step2_sum = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
    printf("transform_step2(cuda)_sum runtime    = %4.5f ms\n", time_cuda_step2_sum);
    exclusive_scan(matrixA_d->tile_nnz, matrixA_d->numtile +1);
    
//cuda-step2-end

//cuda-step3
   
    matrixA_d->blknnz = (int *)malloc((matrixA_d->numtile + 1)* sizeof(int));                                               
    memset(matrixA_d->blknnz,0,(matrixA_d->numtile + 1) * sizeof(int));  
    int *d_blknnz;
    cudaMalloc((void **)&d_blknnz, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_blknnz, matrixA_d->blknnz, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

     //CSR
    
    int csrsize_1=0;
  //  int csrptrlen=0;
    matrixA_d->csr_offset = (int *)malloc((matrixA_d->numtile+1) * sizeof(int));
    memset(matrixA_d->csr_offset, 0, (matrixA_d->numtile+1) * sizeof(int)); 
    int *d_csr_offset;
    cudaMalloc((void **)&d_csr_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_csr_offset, matrixA_d->csr_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

    matrixA_d->csrptr_offset = (int *)malloc((matrixA_d->numtile+1) * sizeof(int));
    memset(matrixA_d->csrptr_offset, 0, (matrixA_d->numtile+1) * sizeof(int));
    int *d_csrptr_offset;
    cudaMalloc((void **)&d_csrptr_offset, (matrixA_d->numtile + 1)* sizeof(int) );
    cudaMemcpy(d_csrptr_offset, matrixA_d->csrptr_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);


int num_tile_row=4;
    unsigned char *col_flag =(unsigned char *)malloc(matrixA_d->tilen*num_tile_row*BLOCK_SIZE * sizeof(unsigned char));
    memset(col_flag, 0, matrixA_d->tilen*num_tile_row*BLOCK_SIZE * sizeof(unsigned char));
    unsigned char *d_col_flag;
    cudaMalloc((void **)&d_col_flag, matrixA_d->tilen*num_tile_row*BLOCK_SIZE * sizeof(unsigned char) );
    cudaMemcpy(d_col_flag, col_flag, matrixA_d->tilen*num_tile_row*BLOCK_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);

cudaMemcpy(d_tile_csr_ptr, matrixA_d->csr_ptr_1, BLOCK_SIZE*(matrixA_d->numtile + 1)*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_rowpointerA, matrixA_d->rowpointer, sizeof(MAT_PTR_TYPE) *(matrixA_d->m+1), cudaMemcpyHostToDevice);
cudaMemcpy(d_tile_columnidx, matrixA_d->tile_columnidx, matrixA_d->numtile*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_tile_nnz, matrixA_d->tile_nnz, (matrixA_d->numtile+1)*sizeof(int), cudaMemcpyHostToDevice);

   double cuda_time_step3=0.0;
   double cuda_time_step3_sum=0.0;
   gettimeofday(&t3, NULL);
    for (int blki=0;blki<matrixA_d->tilem;blki+=num_tile_row)//
    {
//int blki=0;
cudaMemset(d_col_flag,0,matrixA_d->tilen*num_tile_row*BLOCK_SIZE * sizeof(unsigned char));
        int start=blki;
        int end= blki+num_tile_row<matrixA_d->tilem ? end= blki+num_tile_row : end=matrixA_d->tilem;
        num_blocks=matrixA_d->tile_ptr[end]-matrixA_d->tile_ptr[start];
      
        gettimeofday(&t1, NULL);
        
        cuda_step3_kernel<<<num_blocks, 32 >>>(matrixA_d->m, matrixA_d->n,d_rowpointerA, d_columnindexA,
                 matrixA_d->tilem, matrixA_d->tilen, matrixA_d->numtile, d_tile_ptr_A, d_tile_columnidx, d_tile_nnz,d_blknnz,
                 d_csr_offset, d_csrptr_offset,blki,num_tile_row
                 ,BLOCK_SIZE); 
        gettimeofday(&t2, NULL);
        cuda_time_step3 += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        
    }
    gettimeofday(&t4, NULL);
    cudaDeviceSynchronize();
    //double cuda_time_step3  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("cuda_transform_step3 runtime    = %4.5f ms\n", cuda_time_step3);
  cuda_time_step3_sum = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
  printf("cuda_transform_step3 runtime_sum    = %4.5f ms\n", cuda_time_step3_sum);

       cudaMemcpy(matrixA_d->blknnz, d_blknnz, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(matrixA_d->csr_offset, d_csr_offset, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(matrixA_d->csrptr_offset, d_csrptr_offset, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyDeviceToHost);


    exclusive_scan(matrixA_d->csr_offset, matrixA_d->numtile +1);
    exclusive_scan(matrixA_d->csrptr_offset, matrixA_d->numtile +1);

    for (int blki=0;blki<matrixA_d->tilem;blki++)
    {
        int rowlength= blki==matrixA_d->tilem-1 ? matrixA_d->m-(matrixA_d->tilem-1)*BLOCK_SIZE : BLOCK_SIZE ;
        int rowbnum=matrixA_d->tile_ptr[blki+1]-matrixA_d->tile_ptr[blki];
        for (int bi=0;bi<rowbnum;bi++)
        {
   //csr
                    csrsize_1 +=  matrixA_d->blknnz[matrixA_d->tile_ptr[blki]+bi];


        }
    }   
    matrixA_d->blknnznnz = (unsigned char *)malloc((matrixA_d->numtile + 1)* sizeof(unsigned char));
   // memcpy(matrixA_d->blknnznnz,matrixA_d->blknnz,matrixA_d->numtile + 1);
    for (int i = 0; i < matrixA_d->numtile+1; i++)
        matrixA_d->blknnznnz[i] = matrixA_d->blknnz[i];
    //exclusive_scan(blknnz,(numtileA+1));
    exclusive_scan(matrixA_d->blknnz,(matrixA_d->numtile+1));
    int csrtilecount_1 = matrixA_d->numtile;


int nnz_temp =0;
    int tile_count_temp =0;
    for (int blki =0;blki < matrixA_d->tilem; blki ++)
    {
        int start= blki*BLOCK_SIZE;
        int end = blki==matrixA_d->tilem-1 ?  matrixA_d->m : (blki+1)*BLOCK_SIZE ;
        nnz_temp = nnz_temp < matrixA_d->rowpointer[end] - matrixA_d->rowpointer[start] ? matrixA_d->rowpointer[end] - matrixA_d->rowpointer[start] : nnz_temp;
        tile_count_temp = tile_count_temp < matrixA_d->tile_ptr[blki +1] - matrixA_d->tile_ptr[blki] ? matrixA_d->tile_ptr[blki +1] - matrixA_d->tile_ptr[blki] : tile_count_temp;
    }

//cuda-step3-end

//cuda-step4

    matrixA_d->csrsize=csrsize_1;

     //CSR
    //MAT_VAL_TYPE *Tile_csr_Val_1=(MAT_VAL_TYPE*)malloc((csrsize_1)*sizeof(MAT_VAL_TYPE));
    matrixA_d->Tile_csr_Val=(MAT_VAL_TYPE*)malloc((csrsize_1)*sizeof(MAT_VAL_TYPE));
    memset(matrixA_d->Tile_csr_Val, 0, (csrsize_1)*sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *d_Tile_csr_Val;
    cudaMalloc((void **)&d_Tile_csr_Val, (csrsize_1)*sizeof(MAT_VAL_TYPE) );
    cudaMemcpy(d_Tile_csr_Val, matrixA_d->Tile_csr_Val, (csrsize_1)*sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);

    //unsigned char  *Tile_csr_Col_1=(unsigned char*)malloc((csrsize_1)*sizeof(unsigned char));
    matrixA_d->Tile_csr_Col=(unsigned char*)malloc((csrsize_1)*sizeof(unsigned char));
    memset(matrixA_d->Tile_csr_Col, 0, (csrsize_1)*sizeof(unsigned char));
    unsigned char  *d_Tile_csr_Col;
    cudaMalloc((void **)&d_Tile_csr_Col,(csrsize_1)*sizeof(unsigned char) );
    cudaMemcpy(d_Tile_csr_Col, matrixA_d->Tile_csr_Col, (csrsize_1)*sizeof(unsigned char), cudaMemcpyHostToDevice);
//printf("csrtilecount=%d  csrtilecount_1=%d\n",csrtilecount,csrtilecount_1);

    //unsigned char *Tile_csr_Ptr_1=(unsigned char*)malloc((csrtilecount_1 * BLOCK_SIZE)*sizeof(unsigned char));
    matrixA_d->Tile_csr_Ptr=(unsigned char*)malloc((csrtilecount_1 * BLOCK_SIZE)*sizeof(unsigned char));
    memset(matrixA_d->Tile_csr_Ptr, 0, (csrtilecount_1 * BLOCK_SIZE )*sizeof(unsigned char));
    unsigned char *d_Tile_csr_Ptr;
    cudaMalloc((void **)&d_Tile_csr_Ptr,(csrtilecount_1 * BLOCK_SIZE)*sizeof(unsigned char) );
    cudaMemcpy(d_Tile_csr_Ptr, matrixA_d->Tile_csr_Ptr,(csrtilecount_1 * BLOCK_SIZE)*sizeof(unsigned char), cudaMemcpyHostToDevice);

    num_tile_row=4;
   // unsigned thread;
	unsigned char  *csr_colidx_temp_g_1=(unsigned char*)malloc((num_tile_row * nnz_temp )*sizeof(unsigned char));
    unsigned char  *d_csr_colidx_temp_g;
    cudaMalloc((void **)&d_csr_colidx_temp_g, (num_tile_row * nnz_temp )*sizeof(unsigned char));

    MAT_VAL_TYPE *csr_val_temp_g_1=(MAT_VAL_TYPE*)malloc((num_tile_row * nnz_temp)*sizeof(MAT_VAL_TYPE));
    MAT_VAL_TYPE *d_csr_val_temp_g;
    cudaMalloc((void **)&d_csr_val_temp_g, (num_tile_row * nnz_temp)*sizeof(MAT_VAL_TYPE));

    int *tile_count_g_1 = (int *)malloc(num_tile_row * tile_count_temp * sizeof(int));
    int *d_tile_count_g;
    cudaMalloc((void **)&d_tile_count_g, num_tile_row * tile_count_temp * sizeof(int));

    MAT_VAL_TYPE *d_value;
    cudaMalloc((void **)&d_value, (matrixA_d->nnz+1) * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_value, matrixA_d->value, (matrixA_d->nnz+1) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
//printf("csrptr_offset_1[%d]=%d\n",matrixA_d->numtile,matrixA_d->csrptr_offset[matrixA_d->numtile]);
	cudaMemcpy(d_csrptr_offset, matrixA_d->csrptr_offset, (matrixA_d->numtile+1)* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tile_csr_ptr, matrixA_d->csr_ptr_1, ((matrixA_d->numtile + 1) * BLOCK_SIZE) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tile_nnz, matrixA_d->tile_nnz, (matrixA_d->numtile+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tile_columnidx, matrixA_d->tile_columnidx, (matrixA_d->numtile+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tile_ptr_A, matrixA_d->tile_ptr, sizeof(MAT_PTR_TYPE) *(matrixA_d->tilem+1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_csr_offset, matrixA_d->csr_offset, (matrixA_d->numtile + 1)* sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_csr_colidx_temp_g, 0, (num_tile_row * nnz_temp )*sizeof(unsigned char));
    cudaMemset(d_csr_val_temp_g, 0, (num_tile_row * nnz_temp)*sizeof(MAT_VAL_TYPE));
    cudaMemset(d_tile_count_g, 0, num_tile_row * tile_count_temp * sizeof(int));

   // int *mask = matrixA_d->mask;
   unsigned long long int  *d_mask_64;
   unsigned int  *d_mask_32;
   if(matrixA_d->blocksize_flag==1)
   { 
       matrixA_d->mask_64 = (unsigned long long int   *)malloc(matrixA_d->numtile * BLOCK_SIZE * sizeof(unsigned long long int ));
       memset(matrixA_d->mask_64, 0, matrixA_d->numtile * BLOCK_SIZE * sizeof(unsigned long long int ));
    
       cudaMalloc((void **)&d_mask_64, matrixA_d->numtile * BLOCK_SIZE * sizeof(unsigned long long int ));
       cudaMemcpy(d_mask_64, matrixA_d->mask_64, matrixA_d->numtile * BLOCK_SIZE * sizeof(unsigned long long int ), cudaMemcpyHostToDevice);
    }
    else
    {
    
        matrixA_d->mask_32 = (unsigned int   *)malloc(matrixA_d->numtile * BLOCK_SIZE * sizeof(unsigned int ));
        memset(matrixA_d->mask_32, 0, matrixA_d->numtile * BLOCK_SIZE * sizeof(unsigned int ));
    
        cudaMalloc((void **)&d_mask_32, matrixA_d->numtile * BLOCK_SIZE * sizeof(unsigned int ));
        cudaMemcpy(d_mask_32, matrixA_d->mask_32, matrixA_d->numtile * BLOCK_SIZE * sizeof(unsigned int ), cudaMemcpyHostToDevice);
    }

     double cuda_time_step4=0.0;
     double cuda_time_step4_sum=0.0;
     gettimeofday(&t3, NULL);
    for (int blki=0;blki<matrixA_d->tilem;blki+=num_tile_row)//
    {
        //cudaMemset(d_col_flag,0,tilenA*num_tile_row*BLOCK_SIZE * sizeof(unsigned char));
        int start=blki;
        int end= blki+num_tile_row<matrixA_d->tilem ? end= blki+num_tile_row : end=matrixA_d->tilem;
        num_blocks=matrixA_d->tile_ptr[end]-matrixA_d->tile_ptr[start];
       
        gettimeofday(&t1, NULL);
        cudaMemset(d_tile_count_g, 0, num_tile_row * tile_count_temp * sizeof(int));
        
     cuda_step4_kernel_1<<<end-start, 32 >>>(matrixA_d->m, matrixA_d->n,d_rowpointerA, d_columnindexA, d_value,
                 matrixA_d->tilem, matrixA_d->tilen, matrixA_d->numtile, d_tile_ptr_A, d_tile_columnidx, d_tile_nnz, 
                 d_blknnz, d_tile_csr_ptr, nnz_temp, tile_count_temp,
                 d_csr_colidx_temp_g,d_csr_val_temp_g,d_tile_count_g,blki,num_tile_row,BLOCK_SIZE);
        gettimeofday(&t2, NULL);
        cuda_time_step4  += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
       // cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        if(matrixA_d->blocksize_flag==1)
        cuda_step4_kernel_2_64<<<num_blocks, 32 >>>(matrixA_d->m, matrixA_d->n,d_rowpointerA, d_columnindexA, d_value,
                 matrixA_d->tilem, matrixA_d->tilen, matrixA_d->numtile, d_tile_ptr_A, d_tile_columnidx, d_tile_nnz,  
                 d_blknnz, d_tile_csr_ptr, nnz_temp, tile_count_temp,
                 d_csr_colidx_temp_g,d_csr_val_temp_g,d_tile_count_g,
                 d_Tile_csr_Val, d_Tile_csr_Col, d_Tile_csr_Ptr, d_csr_offset, d_csrptr_offset,
              /*   d_Tile_coo_Val, d_Tile_coo_colIdx, d_Tile_coo_rowIdx, d_coo_offset,
                 d_Tile_ell_Val, d_Tile_ell_colIdx, d_blkwidth, d_ell_offset,
                 d_Tile_hyb_Val, d_Tile_hyb_ellcolIdx, d_Tile_hyb_coorowIdx,  d_hyb_coocount, d_hyb_offset,
                 d_Tile_dns_Val, d_dns_offset,
                 d_Tile_dnsrow_Val, d_Tile_dnsrow_idx, d_denserowptr, d_dnsrow_offset,
                 d_Tile_dnscol_Val, d_Tile_dnscol_idx,  d_densecolptr, d_dnscol_offset,
                 d_new_coocount,d_new_coo_value, d_new_coo_colidx, d_new_coo_rowidx,*/
                 blki,num_tile_row,
                 d_mask_64,BLOCK_SIZE); 
         else
         cuda_step4_kernel_2_32<<<num_blocks, 32 >>>(matrixA_d->m, matrixA_d->n,d_rowpointerA, d_columnindexA, d_value,
                 matrixA_d->tilem, matrixA_d->tilen, matrixA_d->numtile, d_tile_ptr_A, d_tile_columnidx, d_tile_nnz,  
                 d_blknnz, d_tile_csr_ptr, nnz_temp, tile_count_temp,
                 d_csr_colidx_temp_g,d_csr_val_temp_g,d_tile_count_g,
                 d_Tile_csr_Val, d_Tile_csr_Col, d_Tile_csr_Ptr, d_csr_offset, d_csrptr_offset,
                 blki,num_tile_row,
                 d_mask_32,BLOCK_SIZE); 

gettimeofday(&t2, NULL);
  cuda_time_step4  += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
  

    }
    gettimeofday(&t4, NULL);

    printf("cuda_transform_step4 runtime    = %4.5f ms\n", cuda_time_step4);
     cuda_time_step4_sum = (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
     printf("cuda_transform_step4_sum runtime    = %4.5f ms\n", cuda_time_step4_sum);

     if(matrixA_d->blocksize_flag==1)
     cudaMemcpy(matrixA_d->mask_64, d_mask_64, matrixA_d->numtile * BLOCK_SIZE * sizeof(unsigned long long int ), cudaMemcpyDeviceToHost);
     else
     cudaMemcpy(matrixA_d->mask_32, d_mask_32, matrixA_d->numtile * BLOCK_SIZE * sizeof(unsigned int ), cudaMemcpyDeviceToHost);


double time_sum=0.0;
time_sum=time_cuda_step1_sum+time_cuda_step2_sum+cuda_time_step3_sum+cuda_time_step4_sum;
printf("cuda data preprocessing time_sum = %f ms\n",time_sum);
printf("---------------------data preprocessing end--------------------\n");

    cudaFree(d_rowpointerA);
    cudaFree(d_columnindexA);
    cudaFree(d_tile_ptr_A);
    cudaFree(d_flag_t);
    cudaFree(d_tile_columnidx);
    cudaFree(d_tile_nnz);
    cudaFree(d_tile_csr_ptr);
    cudaFree(d_j_col);
    cudaFree(d_j_num_t_1);
    cudaFree(d_blknnz);
    cudaFree(d_csr_offset);
    cudaFree(d_csrptr_offset);
    cudaFree(d_Tile_csr_Val);
    cudaFree(d_Tile_csr_Col);
    cudaFree(d_Tile_csr_Ptr);
    cudaFree(d_csr_colidx_temp_g);
    cudaFree(d_csr_val_temp_g);
    cudaFree(d_tile_count_g);
    if(matrixA_d->blocksize_flag==1)
    cudaFree(d_mask_64);
    else
    cudaFree(d_mask_32);

}

void Tile_destroy(Beidou_Tile_Matrix *matrix)
{
    free(matrix->Tile_csr_Col);
    matrix->Tile_csr_Col = NULL;
    free(matrix->Tile_csr_Ptr);
    matrix->Tile_csr_Ptr = NULL;
    free(matrix->Tile_csr_Val);
    matrix->Tile_csr_Val = NULL;
    free(matrix->Tile_coo_colIdx);
    matrix->Tile_coo_colIdx = NULL;
    free(matrix->Tile_coo_rowIdx);
    matrix->Tile_coo_rowIdx = NULL;
    free(matrix->Tile_coo_Val);
    matrix->Tile_coo_Val = NULL;
    free(matrix->Tile_ell_colIdx);
    matrix->Tile_ell_colIdx = NULL;
    free(matrix->Tile_ell_Val);
    matrix->Tile_ell_Val = NULL;
    free(matrix->Tile_hyb_coorowIdx);
    matrix->Tile_hyb_coorowIdx = NULL;
    free(matrix->Tile_hyb_ellcolIdx);
    matrix->Tile_hyb_ellcolIdx = NULL;
    free(matrix->Tile_hyb_Val);
    matrix->Tile_hyb_Val = NULL;
    free(matrix->Tile_dns_Val);
    matrix->Tile_dns_Val = NULL;
    free(matrix->Tile_dnsrow_idx);
    matrix->Tile_dnsrow_idx = NULL;
    free(matrix->Tile_dnsrow_Val);
    matrix->Tile_dnsrow_Val = NULL;
    free(matrix->Tile_dnscol_Val);
    matrix->Tile_dnscol_Val = NULL;
    free(matrix->Tile_dnscol_idx);
    matrix->Tile_dnscol_idx = NULL;
    free(matrix->densecolptr);
    matrix->densecolptr = NULL;
    free(matrix->denserowptr);
    matrix->denserowptr = NULL;
    free(matrix->blkwidth);
    matrix->blkwidth = NULL;
    free(matrix->tile_ptr);
    matrix->tile_ptr = NULL;
    free(matrix->tile_columnidx);
    matrix->tile_columnidx = NULL;
    free(matrix->tile_nnz);
    matrix->tile_nnz = NULL;

    free(matrix->blknnz);
    matrix->blknnz = NULL;
    free(matrix->value);
    matrix->value = NULL;
    free(matrix->columnidx);
    matrix->columnidx = NULL;
    free(matrix->coo_new_matrix_ptr);
    matrix->coo_new_matrix_ptr = NULL;
    free(matrix->coo_new_rowidx);
    matrix->coo_new_rowidx = NULL;
    free(matrix->coo_new_matrix_value);
    matrix->coo_new_matrix_value = NULL;
    free(matrix->coo_new_matrix_colidx);
    matrix->coo_new_matrix_colidx = NULL;

    free(matrix->csr_ptr);
    free(matrix->csr_offset);
    free(matrix->csrptr_offset);
    free(matrix->coo_offset);
    free(matrix->ell_offset);
    free(matrix->hyb_offset);
    free(matrix->dns_offset);
    free(matrix->dnsrow_offset);
    free(matrix->dnscol_offset);
    if(matrix->blocksize_flag==1)
    free(matrix->mask_64);
    else
    free(matrix->mask_32);
    
    matrix->mask_64 = NULL;

    
    matrix->mask_32 = NULL;
    
    free(matrix->blkcoostylerowidx);
    matrix->blkcoostylerowidx = NULL;
    free(matrix->blkcoostylerowidx_colstart);
    matrix->blkcoostylerowidx_colstart = NULL;
    free(matrix->blkcoostylerowidx_colstop);
    matrix->blkcoostylerowidx_colstop = NULL;


}




