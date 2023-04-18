
#include "common.h"
void balance(Beidou_Tile_Matrix *matrixA_d,int bfs_source)
{
    matrixA_d->rowblkblock = 0;
    //int iiiii = 0;
    for (int blki = 0; blki < matrixA_d->tilem; blki++)
    {
        int balancenumblk = matrixA_d->tile_ptr[blki+1] - matrixA_d->tile_ptr[blki];
        if (balancenumblk <= PREFETCH_SMEM_TH) 
            matrixA_d->rowblkblock++;
        else 
        {
            matrixA_d->rowblkblock += ceil((double)balancenumblk / (double)PREFETCH_SMEM_TH);
            //printf("[%i] blki = %i, balancenumblk = %i, rowblkblock += %i\n", iiiii, blki, balancenumblk, balancenumblk / 32); 
            //iiiii++;
        }

    }
    printf("ave blk num = %4.2f, %i, %i\n", (double)matrixA_d->tile_ptr[matrixA_d->tilem] / (double)matrixA_d->tilem, matrixA_d->tilem, matrixA_d->rowblkblock);

    matrixA_d->blkcoostylerowidx = (unsigned int *)malloc(sizeof(unsigned int) * matrixA_d->rowblkblock);
    memset(matrixA_d->blkcoostylerowidx, 0, sizeof(unsigned int) * matrixA_d->rowblkblock);
    matrixA_d->blkcoostylerowidx_colstart = (int *)malloc(sizeof(int) * matrixA_d->rowblkblock);
    memset(matrixA_d->blkcoostylerowidx_colstart, 0, sizeof(int) * matrixA_d->rowblkblock);
    matrixA_d->blkcoostylerowidx_colstop = (int *)malloc(sizeof(int) * matrixA_d->rowblkblock);
    memset(matrixA_d->blkcoostylerowidx_colstop, 0, sizeof(int) * matrixA_d->rowblkblock);
    matrixA_d->blk_colid=(int *)malloc(sizeof(int) * (matrixA_d->tilem+1));
    memset(matrixA_d->blk_colid, 0, sizeof(int) * (matrixA_d->tilem+1));
    
    int rowblkblockcnt = 0;
    for (int blki = 0; blki < matrixA_d->tilem; blki++)
    {
        int balancenumblk = matrixA_d->tile_ptr[blki+1] - matrixA_d->tile_ptr[blki];
//printf("blki=%d  balancenumblk=%d\n",blki,balancenumblk);
        if (balancenumblk <= PREFETCH_SMEM_TH) 
        { // || (bfs_source/32==blki && matrixA_d->blocksize_flag==1) || (bfs_source/64==blki && matrixA_d->blocksize_flag==0)
            matrixA_d->blkcoostylerowidx[rowblkblockcnt] = blki;
            matrixA_d->blk_colid[blki]=rowblkblockcnt;
            rowblkblockcnt++;
        }
        else 
        {
            int numblklocal = ceil((double)balancenumblk / (double)PREFETCH_SMEM_TH);
            int lenblklocal = ceil((double)balancenumblk / (double)numblklocal);
            matrixA_d->blk_colid[blki]=rowblkblockcnt;
            for (int iii = 0; iii < numblklocal; iii++)
            {
                matrixA_d->blkcoostylerowidx[rowblkblockcnt] = blki | 0x80000000; // can generate -0
                matrixA_d->blkcoostylerowidx_colstart[rowblkblockcnt] = matrixA_d->tile_ptr[blki] + iii * lenblklocal;
                if (iii == numblklocal - 1)
                    matrixA_d->blkcoostylerowidx_colstop[rowblkblockcnt] = matrixA_d->tile_ptr[blki] + balancenumblk;
                else 
                    matrixA_d->blkcoostylerowidx_colstop[rowblkblockcnt] = matrixA_d->tile_ptr[blki] + (iii+1) * lenblklocal;

                rowblkblockcnt++;
            }
        }
      //  printf("matrixA_d->blk_colid[blki]=%d\n",matrixA_d->blk_colid[blki]);
    }
matrixA_d->blk_colid[matrixA_d->tilem]=rowblkblockcnt;

}

void densetosparse_vector(int *csrColIdxx,MAT_VAL_TYPE *csrValx,int *csrptrx,MAT_VAL_TYPE *x,int m,int use_numeric,int tile_m,int BLOCK_SIZE)
{
    int num=0;
    for(int i=0;i<m;i++)
    {
        if(x[i]!=0)
        {
            csrColIdxx[num]=i%16;
           // if(use_numeric) 
                csrValx[num]=x[i]; //x[i]
            csrptrx[i/BLOCK_SIZE]++;
            
            
         //   printf("num=%d  csrColIdxx[num]=%d  csrValx[num]=%f\n",num,csrColIdxx[num],csrValx[num]);
            num++;
        }
    }
    exclusive_scan(csrptrx,tile_m+1);
}

int MergeArr(int* a, int alen, int* b, int blen, int* c)
{
	int i = 0;
	int j = 0;
	int k = 0;
	
	while (i != alen && j != blen)
	{
		if (a[i] < b[j])
			c[k++] = a[i++];
		else if(a[i] == b[j])
		{
		    c[k++] = b[j++];
		    i++;
		}
		else
			c[k++] = b[j++];	
	}
	if (i == alen)
	{
		while (j != blen)
		c[k++] = b[j++];
	}
	else
	{
		while (i != alen)
			c[k++] = a[i++];
	}
	return k;
}

__inline__ __device__
int cu_count_1(unsigned long long int value,int block_size)
{
        int count = 0;
        unsigned long long int x[64]={0};
        for (int i = 0; i < block_size; i++){
       
        x[i]=value & 0x01;
            if ((value & 0x01) == 0x01){
                count++;
            }
            value = value >> 1;  //  /2
        }
        return count;
}

int count_1(unsigned long long int value,int BLOCK_SIZE)
{
        int count = 0;
        unsigned long long int x[BLOCK_SIZE]={0};
        for (int i = 0; i < BLOCK_SIZE; i++){
       
        x[i]=value & 0x01;
            if ((value & 0x01) == 0x01){
                count++;
            }
            value = value >> 1;  //  /2
        }
        return count;
}

int count_1_x(unsigned long long int value,int BLOCK_SIZE)
{
        int count = 0;
        unsigned long long int x[BLOCK_SIZE]={0};
        for (int i = 0; i < BLOCK_SIZE; i++){
       
        x[i]=value & 0x01;
            if ((value & 0x01) == 0x01){
                count++;
            }
            value = value >> 1;  //  /2
        }

        return count;
}
void swap_key_tile(int *a , int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}
/*
void swap_key_tile_arr(int *a , int *b)
{
    for(int i=0;i<16;i++)
    {
        int tmp = *a+i;
        *a+i = *b+i;
        *b+i = tmp;
    }

}*/

// quick sort key (child function)
int partition_key_tile(int *key, int length, int pivot_index , int *nnz, int *csr_ptr)
{
    int i  = 0 ;
    int small_length = pivot_index;

    int pivot = key[pivot_index];
    swap_key_tile(&key[pivot_index], &key[pivot_index + (length - 1)]);
    swap_key_tile(&nnz[pivot_index], &nnz[pivot_index + (length - 1)]);
    for(int k=0;k<16;k++) 
    {
        swap_key_tile(&csr_ptr[pivot_index*16+k], &csr_ptr[(pivot_index + (length - 1))*16+k]);
    }
    for(; i < length; i++)
    {
        if(key[pivot_index+i] < pivot)
        {
            swap_key_tile(&key[pivot_index+i], &key[small_length]);
            swap_key_tile(&nnz[pivot_index+i], &nnz[small_length]);
    //        swap_key_tile_arr(&csr_ptr[(pivot_index+i)*16], &csr_ptr[16*small_length]);
            for(int k=0;k<16;k++) 
            {
                swap_key_tile(&csr_ptr[(pivot_index+i)*16+k], &csr_ptr[16*small_length+k]);
            }
            small_length++;
        }
    }

    swap_key_tile(&key[pivot_index + length - 1], &key[small_length]);
    swap_key_tile(&nnz[pivot_index + length - 1], &nnz[small_length]);
  //  swap_key_tile_arr(&csr_ptr[(pivot_index + length - 1)*16], &csr_ptr[16*small_length]);
    for(int k=0;k<16;k++) 
    {
        swap_key_tile(&csr_ptr[(pivot_index + length - 1)*16+k], &csr_ptr[16*small_length+k]);
    }
    return small_length;
}

// quick sort key (main function)
void quick_sort_key_tile(int *key, int length, int *nnz, int *csr_ptr)
{
    if(length == 0 || length == 1)
        return;

    int small_length = partition_key_tile(key, length, 0, nnz, csr_ptr) ;
    quick_sort_key_tile(key, small_length, nnz, csr_ptr);
    quick_sort_key_tile(&key[small_length + 1], length - small_length - 1, &nnz[small_length + 1], &csr_ptr[(small_length + 1)*16]);
    //printf("sort: ")
}

/*
void exclusive_scan_char(unsigned char *input, int length)
{
    if(length == 0 || length == 1)
        return;
    
    unsigned char old_val, new_val;
    
    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i-1];
        old_val = new_val;
    }
}*/

