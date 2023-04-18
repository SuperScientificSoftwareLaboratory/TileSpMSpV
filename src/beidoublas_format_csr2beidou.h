#ifndef _BEIDOUBLAS_FORMAT_CSR2BEIDOU_
#define _BEIDOUBLAS_FORMAT_CSR2BEIDOU_
#include <iostream>
#include <cmath>


void write_Smatrix(int m, int n, int *rowptr, int *colidx, double *val, char *string){

    FILE *fp=fopen(string,"w+");

    if(fp==NULL){
        printf("eeror the file don't open\n");
        return ;
    }
    int column=n;
    int row=m;
    fprintf(fp,"%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp,"%d ",row);
    fprintf(fp,"%d ",column);
    fprintf(fp,"%d\n",rowptr[row]);

    for(int i=0;i<row;i++){
        for(int j=rowptr[i];j<rowptr[i+1];j++){
            fprintf(fp,"%d %d %f\n",colidx[j]+1,i+1,val[j]);
        }
    }

    fclose(fp);
}

void step1_kernel(Beidou_Tile_Matrix *matrix)

{
    int *rowpointer = matrix->rowpointer;
    int m = matrix->m;
    int *columnidx = matrix->columnidx;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    
    // unsigned thread = omp_get_max_threads();
    unsigned thread = matrix->nthreads;
    
    char *flag_g = (char *)malloc(thread * tilen * sizeof(char));

    int BLOCK_SIZE = matrix->TILE_SIZE;

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int thread_id = omp_get_thread_num();
        char *flag = flag_g + thread_id * tilen;
        memset(flag, 0, tilen * sizeof(char));
        int start = blki * BLOCK_SIZE;
        int end = blki == tilem - 1 ? m : (blki + 1) * BLOCK_SIZE;
        for (int j = rowpointer[start]; j < rowpointer[end]; j++)
        {
            int jc = columnidx[j] / BLOCK_SIZE;
            if (flag[jc] == 0)
            {
                flag[jc] = 1;
                tile_ptr[blki]++;
            }
        }
    }
    free(flag_g);
}

void step2_kernel_new(Beidou_Tile_Matrix *matrix, int *tile_csr_ptr)

{
    int m = matrix->m;
    int *rowpointer = matrix->rowpointer;
    int *columnidx = matrix->columnidx;

    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
  //  int *tile_rowidx = matrix->tile_rowidx;
    int *tile_nnz = matrix->tile_nnz;
    int BLOCK_SIZE = matrix->TILE_SIZE;

    // unsigned thread = omp_get_max_threads();
    unsigned thread = matrix->nthreads;
    
    char *col_temp_g = (char *)malloc((thread * tilen) * sizeof(char));

    int *nnz_temp_g = (int *)malloc((thread * tilen) * sizeof(int));

    unsigned char *ptr_per_tile_g = (unsigned char *)malloc((thread * tilen * BLOCK_SIZE) * sizeof(unsigned char));

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int thread_id = omp_get_thread_num();
        char *col_temp = col_temp_g + thread_id * tilen;
        memset(col_temp, 0, tilen * sizeof(char));
        int *nnz_temp = nnz_temp_g + thread_id * tilen;
        memset(nnz_temp, 0, tilen * sizeof(int));
        unsigned char *ptr_per_tile = ptr_per_tile_g + thread_id * tilen * BLOCK_SIZE;
        memset(ptr_per_tile, 0, tilen * BLOCK_SIZE * sizeof(unsigned char));
        int pre_tile = tile_ptr[blki];
        int rowlen = blki == tilem - 1 ? m - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        int start = blki * BLOCK_SIZE;

        for (int ri = 0; ri < rowlen; ri++)
        {
            for (int j = rowpointer[start + ri]; j < rowpointer[start + ri + 1]; j++)
            {
                int jc = columnidx[j] / BLOCK_SIZE;
                col_temp[jc] = 1;
                nnz_temp[jc]++;
                ptr_per_tile[jc * BLOCK_SIZE + ri]++;
            }
        }

        int count = 0;
        for (int blkj = 0; blkj < tilen; blkj++)
        {
            if (col_temp[blkj] == 1)
            {
                tile_columnidx[pre_tile + count] = blkj;
               // tile_rowidx[pre_tile + count] = blki;
                tile_nnz[pre_tile + count] = nnz_temp[blkj];
                for (int ri = 0; ri < rowlen; ri++)
                {
                    tile_csr_ptr[(pre_tile + count) * BLOCK_SIZE + ri] = ptr_per_tile[blkj * BLOCK_SIZE + ri];
                }
                count++;
            }
        }
    }
    free(col_temp_g);
    free(nnz_temp_g);
    free(ptr_per_tile_g);
}

void step3_kernel_new(Beidou_Tile_Matrix *matrix)
{
    int *rowpointer = matrix->rowpointer;
    int *columnidx = matrix->columnidx;
    int m = matrix->m;
    int n = matrix->n;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    char *Format = matrix->Format;
    int *blknnz = matrix->blknnz;
   
    int *tile_csr_ptr = matrix->csr_ptr;
    int *csr_offset = matrix->csr_offset;
    int *csrptr_offset = matrix->csrptr_offset;
  /*  int *coo_offset = matrix->coo_offset;
    int *dns_offset = matrix->dns_offset;
    int *dnsrow_offset = matrix->dnsrow_offset;
    int *dnscol_offset = matrix->dnscol_offset;*/
    int BLOCK_SIZE = matrix->TILE_SIZE;

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int tilenum_per_row = tile_ptr[blki + 1] - tile_ptr[blki];
        int rowlen = blki == tilem - 1 ? m - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        for (int bi = 0; bi < tilenum_per_row; bi++)
        {
            int collen = tile_columnidx[tile_ptr[blki] + bi] == tilen - 1 ? n - (tilen - 1) * BLOCK_SIZE : BLOCK_SIZE;
            int tile_id = tile_ptr[blki] + bi;
            int tilennz = tile_nnz[tile_id + 1] - tile_nnz[tile_id];
           

Format[tile_id] = 0;
                blknnz[tile_id] = tilennz;
                csr_offset[tile_id] = tilennz;
                csrptr_offset[tile_id] = BLOCK_SIZE;
                continue;


        }
    }
}

void step4_kernel(Beidou_Tile_Matrix *matrix, int *csr_ptr, int *hyb_coocount, int nnz_temp, int tile_count_temp,
                  int *csr_offset, int *csrptr_offset//, int *coo_offset, int *ell_offset, int *hyb_offset, int *dns_offset, int *dnsrow_offset, int *dnscol_offset,
               //   MAT_VAL_TYPE *new_coo_value, int *new_coo_colidx, int *new_coo_rowidx, int *new_coocount
               )
{
    int *rowpointer = matrix->rowpointer;
    int *columnidx = matrix->columnidx;
    MAT_VAL_TYPE *value = matrix->value;
    int m = matrix->m;
    int n = matrix->n;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    char *Format = matrix->Format;
    char *blkwidth = matrix->blkwidth;
    MAT_VAL_TYPE *Tile_csr_Val = matrix->Tile_csr_Val;
    unsigned char *Tile_csr_Col = matrix->Tile_csr_Col;
    unsigned char *Tile_csr_Ptr = matrix->Tile_csr_Ptr;
  
    unsigned long long int *mask_64 = matrix->mask_64;
    unsigned int *mask_32 = matrix->mask_32;
    int BLOCK_SIZE = matrix->TILE_SIZE;

    // unsigned thread = omp_get_max_threads();
    unsigned thread = matrix->nthreads;

    unsigned char *csr_colidx_temp_g = (unsigned char *)malloc((thread * nnz_temp) * sizeof(unsigned char));
    MAT_VAL_TYPE *csr_val_temp_g = (MAT_VAL_TYPE *)malloc((thread * nnz_temp) * sizeof(MAT_VAL_TYPE));
    int *tile_count_g = (int *)malloc(thread * tile_count_temp * sizeof(int));

    #pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {

        int thread_id = omp_get_thread_num();
        unsigned char *csr_colidx_temp = csr_colidx_temp_g + thread_id * nnz_temp;
        MAT_VAL_TYPE *csr_val_temp = csr_val_temp_g + thread_id * nnz_temp;
        int *tile_count = tile_count_g + thread_id * tile_count_temp;
        memset(csr_colidx_temp, 0, (nnz_temp) * sizeof(unsigned char));
        memset(csr_val_temp, 0, (nnz_temp) * sizeof(MAT_VAL_TYPE));
        memset(tile_count, 0, (tile_count_temp) * sizeof(int));
        int tilenum_per_row = tile_ptr[blki + 1] - tile_ptr[blki];
        int rowlen = blki == tilem - 1 ? m - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        int start = blki * BLOCK_SIZE;
        int end = blki == tilem - 1 ? m : (blki + 1) * BLOCK_SIZE;

        for (int blkj = rowpointer[start]; blkj < rowpointer[end]; blkj++)
        {
            int jc_temp = columnidx[blkj] / BLOCK_SIZE;
            for (int bi = 0; bi < tilenum_per_row; bi++)
            {
                int tile_id = tile_ptr[blki] + bi;
                int jc = tile_columnidx[tile_id];
                int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr[blki]];
                if (jc == jc_temp)
                {
                    csr_val_temp[pre_nnz + tile_count[bi]] = value[blkj];
                    csr_colidx_temp[pre_nnz + tile_count[bi]] = columnidx[blkj] - jc * BLOCK_SIZE;
                    tile_count[bi]++;
                    break;
                }
            }
        }
        for (int bi = 0; bi < tilenum_per_row; bi++)
        {
            int tile_id = tile_ptr[blki] + bi;
            int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr[blki]];
            int tilennz = tile_nnz[tile_id + 1] - tile_nnz[tile_id];
            int collen = tile_columnidx[tile_id] == tilen - 1 ? n - (tilen - 1) * BLOCK_SIZE : BLOCK_SIZE;
            int format = Format[tile_id];
            switch (format)
            {
            case 0:
            {
                int offset = csr_offset[tile_id];
                int ptr_offset = csrptr_offset[tile_id];

                int *ptr_temp = csr_ptr + tile_id * BLOCK_SIZE;
                exclusive_scan(ptr_temp, BLOCK_SIZE);

                for (int ri = 0; ri < BLOCK_SIZE; ri++)
                {
                    int start = ptr_temp[ri];
                    int stop = ri == BLOCK_SIZE - 1 ? tilennz : ptr_temp[ri + 1];
                    ;
                    for (int k = start; k < stop; k++)
                    {
                        unsigned char colidx = csr_colidx_temp[pre_nnz + k];
                  //      Tile_csr_Val[offset + k] = csr_val_temp[pre_nnz + k];
                  //      Tile_csr_Col[offset + k] = csr_colidx_temp[pre_nnz + k];
                        if(matrix->blocksize_flag!=1)
                        mask_32[tile_id * BLOCK_SIZE + ri] |= (((unsigned  int )0x1) << (BLOCK_SIZE - colidx - 1));
                        else
                        mask_64[tile_id * BLOCK_SIZE + ri] |= (((unsigned long long int )0x1) << (BLOCK_SIZE - colidx - 1));
                    }
                  //  Tile_csr_Ptr[ptr_offset + ri] = ptr_temp[ri];
                }
                break;
            }

          
            default:
                break;
            }
        }
    }
    free(csr_colidx_temp_g);
    free(csr_val_temp_g);
    free(tile_count_g);
}


int binary_search_right_boundary_kernel_LBLT(const int *row_pointer,
                                             const int key_input,
                                             const int size)
{
    int start = 0;
    int stop = size - 1;
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


void format_transform(Beidou_Tile_Matrix *matrix)
{
   
    matrix->nthreads = omp_get_max_threads();
    int nthreads = matrix->nthreads;
    
if(matrix->blocksize_flag!=1)
    matrix->TILE_SIZE = 32;
else
    matrix->TILE_SIZE = 64;
    int BLOCK_SIZE = matrix->TILE_SIZE;

    step1_kernel(matrix);
    exclusive_scan(matrix->tile_ptr, matrix->tilem + 1);

    matrix->numtile = matrix->tile_ptr[matrix->tilem];
    printf("matrix->numtile=%d\n",matrix->numtile);
    matrix->tile_columnidx = (int *)malloc(matrix->numtile * sizeof(int));
    memset(matrix->tile_columnidx, 0, matrix->numtile * sizeof(int));

    matrix->tile_nnz = (int *)malloc((matrix->numtile + 1) * sizeof(int));
    memset(matrix->tile_nnz, 0, (matrix->numtile + 1) * sizeof(int));
    
    
    matrix->csr_ptr = (int *)malloc((matrix->numtile * BLOCK_SIZE) * sizeof(int));
    memset(matrix->csr_ptr, 0, (matrix->numtile * BLOCK_SIZE) * sizeof(int));

    step2_kernel_new(matrix, matrix->csr_ptr);

    exclusive_scan(matrix->tile_nnz, matrix->numtile + 1);

    matrix->Format = (char *)malloc(matrix->numtile * sizeof(char));
    memset(matrix->Format, 0, matrix->numtile * sizeof(char));

    matrix->blknnz = (int *)malloc((matrix->numtile + 1) * sizeof(int));
    memset(matrix->blknnz, 0, (matrix->numtile + 1) * sizeof(int));

   
    int csrsize = 0;
    matrix->csr_offset = (int *)malloc((matrix->numtile + 1) * sizeof(int));
    memset(matrix->csr_offset, 0, (matrix->numtile + 1) * sizeof(int));
    matrix->csrptr_offset = (int *)malloc((matrix->numtile + 1) * sizeof(int));
    memset(matrix->csrptr_offset, 0, (matrix->numtile + 1) * sizeof(int));
 
    step3_kernel_new(matrix);

    exclusive_scan(matrix->csr_offset, matrix->numtile + 1);
    exclusive_scan(matrix->csrptr_offset, matrix->numtile + 1);
 
    for (int blki = 0; blki < matrix->tilem; blki++)
    {
        int rowlength = blki == matrix->tilem - 1 ? matrix->m - (matrix->tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        int rowbnum = matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki];
        for (int bi = 0; bi < rowbnum; bi++)
        {
            char format = matrix->Format[matrix->tile_ptr[blki] + bi];
            switch (format)
            {
            case 0:
                csrsize += matrix->blknnz[matrix->tile_ptr[blki] + bi];
                break;

   
            default:
                break;
            }
        }
    }
    
   
    exclusive_scan(matrix->blknnz, (matrix->numtile + 1));

    int *formatnum = (int *)malloc(7 * sizeof(int));
    memset(formatnum, 0, 7 * sizeof(int));

    for (int j = 0; j < 7; j++)
    {
        for (int i = 0; i < matrix->numtile; i++)
        {
            if (matrix->Format[i] == j)
            {
                formatnum[j]++;
            }
        }
    }

    int csrtilecount = formatnum[0];
    //printf("csrtilecount=%d\n",csrtilecount);
    int nnz_temp = 0;
    int tile_count_temp = 0;
    for (int blki = 0; blki < matrix->tilem; blki++)
    {
        int start = blki * BLOCK_SIZE;
        int end = blki == matrix->tilem - 1 ? matrix->m : (blki + 1) * BLOCK_SIZE;
        nnz_temp = nnz_temp < matrix->rowpointer[end] - matrix->rowpointer[start] ? matrix->rowpointer[end] - matrix->rowpointer[start] : nnz_temp;
        tile_count_temp = tile_count_temp < matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki] ? matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki] : tile_count_temp;
    }

 
if(matrix->blocksize_flag!=1)
{
matrix->mask_32 = (unsigned int *)malloc(matrix->numtile * BLOCK_SIZE * sizeof(unsigned int));
    memset(matrix->mask_32, 0, matrix->numtile * BLOCK_SIZE * sizeof(unsigned int));
}
else
{
matrix->mask_64 = (unsigned long long int *)malloc(matrix->numtile * BLOCK_SIZE * sizeof(unsigned long long int));
    memset(matrix->mask_64, 0, matrix->numtile * BLOCK_SIZE * sizeof(unsigned long long int));
}

    step4_kernel(matrix, matrix->csr_ptr, matrix->hyb_coocount, nnz_temp, tile_count_temp,
                 matrix->csr_offset, matrix->csrptr_offset);
}
 
#endif
