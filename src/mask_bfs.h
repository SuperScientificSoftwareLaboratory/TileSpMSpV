__global__ 
void mask_pull_flag(int *d_mask_x_c,int *d_mask_x_c_1,unsigned long long int *mask_flag,unsigned long long int *flag_1,int tilem,int m,unsigned long long int * d_flag_1_c)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

   //  printf("%d  tilem=%d\n",global_id,tilem);
    if (global_id < tilem)
    {
        unsigned long long int sum=~mask_flag[global_id];
      //  printf("%d  tilem=%d  sum=%llu\n",global_id,tilem,sum);
        
        if(sum != 0)
        {   
            unsigned long long int tmp1=sum;
            unsigned long long int num=0;
            while(tmp1)
            {    
                if(tmp1&1==1)
                {   
                    int x_id=(global_id<<6) + 63 - num;
                    if(x_id<m)
                    {      
                        unsigned long long int k=atomicAdd(&flag_1[0],1); 
                        d_mask_x_c[k]=x_id;
                    }
                }
                tmp1 = tmp1 >> 1;
                num++;   
            }
            
        }
    }
}



__global__ 
void mask_spmv_64_v2(int rbnum, int cbnum, int rowA, int colA, MAT_PTR_TYPE *d_rowblock_ptr, int *d_columnid,int rowblkblock,      unsigned int * d_blkcoostylerowidx, int * d_blkcoostylerowidx_colstart, int * d_blkcoostylerowidx_colstop,
unsigned long long int  *d_mask_x,unsigned long long int  *d_mask_x_1, unsigned long long int  *d_mask_y, unsigned long long int  *mask,unsigned long long int *mask_flag ,unsigned long long int *flag_1
,int *d_mask_x_c_1, int *d_mask_x_c,int *d_idx_xxx,unsigned long long int *d_flag_1_c,unsigned long long int *d_flag_2
)
{    
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5; // WARP_SIZE;

    const int local_warp_id = threadIdx.x >> 5; // / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    if (blki_blc < rowblkblock)
    {
        int coostyleblkrowidx = d_blkcoostylerowidx[blki_blc];
        int signbit = (coostyleblkrowidx >> 31) & 0x1;

        int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;

        int rowblkjstart = signbit == 1 ? d_blkcoostylerowidx_colstart[blki_blc] : d_rowblock_ptr[blki];
        int rowblkjstop = signbit == 1 ? d_blkcoostylerowidx_colstop[blki_blc] : d_rowblock_ptr[blki+1];

        
        for (int blkj = rowblkjstart; blkj < rowblkjstop; blkj+=32)
        {
	    
	        int blkj_laneid = blkj+lane_id;
            if(blkj_laneid>= rowblkjstop) break;
          
            unsigned long long int mask_x_i=d_mask_x[d_columnid[blkj_laneid]];
            if(mask_x_i==0) continue;
        
            unsigned long long int sum_x=0;
       
            for(int rj=0;rj<BLOCK_SIZE_64;rj++)
            {
            
                if((mask[(blkj_laneid<<6) + rj] & mask_x_i)!=0)
                {
                    sum_x |= ((unsigned long long int )1) << (63-rj);
                }
            
            }
            unsigned long long int sum=(~(mask_flag[blki]&sum_x))&sum_x;
             
            if(sum != 0)
            {   
                unsigned long long int tmp1=sum;
                unsigned long long int num=0;
                while(tmp1)
               {    
                    if(tmp1&1==1)
                    {
                        int y_id=(blki<<6) + 63 - num;          
                        unsigned long long int one=1;
                        unsigned long long int kkk=atomicOr(&d_flag_1_c[y_id],one);
                         
                        if(kkk==0)
                        {
                             unsigned long long int k=atomicAdd(&flag_1[0],1);   
                             d_mask_x_c_1[k]=y_id;
                                
                        }
                    }
                    tmp1 = tmp1 >> 1;
                    num++;   
                }
                atomicOr(&d_mask_x_1[blki],sum);
                atomicOr(&mask_flag[blki],sum_x);
            }
           
        }
    }

}


__global__ 
void mask_spmv_32_v2(int rbnum, int cbnum, int rowA, int colA, MAT_PTR_TYPE *d_rowblock_ptr, int *d_columnid,int rowblkblock, 
                  unsigned int * d_blkcoostylerowidx, int * d_blkcoostylerowidx_colstart, int * d_blkcoostylerowidx_colstop,
                  unsigned int *d_mask_x,unsigned int *d_mask_x_1, unsigned int *d_mask_y, unsigned int *mask,unsigned int *mask_flag ,unsigned int *flag_1)
{    
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
   // printf("global_id=%d\n",global_id);
    const int blki_blc = global_id >> 5; // WARP_SIZE;
    const int local_warp_id = threadIdx.x >> 5; // / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;


    if (blki_blc < rowblkblock)
    {
 
        int coostyleblkrowidx = d_blkcoostylerowidx[blki_blc];
        int signbit = (coostyleblkrowidx >> 31) & 0x1;

        int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;

        int rowblkjstart = signbit == 1 ? d_blkcoostylerowidx_colstart[blki_blc] : d_rowblock_ptr[blki];
        int rowblkjstop = signbit == 1 ? d_blkcoostylerowidx_colstop[blki_blc] : d_rowblock_ptr[blki+1];

        for (int blkj = rowblkjstart; blkj < rowblkjstop; blkj+=32)
        {
	    
	        int blkj_laneid = blkj+lane_id;
            if(blkj_laneid>= rowblkjstop) break;
          
            unsigned int mask_x_i=d_mask_x[d_columnid[blkj_laneid]];
            if(mask_x_i==0) continue;
         
            unsigned int sum_x=0;
       
            for(int rj=0;rj<BLOCK_SIZE_32;rj++)
            {
                if((mask[(blkj_laneid<<5) + rj] & mask_x_i)!=0)
                {
                    sum_x |= 1 << (31-rj);
                }
            }
            if(sum_x!=0)
            {
            
                atomicOr(&d_mask_y[blki],sum_x);
                atomicOr(&d_mask_x_1[blki], ~(mask_flag[blki] & d_mask_y[blki]) & d_mask_y[blki]);
                if(d_mask_x_1[blki]!=0){ 
                    flag_1[0]=1;
               
                }
    	     
    	         if(signbit)
    	            atomicOr(&mask_flag[blki],d_mask_y[blki]);
    	         else
    	            mask_flag[blki]= mask_flag[blki] | d_mask_y[blki];
    	       
            }
            
           
        }
}

}



__global__ 
void mask_spmv_32_v1(int rbnum, int cbnum, int rowA, int colA, MAT_PTR_TYPE *d_rowblock_ptr, int *d_columnid,int rowblkblock, 
unsigned int * d_blkcoostylerowidx, int * d_blkcoostylerowidx_colstart, int * d_blkcoostylerowidx_colstop,int * d_blk_colid,
                  unsigned int *d_mask_x,unsigned int *d_mask_x_1, unsigned int *d_mask_y, unsigned int *mask,
                  unsigned int *mask_flag ,unsigned int *flag_1,unsigned int *d_mask_x_c,unsigned int *d_mask_x_c_1, int *idx_xxx,unsigned int * d_flag_1_c,unsigned int * flag_2)
{    

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int blki_blc = global_id >> 5; // WARP_SIZE;
    const int local_warp_id = threadIdx.x >> 5; // / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    unsigned int x_id=d_mask_x_c[blki_blc];
    int blki=x_id >> 5;
    unsigned int x_offset=x_id%32;
    if(blki_blc<flag_2[0])
    {
  
        for(int blkj=d_rowblock_ptr[blki]+lane_id;blkj<d_rowblock_ptr[blki+1];blkj+=32)//WARP_SIZE
        {
              
            unsigned int sum_x=mask[(blkj<<5) + x_offset] ;
            int blki_y=d_columnid[blkj];
                
            unsigned int sum=(~(mask_flag[blki_y]&sum_x))&sum_x;
               
            if(sum != 0)
            {   
                unsigned int tmp1=sum;
                unsigned int num=0;
                   
                while(tmp1)
                {    
                      
                    if(tmp1&1==1)
                    {
                        int y_id=(blki_y<<5) + 31 - num;
                        unsigned int one=1;
                        unsigned int kkk=atomicOr(&d_flag_1_c[y_id],one);
                        if(kkk==0)
                        {
                             unsigned int k=atomicAdd(&flag_1[0],1);
                             d_mask_x_c_1[k]=y_id;
                                 
                        }
                    }
                    tmp1 = tmp1 >> 1;
                    num++;   
                }
                atomicOr(&d_mask_x_1[blki_y],sum);
                atomicOr(&mask_flag[blki_y],sum_x);
            }
        }
    }
}


__global__ 
void mask_spmv_64_v1(int rbnum, int cbnum, int rowA, int colA, MAT_PTR_TYPE *d_rowblock_ptr, int *d_columnid,int rowblkblock, 
unsigned int * d_blkcoostylerowidx, int * d_blkcoostylerowidx_colstart, int * d_blkcoostylerowidx_colstop,int * d_blk_colid,
unsigned long long int *d_mask_x,unsigned long long int *d_mask_x_1, unsigned long long int *d_mask_y, unsigned long long int *mask,
unsigned long long int *mask_flag ,unsigned long long int *flag_1,int *d_mask_x_c,int *d_mask_x_c_1, int *idx_xxx,
unsigned long long int * d_flag_1_c,unsigned long long int * flag_2)
{    

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5; // WARP_SIZE;
    const int local_warp_id = threadIdx.x >> 5; // / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    if(blki_blc<flag_2[0])
    {
        int x_id=d_mask_x_c[blki_blc];
        int blki=x_id >> 6;
        int x_offset=x_id%64;

        for(int blkj=d_rowblock_ptr[blki]+lane_id;blkj<d_rowblock_ptr[blki+1];blkj+=32)//WARP_SIZE
        {              
                unsigned long long int sum_x=mask[(blkj<<6) + x_offset] ;
                int blki_y=d_columnid[blkj];
                unsigned long long int sum=(~(mask_flag[blki_y]&sum_x))&sum_x;
               
                if(sum != 0)
                {   
                    unsigned long long int tmp1=sum;
                    unsigned long long int num=0;
                    while(tmp1)
                    {    
                        if(tmp1&1==1)
                        {
                            int y_id=(blki_y<<6) + 63 - num;          
                            unsigned long long int one=1;
                            unsigned long long int kkk=atomicOr(&d_flag_1_c[y_id],one);
                            
                            if(kkk==0)
                            {
                                 unsigned long long int k=atomicAdd(&flag_1[0],1);   
                                 d_mask_x_c_1[k]=y_id;  
                             }
                        }
                         tmp1 = tmp1 >> 1;
                         num++;   
                    }
                    atomicOr(&d_mask_x_1[blki_y],sum);
                    atomicOr(&mask_flag[blki_y],sum_x);
                }
        }
    }
}



__global__ 
void mask_spmv_64_pull_v1(int rbnum, int cbnum, int rowA, int colA, MAT_PTR_TYPE *d_rowblock_ptr, int *d_columnid,int rowblkblock, 
unsigned int * d_blkcoostylerowidx, int * d_blkcoostylerowidx_colstart, int * d_blkcoostylerowidx_colstop,int * d_blk_colid,
unsigned long long int *d_mask_x,unsigned long long int *d_mask_x_1, unsigned long long int *d_mask_y, unsigned long long int *mask,unsigned long long int *mask_flag,
unsigned long long int *mask_flag_1 ,unsigned long long int *flag_1,int *d_mask_x_c,int *d_mask_x_c_1, int *idx_xxx,
unsigned long long int * d_flag_1_c,unsigned long long int * flag_2,int pull_num_x)
{    

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5; // WARP_SIZE;
    const int local_warp_id = threadIdx.x >> 5; // / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    if(blki_blc<pull_num_x )
    {
        
        int x_id=d_mask_x_c[blki_blc];
        
    
        if(x_id==-1) return ;
        int blki=x_id >> 6;
        int x_offset=x_id%64;
        int flag=0;
        unsigned long long int sum_x=0;
        int blki_y=0;
        
     
        for(int blkj=d_rowblock_ptr[blki]+lane_id;blkj<d_rowblock_ptr[blki+1];blkj+=32)//WARP_SIZE
        {              
               if(d_mask_x_c[blki_blc]==-1) break;
               
                sum_x=mask[(blkj<<6) + x_offset] ;//*64
                blki_y=d_columnid[blkj];
                sum_x=mask_flag[blki_y]&sum_x;
                if(sum_x != 0)
                { 
                    unsigned long long int kkk=atomicOr(&d_flag_1_c[x_id],1);
                    
                    if(kkk==0)
                    {   
                        unsigned long long int num=atomicAdd(&flag_1[0],1);
                        sum_x = (((unsigned long long int )0x1) << (63 - x_offset));
                        unsigned long long int sum_x_1=sum_x;
                        unsigned long long int k=sum_x_1 & mask_flag_1[blki];
                    
                        atomicOr(&mask_flag_1[blki],sum_x);
                    
                        d_mask_x_c[blki_blc]=-1;
                        flag=1;
                    }
                    break;
                }

        }
     }
}


