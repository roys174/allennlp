SOPA = """
            
extern "C" {
     __global__ void sopa_fwd(
                const float * __restrict__ u, 
                const float * __restrict__ c1_init,
                const float * __restrict__ c2_init,
                const float * __restrict__ d_init,
                const int len, 
                const int batch, 
                const int dim, 
                const int k,
                float * __restrict__ c1,
                float * __restrict__ c2,
                float * __restrict__ d) {
        assert (k == K);
        int ncols = batch*dim;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;
        int ncols_u = ncols*k;

        const float *up = u + (col*k);
        float *c1p = c1 + col;
        float *c2p = c2 + col;
        float *dp = d + col;
        float cur_c1 = *(c1_init + col);
        float cur_c2 = *(c2_init + col);
        float cur_d = *(d_init + col);
        
        for (int row = 0; row < len; ++row) {
            float x_tilde1 = *(up);
            float x_tilde2 = *(up+1);
            float x_tilde3 = *(up+2);
            float forget1 = *(up+3);
            float forget2 = *(up+4);
            float selfloop = *(up+5);
            
            cur_c1 = (cur_c1 - x_tilde1) * forget1 + x_tilde1;
            float tmp = x_tilde3 * cur_d;
            cur_c2 = (cur_c2 - tmp) * forget2 + tmp;
            cur_d = (cur_d - x_tilde2) * selfloop + x_tilde2;
            
            *c1p = cur_c1;
            *c2p = cur_c2;
            *dp = cur_d;
            
            up += ncols_u;
            c1p += ncols;
            c2p += ncols;
            dp += ncols;
        }
    }
    
    __global__ void sopa_bwd(
                const float * __restrict__ u, 
                const float * __restrict__ c1_init,
                const float * __restrict__ c2_init,
                const float * __restrict__ d_init,
                const float * __restrict__ c1,
                const float * __restrict__ c2,
                const float * __restrict__ d,
                const float * __restrict__ grad_c1, 
                const float * __restrict__ grad_c2, 
                const float * __restrict__ grad_last_c1,
                const float * __restrict__ grad_last_c2,
                const float * __restrict__ grad_last_d,
                const int len, 
                const int batch, 
                const int dim, 
                const int k,
                float * __restrict__ grad_u, 
                float * __restrict__ grad_c1_init,
                float * __restrict__ grad_c2_init,
                float * __restrict__ grad_d_init) {
        assert (k == K);
        int ncols = batch*dim;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;

        float gd = 0.f;
        float cur_c1 = *(grad_last_c1 + col);
        float cur_c2 = *(grad_last_c2 + col);
        float cur_d = *(grad_last_d + col);

        const float *up = u + (col*k) + (len-1)*ncols_u;
        const float *c1p = c1 + col + (len-1)*ncols;
        const float *c2p = c2 + col + (len-1)*ncols;
        const float *dp = d + col + (len-1)*ncols;
        
        const float *gc1p = grad_c1 + col + (len-1)*ncols;
        const float *gc2p = grad_c2 + col + (len-1)*ncols;
        float *gup = grad_u + (col*k) + (len-1)*ncols_u;
        
        for (int row = len-1; row >= 0; --row) {
            float x_tilde1 = *(up);
            float x_tilde2 = *(up+1);
            float x_tilde3 = *(up+2);
            float forget1 = *(up+3);
            float forget2 = *(up+4);
            float selfloop = *(up+5);
        
            const float c1_val = *c1p;
            const float c2_val = *c2p;
            const float d_val = *dp; 
            
            const float prev_c1_val = (row>0) ? (*(c1p-ncols)) : (*(c1_init+col));
            const float prev_c2_val = (row>0) ? (*(c2p-ncols)) : (*(c2_init+col));
            const float prev_d_val = (row>0) ? (*(dp-ncols)) : (*(d_init+col));
            
            const float gc1 = *(gc1p) + cur_c1;
            const float gc2 = *(gc2p) + cur_c2;
            
            float ginput1 = gc1*(1.f-forget1);
            *(gup) = ginput1;
            
            float gforget1 = gc1*(prev_c1_val-x_tilde1);
            *(gup+3) = gforget1;
            
            float ginput3 = gc2*(1.f-forget2)*prev_d_val;
            *(gup+2) = ginput3;
            
            float gforget2 = gc2*(prev_c2_val-x_tilde3*prev_d_val);
            *(gup+4) = gforget2;
            
            float ginput2 = gd*(1.f-selfloop);
            *(gup+1) = ginput2;
            
            float gselfloop = gd*(prev_d_val-x_tilde2);
            *(gup+5) = gselfloop;
            
            cur_d = gd * selfloop;
            gd=gc2*(1.f-forget2)*x_tilde3 + cur_d;
            
            cur_c1 = gc1 * forget1;
            cur_c2 = gc2 * forget2;

            up -= ncols_u; 
            c1p -= ncols;
            c2p -= ncols;
            dp -= ncols;
            gup -= ncols_u;
            gc1p -= ncols;
            gc2p -= ncols;
        }
        
        *(grad_c1_init + col) = cur_c1;
        *(grad_c2_init + col) = cur_c2;
        *(grad_d_init + col) = gd;
    }
}
"""
