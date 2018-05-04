SRU = """
extern "C" {

    __global__ void sru_fwd(const float * __restrict__ u,
                            const float * __restrict__ init,
                            const int len, 
                            const int batch, 
                            const int d, 
                            const int k,
                            float * __restrict__ c,
                            const int activation_type) {
        assert (k == 2);
        int ncols = batch*d;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;
        float cur = *(init + col);

        const float *up = u + (col*k);
        float *cp = c + col;
        for (int row = 0; row < len; ++row) {
            float g1 = *(up+1);
            cur = (cur-(*up))*g1 + (*up);
            cur = calc_activation(activation_type, cur);
            *cp = cur;
            up += ncols_u;
            cp += ncols;
        }
    }

    __global__ void sru_bwd(const float * __restrict__ u, 
                            const float * __restrict__ init,
                            const float * __restrict__ c,
                            const float * __restrict__ grad_c, 
                            const float * __restrict__ grad_last,
                            const int len, 
                            const int batch, 
                            const int d, 
                            const int k,
                            float * __restrict__ grad_u,
                            float * __restrict__ grad_init,
                            const int activation_type) {
        assert(k == 2);

        int ncols = batch*d;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;

        // this is actually zero
        float cur = *(grad_last + col);

        const float *up = u + (col*k) + (len-1)*ncols_u;
        const float *cp = c + col + (len-1)*ncols;

        const float *gcp = grad_c + col + (len-1)*ncols;
        float *gup = grad_u + (col*k) + (len-1)*ncols_u;

        for (int row = len-1; row >= 0; --row) {
            const float g1 = *(up+1);
            const float c_val = *cp;
            const float u_val = *up;
            const float prev_c_val = (row>0) ? (*(cp-ncols)) : (*(init+col));

            // grad wrt c
            const float gc = (*gcp + cur)*calc_grad_activation(activation_type, c_val);

            // grad wrt u0
            *gup = gc*(1-g1);

            float gg1 = gc*(prev_c_val-u_val);
            *(gup+1) = gg1;

            // grad wrt c'
            cur = gc*g1;
            
            up -= ncols_u;
            cp -= ncols;
            gup -= ncols_u;
            gcp -= ncols;
        }

        *(grad_init +col) = cur;
    }
}
"""
