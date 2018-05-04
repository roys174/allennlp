SRU_BIDIR= """
extern "C" {

    __global__ void sru_bi_fwd(const float * __restrict__ u,
                               const float * __restrict__ init,
                               const int len, 
                               const int batch, 
                               const int d, 
                               const int k,
                               float * __restrict__ c,
                               const int activation_type) {
        assert (k == 2);

        int ncols = batch*d*2;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;
        float cur = *(init + col);

        const int d2 = d*2;
        const bool flip = (col%d2) >= d;

        const float *up = u + (col*k);
        float *cp = c + col;

        if (flip) {
            up += (len-1)*ncols_u;
            cp += (len-1)*ncols;
        }

        int ncols_u_ = flip ? -ncols_u : ncols_u;
        int ncols_ = flip ? -ncols : ncols;

        for (int row = 0; row < len; ++row) {
            float g1 = *(up+1);
            cur = (cur-(*up))*g1 + (*up);
            cur = calc_activation(activation_type, cur);
            *cp = cur;
            up += ncols_u_;
            cp += ncols_;
        }

    }

    __global__ void sru_bi_bwd(const float * __restrict__ u, 
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
        
        int ncols = batch*d*2;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= ncols) return;

        int ncols_u = ncols*k;
        float cur = *(grad_last + col);

        const int d2 = d*2;
        const bool flip = ((col%d2) >= d);

        const float *up = u + (col*k);
        const float *cp = c + col;
        
        const float *gcp = grad_c + col;
        float *gup = grad_u + (col*k);

        if (!flip) {
            up += (len-1)*ncols_u;
            cp += (len-1)*ncols;
            gup += (len-1)*ncols_u;
            gcp += (len-1)*ncols;
        }

        int ncols_u_ = flip ? -ncols_u : ncols_u;
        int ncols_ = flip ? -ncols : ncols;

        for (int row = 0; row < len; ++row) {
            const float g1 = *(up+1);
            const float c_val = *cp;
            const float u_val = *up;
            const float prev_c_val = (row<len-1) ? (*(cp-ncols_)) : (*(init+col));

            const float gc = (*gcp + cur)*calc_grad_activation(activation_type, c_val);

            // grad wrt u0
            *gup = gc*(1-g1);

            // grad wrt g1, u1, and bias1
            float gg1 = gc*(prev_c_val-u_val);
            *(gup+1) = gg1;

            // grad wrt c'
            cur = gc*g1;

            up -= ncols_u_;
            cp -= ncols_;
            gup -= ncols_u_;
            gcp -= ncols_;
        }
        *(grad_init +col) = cur;
    }
}
"""
