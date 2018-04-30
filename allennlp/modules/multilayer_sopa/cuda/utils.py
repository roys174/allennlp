UTIL = """
extern "C" {
    const int K = 6;
    __forceinline__ __device__ float sigmoidf(float x)
    {
        return 1.f / (1.f + expf(-x));
    }

    __forceinline__ __device__ float reluf(float x)
    {
        return (x > 0.f) ? x : 0.f;
    }

    __forceinline__ __device__ float seluf(float x)
    {
        return 1.0507009873554804934193349852946f * (
            (x > 0.f) ? x : 1.6732632423543772848170429916717f * (expf(x)-1.f)
        );
    }

    __forceinline__ __device__ float calc_activation(int type, float x)
    {
        switch (type) {
            case 0:
                return x;
            case 1:
                return tanh(x);
            case 2:
                return reluf(x);
            case 3:
                return seluf(x);
        }
        return x;
    }

    __forceinline__ __device__ float calc_grad_activation(int type, float x)
    {
        switch (type) {
            case 0:
                return 1.f;
            case 1:
                return 1.f-x*x;
            case 2:
                return (x > 0.f) ? 1.f : 0.f;
            case 3:
                return (x > 0.f) ? 1.0507009873554804934193349852946f :
                    x + 1.7580993408473766f;
        }
        return 1.f;
    }
}
"""
