#ifndef RAYMARCHER_CUDA_ERROR_H
#define RAYMARCHER_CUDA_ERROR_H

#include <cuda.h>
#include <cuda_runtime.h>

#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans)  cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char*file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define __dbg_assert(cond) if(!(cond)) { \
        printf("----------\nAssertion Failed at %s:%d\n\t%s\n", __FILE__, __LINE__, #cond); \
    }

#define __dbg_exec(expr) expr
#else
#define cudaCheckError(ans) ans
#define __dbg_assert(cond)
#define __dbg_exec(expr)
#endif

#endif //RAYMARCHER_CUDA_ERROR_H
