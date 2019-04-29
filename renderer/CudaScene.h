#ifndef RAYMARCHER_CUDASCENE_H
#define RAYMARCHER_CUDASCENE_H

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#endif

#include <vector>

#include "CudaPrimitive.h"

class CudaScene {
public:
    explicit CudaScene(std::vector<CudaPrimitive *> primitives);
    ~CudaScene();

    void initCudaData();
private:
    std::vector<CudaSphere> spheres;

    CudaSphere *cudaDeviceSphereData;
};

#if defined(__CUDACC__)
__device__ float deviceSdf(glm::vec3 p);
__device__ glm::vec3 deviceNormal(glm::vec3 p);
#endif

#endif //RAYMARCHER_CUDASCENE_H
