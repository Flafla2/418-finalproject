#ifndef RAYMARCHER_CUDASCENE_H
#define RAYMARCHER_CUDASCENE_H

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

__device__ float deviceSdf(glm::vec3 p);
__device__ glm::vec3 deviceNormal(glm::vec3 p);


#endif //RAYMARCHER_CUDASCENE_H
