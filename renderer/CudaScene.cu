#include "CudaScene.h"

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

CudaScene::CudaScene(std::vector<CudaPrimitive *> primitives) {
    for ( auto & p : primitives ) {
        CudaSphere *s = dynamic_cast<CudaSphere *>(p);
        if (s) {
            spheres.push_back(*s);
        }
    }
}

CudaScene::~CudaScene() = default;

struct SceneConstants {
    CudaSphere *sphereData;
    int nSphere;
};

__constant__ SceneConstants cudaConstSceneParams;

__device__ float deviceSdf(glm::vec3 p) {
    float ret = 1000000.0f;
    for (int i = 0; i < cudaConstSceneParams.nSphere; ++i) {
        float sdf = cudaConstSceneParams.sphereData[i].sdf(p);
        ret = glm::min(sdf, ret);
    }
    return ret;
}

__device__ glm::vec3 deviceNormal(glm::vec3 p) {
    const float eps = 0.001f;
    const glm::vec3 x = glm::vec3(eps, 0, 0);
    const glm::vec3 y = glm::vec3(0, eps, 0);
    const glm::vec3 z = glm::vec3(0, 0, eps);

    glm::vec3 ret(
            deviceSdf(p + x) - deviceSdf(p - x),
            deviceSdf(p + y) - deviceSdf(p - y),
            deviceSdf(p + z) - deviceSdf(p - z)
    );
    return glm::normalize(ret);
}

void CudaScene::initCudaData() {
    static bool cudaDataInitialized = false;
    if (cudaDataInitialized) {
        std::cerr << "ERROR: Scene Cuda data already initialized!  Exiting." << std::endl;
        exit(1);
    }
    cudaDataInitialized = true;

    size_t const sphereSize = sizeof(Sphere) * spheres.size();
    cudaMalloc(&cudaDeviceSphereData, sphereSize);
    cudaMemcpy(cudaDeviceSphereData, spheres.data(), sphereSize, cudaMemcpyHostToDevice);

    SceneConstants params;
    params.sphereData = cudaDeviceSphereData;
    params.nSphere = spheres.size();

    cudaMemcpyToSymbol(cudaConstSceneParams, &params, sizeof(SceneConstants));
}