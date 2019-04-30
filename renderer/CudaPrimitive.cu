#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

#define GLM_FORCE_CUDA
#include <glm/geometric.hpp>

#include "CudaPrimitive.h"

__device__ __host__
float SphereSDF(CudaSphere const& sphere, glm::vec3 p) {
    return glm::distance(p, sphere.center) - sphere.radius;
}

__device__ __host__
float BoxSDF(CudaBox const& box, glm::vec3 p) {
    p = p - box.center;
    glm::vec3 d = glm::abs(p) - box.dim;
    return glm::length(glm::max(d, glm::vec3(0.0))) + glm::min(glm::max(d.x,glm::max(d.y,d.z)), 0.0f);
}