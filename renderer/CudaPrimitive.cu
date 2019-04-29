#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/geometric.hpp>

#include "CudaPrimitive.h"

__device__ __host__
float CudaSphere::sdf(glm::vec3 p) const {
    return glm::distance(p, center) - radius;
}

__device__ __host__
float CudaBox::sdf(glm::vec3 p) const {
    p = p - center;
    glm::vec3 d = glm::abs(p) - dim;
    return glm::length(glm::max(d, glm::vec3(0.0))) + glm::min(glm::max(d.x,glm::max(d.y,d.z)), 0.0f);
}