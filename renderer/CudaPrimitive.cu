#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

#define GLM_FORCE_CUDA
#include <glm/geometric.hpp>

#include "CudaPrimitive.h"

__device__ __host__
float CudaSphere::sdf(glm::vec3 p) const {
    printf("CudaSphere sdf called.  p: (%f, %f, %f), center: (%f, %f, %f) radius: %f\n",
            p.x, p.y, p.z, center.x, center.y, center.z, radius);
    return glm::distance(p, center) - radius;
}

__device__ __host__
float CudaBox::sdf(glm::vec3 p) const {
    p = p - center;
    glm::vec3 d = glm::abs(p) - dim;
    return glm::length(glm::max(d, glm::vec3(0.0))) + glm::min(glm::max(d.x,glm::max(d.y,d.z)), 0.0f);
}