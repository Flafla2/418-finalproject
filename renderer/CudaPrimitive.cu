#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

#define GLM_FORCE_CUDA
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
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

__device__ __host__
float TorusSDF(CudaTorus const& torus, glm::vec3 p) {
    p = p - torus.center;
    glm::vec2 q = glm::vec2(glm::length(p.xz()) - torus.radius, p.y);
    return glm::length(q) - torus.thickness;
}

__device__ __host__
float CylinderSDF(CudaCylinder const& cylinder, glm::vec3 p) {
    p = p - cylinder.center;
    return glm::length(p.xz()) - cylinder.radius;
}

__device__ __host__
float ConeSDF(CudaCone const& cone, glm::vec3 p) {
    p = p - cone.center;
    float q = glm::length(p.xy());
    return glm::dot(cone.dir, glm::vec2(q, p.z));
}

__device__ __host__
float PlaneSDF(CudaPlane const& plane, glm::vec3 p) {
    p = p - plane.center;
    return glm::dot(p, plane.normal) + plane.offset;
}
