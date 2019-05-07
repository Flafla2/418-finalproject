#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/geometric.hpp>

#include "CudaPrimitive.h"

__device__ __host__
float SphereSDF(CudaSphere const& sphere, glm::vec3 p) {
    p = sphere.world2local * glm::vec4(p, 1.f);
    return glm::length(p) - sphere.radius;
}

__device__ __host__
float BoxSDF(CudaBox const& box, glm::vec3 p) {
    p = box.world2local * glm::vec4(p, 1.f);
    glm::vec3 d = glm::abs(p) - box.dim;
    return glm::length(glm::max(d, glm::vec3(0.0))) + glm::min(glm::max(d.x,glm::max(d.y,d.z)), 0.0f);
}

__device__ __host__
float TorusSDF(CudaTorus const& torus, glm::vec3 p) {
    p = torus.world2local * glm::vec4(p, 1.f);
    glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x,p.z)) - torus.radius, p.y);
    return glm::length(q) - torus.thickness;
}

__device__ __host__
float CylinderSDF(CudaCylinder const& cylinder, glm::vec3 p) {
    p = cylinder.world2local * glm::vec4(p, 1.f);
    return glm::length(glm::vec2(p.x,p.z)) - cylinder.radius;
}

__device__ __host__
float ConeSDF(CudaCone const& cone, glm::vec3 p) {
    p = cone.world2local * glm::vec4(p, 1.f);
    float q = glm::length(glm::vec2(p.x,p.y));
    return glm::dot(cone.dir, glm::vec2(q, p.z));
}

__device__ __host__
float PlaneSDF(CudaPlane const& plane, glm::vec3 p) {
    p = plane.world2local * glm::vec4(p, 1.f);
    return glm::dot(p, plane.normal) + plane.offset;
}
