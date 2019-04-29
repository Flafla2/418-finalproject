#include <glm/geometric.hpp>

#include "RefPrimitive.h"

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

__device__ __host__
float CudaTorus::sdf(glm::vec3 p) const{
    glm::vec2 q = glm::vec2(glm::length(center.xz) - t.x, center.y);
    return glm::length(q) - t.y;
}

__device__ __host__
float CudaCylinder::sdf(glm::vec3 p) const{
    return glm::length(center.xz - dim.xy) - dim.z;

}

__device__ __host__
float CudaCone::sdf(glm::vec3 p) const{

    float q = glm::length(p.xy);
    return glm::dot(dim, glm::vec2(q, center.z));
}

__device__ __host__
float CudaPlane::sdf(glm::vec3 p) const{
    return glm::dot(center, dim.xyz) + dim.w;
}
