#include <glm/geometric.hpp>

#include "RefPrimitive.h"

float RefSphere::sdf(glm::vec3 p) const {
    return glm::distance(p, center) - radius;
}

float RefBox::sdf(glm::vec3 p) const {
    p = p - center;
    glm::vec3 d = glm::abs(p) - dim;
    return glm::length(glm::max(d, glm::vec3(0.0))) + glm::min(glm::max(d.x,glm::max(d.y,d.z)), 0.0f);
}

float RefTorus::sdf(glm::vec3 p) const {
    glm::vec2 q = glm::vec2(glm::length(center.xz) - t.x, center.y);
    return glm::length(q) - t.y;
}

float RefCylinder::sdf(glm::vec3 p) const{
    return glm::length(center.xz - dim.xy) - dim.z;
}

float RefCone::sdf(glm::vec3 p) const{
    float q = glm::length(p.xy);
    return glm::dot(dim, glm::vec2(q, center.z));
}

float RefPlane::sdf(glm::vec3 p) const{
    return glm::dot(center, dim.xyz) + dim.w;
}