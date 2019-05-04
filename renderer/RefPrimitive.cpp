#define GLM_SWIZZLE
#include <glm/glm.hpp>
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
    p = p - center;
    glm::vec2 q = glm::vec2(glm::length(p.xz()) - radius, p.y);
    return glm::length(q) - thickness;
}

float RefCylinder::sdf(glm::vec3 p) const{
    p = p - center;
    return glm::length(p.xz()) - radius;
}

float RefCone::sdf(glm::vec3 p) const{
    p = p - center;
    float q = glm::length(p.xy());
    return glm::dot(dir, glm::vec2(q, p.z));
}

float RefPlane::sdf(glm::vec3 p) const {
    p = p - center;
    return glm::dot(p, normal) + offset;
}