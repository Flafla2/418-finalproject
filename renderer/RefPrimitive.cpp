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