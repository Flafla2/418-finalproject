#include <glm/geometric.hpp>

#include "Primitive.h"

float Sphere::sdf(glm::vec3 p) const {
    return glm::distance(p, center) - radius;
}

float Box::sdf(glm::vec3 p) const {
    p = p - center;
    glm::vec3 d = glm::abs(p) - dim;
    return glm::length(glm::max(d, 0.0)) + glm::min(glm::max(d.x,glm::max(d.y,d.z)), 0.0);
}