#include <glm/geometric.hpp>

#include "Primitive.h"

float Sphere::sdf(glm::vec3 p) const {
    return glm::distance(p, center) - radius;
}
