#include "Scene.h"

#include <limits>
#include <algorithm>

glm::vec3 Scene::normal(glm::vec3 p) {
    static const float eps = 0.001f;
    static const glm::vec3 x = glm::vec3(eps, 0, 0);
    static const glm::vec3 y = glm::vec3(0, eps, 0);
    static const glm::vec3 z = glm::vec3(0, 0, eps);

    glm::vec3 ret(
            sdf(p + x) - sdf(p - x),
            sdf(p + y) - sdf(p - y),
            sdf(p + z) - sdf(p - z)
    );
    return glm::normalize(ret);
}