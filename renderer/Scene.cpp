#include "Scene.h"

#include <limits>
#include <algorithm>

Scene::Scene(std::vector<Primitive *> primitives) : primitives(primitives) { }

Scene::~Scene() {

}

float Scene::sdf(glm::vec3 p) {
    float ret = std::numeric_limits<float>::infinity();

    for (auto & primitive : primitives) {
        ret = std::min(primitive->sdf(p), ret);
    }

    return ret;
}

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