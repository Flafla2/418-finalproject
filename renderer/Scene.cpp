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