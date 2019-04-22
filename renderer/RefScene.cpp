#include "RefScene.h"

RefScene::RefScene(std::vector<Primitive *> primitives) : primitives(primitives) { }

RefScene::~RefScene() { }

float RefScene::sdf(glm::vec3 p) {
    float ret = std::numeric_limits<float>::infinity();

    for (auto & primitive : primitives) {
        ret = std::min(primitive->sdf(p), ret);
    }

    return ret;
}
