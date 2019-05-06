#include "RefScene.h"

RefScene::RefScene(std::vector<RefPrimitive *> primitives) : primitives(primitives) { }

RefScene::~RefScene() { }

float RefScene::sdf(glm::vec3 const& p) {
    float ret = std::numeric_limits<float>::infinity();

    for (auto & primitive : primitives) {
        ret = std::min(primitive->sdf(p), ret);
    }

    return ret;
}

glm::vec3 RefScene::normal(glm::vec3 const& p) {
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