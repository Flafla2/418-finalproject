#include "CudaScene.h"

CudaScene::CudaScene(std::vector<Primitive *> primitives) {
    for ( auto & p : primitives ) {
        Sphere *s = dynamic_cast<Sphere *>(p);
        if (s) {
            spheres.push_back(*s);
        }
    }
}

CudaScene::~CudaScene() = default;

float CudaScene::sdf(glm::vec3 p) {
    float ret = std::numeric_limits<float>::infinity();

    for (auto & s : spheres) {
        ret = std::min(s.sdf(p), ret);
    }

    return ret;
}
