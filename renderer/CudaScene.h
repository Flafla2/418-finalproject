#ifndef RAYMARCHER_CUDASCENE_H
#define RAYMARCHER_CUDASCENE_H

#include <vector>

#include "Scene.h"

class CudaScene : public Scene {
public:
    explicit CudaScene(std::vector<Primitive *> primitives);
    ~CudaScene();

    float sdf(glm::vec3 p) override;
private:
    std::vector<Sphere> spheres;
};


#endif //RAYMARCHER_CUDASCENE_H
