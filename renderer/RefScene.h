#ifndef RAYMARCHER_REFSCENE_H
#define RAYMARCHER_REFSCENE_H

#include "Primitive.h"
#include "Scene.h"

class RefScene : public Scene {
public:
    explicit RefScene(std::vector<Primitive *> primitives);
    ~RefScene();

    float sdf(glm::vec3 p) override;
private:
    std::vector<Primitive *> primitives;
};


#endif //RAYMARCHER_REFSCENE_H
