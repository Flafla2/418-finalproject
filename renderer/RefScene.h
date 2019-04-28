#ifndef RAYMARCHER_REFSCENE_H
#define RAYMARCHER_REFSCENE_H

#include "RefPrimitive.h"
#include "Scene.h"

class RefScene : public Scene {
public:
    explicit RefScene(std::vector<RefPrimitive *> primitives);
    ~RefScene();

    float sdf(glm::vec3 p) override;
private:
    std::vector<RefPrimitive *> primitives;
};


#endif //RAYMARCHER_REFSCENE_H
