#ifndef RAYMARCHER_SCENE_H
#define RAYMARCHER_SCENE_H

#include <vector>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

#include "Primitive.h"

class Scene {
public:
    Scene(std::vector<Primitive *> primitives);
    ~Scene();

    float sdf(glm::vec3 p);
    glm::vec3 normal(glm::vec3 p);
private:
    std::vector<Primitive *> primitives;
};


#endif //RAYMARCHER_SCENE_H
