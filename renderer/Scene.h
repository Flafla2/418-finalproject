#ifndef RAYMARCHER_SCENE_H
#define RAYMARCHER_SCENE_H

#include <vector>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

class Scene {
public:
    virtual float sdf(glm::vec3 p) = 0;

    glm::vec3 normal(glm::vec3 p);
};


#endif //RAYMARCHER_SCENE_H
