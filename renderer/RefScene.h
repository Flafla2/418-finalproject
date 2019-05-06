#ifndef RAYMARCHER_REFSCENE_H
#define RAYMARCHER_REFSCENE_H

#include <glm/glm.hpp>
#include <vector>

#include "RefPrimitive.h"

class RefScene {
public:
    explicit RefScene(std::vector<RefPrimitive *> primitives);
    ~RefScene();

    float sdf(glm::vec3 const& p);
    glm::vec3 normal(glm::vec3 const& p);
private:
    std::vector<RefPrimitive *> primitives;
};


#endif //RAYMARCHER_REFSCENE_H
