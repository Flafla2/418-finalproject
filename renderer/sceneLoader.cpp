#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <functional>

#include "sceneLoader.h"
#include "util.h"

#include "Primitive.h"

/// Gets a random float
/// \return A random floating point value between 0 and 1
static float randomFloat() {
    return static_cast<float>(rand()) / RAND_MAX;
}

Scene *SceneLoader::loadScene(SceneName sceneName)
{
    std::vector<Primitive *> prims;
    if (sceneName == TEST_SCENE) {
        prims.push_back(new Sphere(glm::vec3(-1.2f,0,0), 1.f));
        prims.push_back(new Sphere(glm::vec3( 1.2f,0,0), 1.f));
    } else {
        fprintf(stderr, "Error: cann't load scene (unknown scene)\n");
        return nullptr;
    }

    printf("Loaded scene\n");
    return new Scene(prims);
}
