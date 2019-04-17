#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <functional>

#include "sceneLoader.h"
#include "util.h"

// randomFloat --
//
// return a random floating point value between 0 and 1
static float
randomFloat() {
    return static_cast<float>(rand()) / RAND_MAX;
}

void SceneLoader::loadScene(SceneName sceneName)
{
    if (sceneName == TEST_SCENE) {
        // TODO
    } else {
        fprintf(stderr, "Error: cann't load scene (unknown scene)\n");
        return;
    }

    printf("Loaded scene\n");
}
