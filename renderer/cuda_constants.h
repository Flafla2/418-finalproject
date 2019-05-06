#ifndef __CUDA_CONSTANTS_STRUCTS__
#define __CUDA_CONSTANTS_STRUCTS__

#include "renderer.h"

struct GlobalConstants {
    SceneName sceneName;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

struct SceneConstants {
    char *bytecode;
    int bytecodeSize;
};

#endif


