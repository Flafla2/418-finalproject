#ifndef __CUDA_CONSTANTS_STRUCTS__
#define __CUDA_CONSTANTS_STRUCTS__

#include "renderer.h"
#include "Cubemap.h"

struct GlobalConstants {
    SceneName sceneName;

    int imageWidth;
    int imageHeight;
    float* imageData;

    PngImage background;
    PngImage lighting;
};

struct SceneConstants {
    char *bytecode;
    int bytecodeSize;
};

#endif


