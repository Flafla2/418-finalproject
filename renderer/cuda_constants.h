
#include "renderer.h"

#ifndef __CUDA_CONSTANTS_STRUCTS__
#define __CUDA_CONSTANTS_STRUCTS__
struct GlobalConstants {
    SceneName sceneName;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

struct SceneConstants {
    CudaSphere *sphereData;
    int nSphere;
};
#endif

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).

// These need to be redefined for every .cu file
__constant__ GlobalConstants cuConstRendererParams;
__constant__ SceneConstants cudaConstSceneParams;

