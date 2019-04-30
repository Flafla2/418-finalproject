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
    CudaSphere *sphereData;
    int nSphere;
};

// Define the constants in the first .cu file that is included in
__constant__ GlobalConstants cuConstRendererParams;
__constant__ SceneConstants cudaConstSceneParams;

#else

// Refer to the original file with an extern otherwise
// See https://stackoverflow.com/questions/7959174/nvcc-combine-extern-and-constant
//     ^^ Second answer, because the primary was written pre-cuda 5.0
extern __constant__ GlobalConstants cuConstRendererParams;
extern __constant__ SceneConstants cudaConstSceneParams;

#endif


