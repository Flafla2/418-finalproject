#ifndef __SCENE_LOADER_H__
#define __SCENE_LOADER_H__

#include "renderer.h"

#include "RefScene.h"

#if WITH_CUDA && defined(__CUDACC__)
#include "CudaScene.h"
#endif

namespace SceneLoader {
#if WITH_CUDA && defined(__CUDACC__)
    CudaScene *loadSceneCuda(SceneName sceneName);
#endif
    RefScene *loadSceneRef(SceneName sceneName);
}

#endif
