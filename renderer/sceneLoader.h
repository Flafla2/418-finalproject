#ifndef __SCENE_LOADER_H__
#define __SCENE_LOADER_H__

#include "renderer.h"

#include "Scene.h"
#include "RefScene.h"
#if WITH_CUDA
#include "CudaScene.h"
#endif

namespace SceneLoader {
#if WITH_CUDA
    CudaScene *loadSceneCuda(SceneName sceneName);
#endif
    RefScene *loadSceneRef(SceneName sceneName);
}

#endif
