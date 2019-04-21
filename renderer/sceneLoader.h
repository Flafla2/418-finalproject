#ifndef __SCENE_LOADER_H__
#define __SCENE_LOADER_H__

#include "renderer.h"
#include "Scene.h"

namespace SceneLoader {
    Scene *loadScene(SceneName sceneName);
}

#endif
