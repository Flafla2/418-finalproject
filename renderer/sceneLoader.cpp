#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <functional>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "sceneLoader.h"
#include "util.h"

#include "RefPrimitive.h"

glm::mat4x4 invTRS(
        glm::vec3 const& translate,
        glm::quat const& rotate = glm::identity<glm::quat>(),
        glm::vec3 const& scale = glm::vec3(1.f,1.f,1.f)) {
    glm::mat4x4 ret = glm::identity<glm::mat4x4>();
    ret = glm::scale(ret, glm::vec3(1.f/scale.x,1.f/scale.y,1.f/scale.z));
    ret *= glm::mat4_cast(-rotate);
    ret = glm::translate(ret, -translate);

    return ret;
}

static std::vector<RefPrimitive *> refPrimsFromScene(SceneName sceneName) {
    std::vector<RefPrimitive *> prims;
    if (sceneName == TEST_SCENE) {
        prims.push_back(new RefSphere(1.f, invTRS(glm::vec3(-1.5f,-1,0))));
        prims.push_back(new RefTorus(1.f, .2f, invTRS(glm::vec3( 1.5f,-1,0))));

        auto cone = new RefCone(glm::normalize(glm::vec2(1,1)));
        auto bounds = new RefBox(glm::vec3(1,1,1));
        prims.push_back(new RefCombine(cone, bounds, ISECT, invTRS(glm::vec3(0,1,0))));
    } else {
        fprintf(stderr, "Error: cann't load scene (unknown scene)\n");
    }
    return prims;
}

#if WITH_CUDA
CudaScene *SceneLoader::loadSceneCuda(SceneName sceneName)
{
    auto prims = refPrimsFromScene(sceneName);
    printf("Loaded scene\n");
    CudaScene *scene = new CudaScene(prims);
    printf("CUDA bytecode generated\n");
    return scene;
}
#endif

RefScene *SceneLoader::loadSceneRef(SceneName sceneName)
{
    auto prims = refPrimsFromScene(sceneName);

    printf("Loaded scene\n");
    return new RefScene(prims);
}