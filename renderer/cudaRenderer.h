#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "renderer.h"
#include "Scene.h"

class CudaRenderer : public Renderer {

private:
    Image* image;
    SceneName sceneName;
    Scene *scene;

    float* cudaDeviceImageData;

public:

    CudaRenderer();
    ~CudaRenderer();

    const Image* getImage() override;

    void setup() override;

    void loadScene(SceneName name) override;

    void allocOutputImage(int width, int height) override;

    void clearImage() override;

    void advanceAnimation() override;

    void render() override;

    void shadePixel(
            float pixelCenterX, float pixelCenterY,
            float* pixelData, glm::mat4x4 invProj,
            glm::mat4x4 invView, glm::vec3 camPos);
};

#endif
