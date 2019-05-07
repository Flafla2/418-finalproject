#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "renderer.h"
#include "CudaScene.h"

class CudaRenderer : public Renderer {

private:
    Image* image;
    SceneName sceneName;
    CudaScene *scene;

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

    bool emitBytecode = false;
};

#endif
