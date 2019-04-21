#ifndef __REF_RENDERER_H__
#define __REF_RENDERER_H__

#include "renderer.h"
#include "Scene.h"

class RefRenderer : public Renderer {

private:

    Image* image;
    SceneName sceneName;
    Scene *scene;

public:

    RefRenderer();
    virtual ~RefRenderer();

    const Image* getImage();

    void setup();

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void render();

    void shadePixel(
        float pixelCenterX, float pixelCenterY,
        float* pixelData);
};


#endif
