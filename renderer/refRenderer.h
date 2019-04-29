#ifndef __REF_RENDERER_H__
#define __REF_RENDERER_H__

#include "renderer.h"
#include "RefScene.h"

class RefRenderer : public Renderer {

private:

    Image* image;
    SceneName sceneName;
    RefScene *scene;

public:

    RefRenderer();
    ~RefRenderer() override;

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
