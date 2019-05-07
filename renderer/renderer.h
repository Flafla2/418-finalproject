#ifndef __RENDERER_H__
#define __RENDERER_H__

#include "Cubemap.h"

struct Image;

typedef enum {
    INVALID,
    TEST_SCENE1,
    TEST_SCENE2,
    TEST_SCENE3,
    TEST_SCENE4,
    TEST_SCENE5
} SceneName;

class Renderer {

public:

    virtual ~Renderer() = default;

    virtual const Image* getImage() = 0;

    virtual void setup() = 0;

    virtual void loadScene(SceneName name) = 0;

    virtual void allocOutputImage(int width, int height) = 0;

    virtual void clearImage() = 0;

    virtual void advanceAnimation() = 0;

    virtual void render() = 0;

    PngImage lighting; // Lighting using IBL
    PngImage background; // Background texture
};


#endif
