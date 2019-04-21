#ifndef __RENDERER_H__
#define __RENDERER_H__

struct Image;

typedef enum {
    TEST_SCENE
} SceneName;

class Renderer {

public:

    virtual ~Renderer() { };

    virtual const Image* getImage() = 0;

    virtual void setup() = 0;

    virtual void loadScene(SceneName name) = 0;

    virtual void allocOutputImage(int width, int height) = 0;

    virtual void clearImage() = 0;

    virtual void advanceAnimation() = 0;

    virtual void render() = 0;

};


#endif
