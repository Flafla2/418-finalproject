#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>

#include "refRenderer.h"
#include "image.h"
#include "sceneLoader.h"
#include "util.h"

RefRenderer::RefRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;
}

RefRenderer::~RefRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }
}

const Image*
RefRenderer::getImage() {
    return image;
}

void
RefRenderer::setup() {
    // nothing to do here
}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
RefRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
RefRenderer::clearImage() {
    image->clear(1.f, 1.f, 1.f, 1.f);
}

void
RefRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    SceneLoader::loadScene(sceneName);
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
RefRenderer::advanceAnimation() {
    
}

// shadePixel --
//
// Computes the contribution of the specified circle to the
// given pixel.  All values are provided in normalized space, where
// the screen spans [0,2]^2.  The color/opacity of the circle is
// computed at the pixel center.
void
RefRenderer::shadePixel(
    float pixelCenterX, float pixelCenterY,
    float* pixelData)
{
    pixelData[0] = pixelCenterX;
    pixelData[1] = pixelCenterY;
    pixelData[2] = 0.0;
    pixelData[3] = 1.0;
}

void
RefRenderer::render() {
    float invWidth = 1.f / image->width;
    float invHeight = 1.f / image->height;

    // for each pixel in the bounding box, determine the circle's
    // contribution to the pixel.  The contribution is computed in
    // the function shadePixel.  Since the circle does not fill
    // the bounding box entirely, not every pixel in the box will
    // receive contribution.
    for (int pixelY = 0; pixelY < image->height; pixelY++) {

        // pointer to pixel data
        float* imgPtr = &image->data[4 * (pixelY * image->width)];

        for (int pixelX = 0; pixelX < image->width; pixelX++) {

            // When "shading" the pixel ("shading" = computing the
            // circle's color and opacity at the pixel), we treat
            // the pixel as a point at the center of the pixel.
            // We'll compute the color of the circle at this
            // point.  Note that shading math will occur in the
            // normalized [0,1]^2 coordinate space, so we convert
            // the pixel center into this coordinate space prior
            // to calling shadePixel.
            float pixelCenterNormX = invWidth * (static_cast<float>(pixelX) + 0.5f);
            float pixelCenterNormY = invHeight * (static_cast<float>(pixelY) + 0.5f);
            shadePixel(pixelCenterNormX, pixelCenterNormY, imgPtr);
            imgPtr += 4;
        }
    }
}

void RefRenderer::dumpParticles(const char* filename) {

    FILE* output = fopen(filename, "w");

    fprintf(output, "%d\n", numCircles);
    for (int i=0; i<numCircles; i++) {
        fprintf(output, "%f %f %f   %f %f %f   %f\n",
                position[3*i+0], position[3*i+1], position[3*i+2],
                velocity[3*i+0], velocity[3*i+1], velocity[3*i+2],
                radius[i]);
    }
    fclose(output);

}
