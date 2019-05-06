#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>
#include <ctime>
#include <iostream>

#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include "cycleTimer.h"
#include "refRenderer.h"
#include "image.h"
#include "sceneLoader.h"
#include "util.h"

RefRenderer::RefRenderer() {
    image = nullptr;
    scene = nullptr;
    sceneName = INVALID;
}

RefRenderer::~RefRenderer() {
    delete image;
}

const Image* RefRenderer::getImage() {
    return image;
}

void RefRenderer::setup() {
    // nothing to do here
}

/// Allocate buffer the renderer will render into.
/// \param width Image width
/// \param height Image height
void RefRenderer::allocOutputImage(int width, int height) {
    delete image; // does nothing if image == nullptr

    image = new Image(width, height);
}


/// Clear's the renderer's target image.  The state of the image after
/// the clear depends on the scene being rendered.
void RefRenderer::clearImage() {
    image->clear(1.f, 1.f, 1.f, 1.f);
}

/// Loads the scene with the given scene name
/// \param name SceneName to use
void RefRenderer::loadScene(SceneName name) {
    sceneName = name;
    scene = SceneLoader::loadSceneRef(sceneName);
}

/// Advance the simulation one time step.
void RefRenderer::advanceAnimation() {
    
}

/// Shades the scene at the specified pixel, given the pixel coordinate in clip space.
/// \param pixelCenterX X coordinate of center of pixel in [0,1] range
/// \param pixelCenterY Y coordinate of center of pixel in [0,1] range
/// \param pixelData Pointer to data that will be written
/// \param invProj Inverse of camera projection matrix
/// \param invView Inverse of camera view matrix
/// \param camPos Camera position
void RefRenderer::shadePixel(
    float pixelCenterX, float pixelCenterY,
    float* pixelData, glm::mat4x4 const& invProj,
    glm::mat4x4 const& invView, glm::vec3 const& camPos)
{
    // Inverse project to get point on near clip plane (in NDC, z = -1 corresponds to the
    // near clip plane.  Also w = 1.0 in NDC)
    glm::vec4 ptView  = invProj * glm::vec4(pixelCenterX*2-1, pixelCenterY*2-1, -1.f, 1.f);
    // Apply homogenous coordinate from projection matrix
    ptView /= ptView.w;
    // Bring view space point into world space
    glm::vec3 ptWorld = glm::vec3(invView * ptView);

    glm::vec3 ray = glm::normalize(ptWorld - camPos);

    float t = 0.f;
    for (int march = 0; march < 64; ++march) {

        glm::vec3 p = camPos + ray * t;
        float sdf = scene->sdf(p);

        if (sdf < 0.01f) {
            // hit something!
            glm::vec3 normal = scene->normal(p);
            const float rt1_3 = 0.5773502692f;
            float ndotl = glm::dot(normal, -glm::vec3(rt1_3,-rt1_3,rt1_3));

            pixelData[0] = pixelData[1] = pixelData[2] = ndotl;
            pixelData[3] = 1.0f;

            return;
        } else if (t > 10.0f) {
            break;
        } else {
            t += sdf;
        }

    }

    pixelData[0] = (ray.x+1)/2;
    pixelData[1] = (ray.y+1)/2;
    pixelData[2] = (ray.z+1)/2;
    pixelData[3] = 1.0;
}

void RefRenderer::render() {
    float invWidth = 1.f / image->width;
    float invHeight = 1.f / image->height;

    static double begin = CycleTimer::currentSeconds();
    double cur = CycleTimer::currentSeconds();

    double elapsed_secs = cur - begin;

    glm::vec3 camPos(glm::sin(elapsed_secs) * 5.0f, 0.f, glm::cos(elapsed_secs) * 5.0f);
    glm::vec3 camLook(0.f, 0.f, 0.f);
    glm::vec3 camUp(0.f, 1.f, 0.f);

    static float aspect = float(image->width) / image->height;

    glm::mat4x4 invView = glm::inverse(glm::lookAt(camPos, camLook, camUp));
    static glm::mat4x4 invProj = glm::inverse(glm::perspective(30.0f, aspect, 0.3f, 200.0f));

    for (int pixelY = 0; pixelY < image->height; pixelY++) {

        // pointer to pixel data
        float *imgPtr = &image->data[4 * (pixelY * image->width)];

        for (int pixelX = 0; pixelX < image->width; pixelX++) {

            float pixelCenterNormX = invWidth * (static_cast<float>(pixelX) + 0.5f);
            float pixelCenterNormY = 1.0f - invHeight * (static_cast<float>(pixelY) + 0.5f);
            shadePixel(pixelCenterNormX, pixelCenterNormY, imgPtr, invProj, invView, camPos);
            imgPtr += 4;
        }
    }
}