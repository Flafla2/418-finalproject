#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>
#include <ctime>

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "cudaRenderer.h"
#include "image.h"
#include "sceneLoader.h"
#include "util.h"
#include "cycleTimer.h"
#include "Cubemap.h"

#define DEBUG
#include "cuda_error.h"
#include "cuda_constants.h"


////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

__constant__ GlobalConstants cuConstRendererParams;
__constant__ SceneConstants cudaConstSceneParams;


/// Clear the image, setting all pixels to the specified color rgba
/// \param r Red color component (0-1 range)
/// \param g Green color component (0-1 range)
/// \param b Blue color component (0-1 range)
/// \param a Alpha color component (0-1 range)
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // Write to global memory: As an optimization, this code uses a float4
    // store, which results in more efficient code than if it were coded as
    // four separate float stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

#define MAX_STEPS 64

__device__ __inline__ void
shadePixel(float2 pixelCenter, float4* imagePtr, glm::mat4x4 invProj,
           glm::mat4x4 invView, glm::vec3 camPos) {
    float4 ret;

    // Inverse project to get point on near clip plane (in NDC, z = -1 corresponds to the
    // near clip plane.  Also w = 1.0 in NDC)
    glm::vec4 ptView  = invProj * glm::vec4(pixelCenter.x*2-1, pixelCenter.y*2-1, -1.f, 1.f);
    // Apply homogenous coordinate from projection matrix
    ptView /= ptView.w;
    // Bring view space point into world space
    glm::vec4 ptWorld = invView * ptView;

    glm::vec3 ray = glm::normalize(glm::vec3(ptWorld) - camPos);

    float t = 0.f;
    int march;
    for (march = 0; march < MAX_STEPS; ++march) {

        glm::vec3 p = camPos + ray * t;
        float sdf = deviceSdf(p);

        if (sdf < 0.01f) {
            // hit something!
            glm::vec3 normal = deviceNormal(p);
            if(!cuConstRendererParams.perfVis) {
                if(cuConstRendererParams.lighting.rgb_exp == nullptr) {
                    const float rt1_3 = 0.5773502692f;
                    float ndotl = glm::dot(normal, -glm::vec3(rt1_3,-rt1_3,rt1_3));

                    ret.x = ndotl;
                    ret.y = ndotl;
                    ret.z = ndotl;
                    ret.w = 1.0;
                } else {
                    glm::vec2 uv = Cubemap::dir2uv(normal);

                    glm::vec4 l = sample_exp(cuConstRendererParams.lighting,uv);
                    ret.x = l.r;
                    ret.y = l.g;
                    ret.z = l.b;
                    ret.w = l.a;
                }
            }

            break;
        } else if (t > 10.0f) {
            if(!cuConstRendererParams.perfVis) {
                if (cuConstRendererParams.background.rgb_exp != nullptr) {
                    glm::vec2 uv = Cubemap::dir2uv(ray);
                    glm::vec4 res = sample_exp(cuConstRendererParams.background,uv);
                    //std::cout << res.r << "," << res.g << "," << res.b << std::endl;
                    ret.x = res.r;
                    ret.y = res.g;
                    ret.z = res.b;
                    ret.w = res.a;
                } else {
                    ret.x = (ray.x + 1) / 2;
                    ret.y = (ray.y + 1) / 2;
                    ret.z = (ray.z + 1) / 2;
                    ret.w = 1.0;
                }
            }

            break;
        } else {
            t += sdf;
        }

    }

    if (march >= MAX_STEPS && !cuConstRendererParams.perfVis) {
        if (cuConstRendererParams.background.rgb_exp != nullptr) {
            glm::vec2 uv = Cubemap::dir2uv(ray);
            glm::vec4 res = sample_exp(cuConstRendererParams.background,uv);
            //std::cout << res.r << "," << res.g << "," << res.b << std::endl;
            ret.x = res.r;
            ret.y = res.g;
            ret.z = res.b;
            ret.w = res.a;
        } else {
            ret.x = (ray.x + 1) / 2;
            ret.y = (ray.y + 1) / 2;
            ret.z = (ray.z + 1) / 2;
            ret.w = 1.0;
        }
    }

    if (cuConstRendererParams.perfVis) {
        ret.z = 1.f - float(march) / 64;
        ret.z = ret.z * ret.z * ret.z; // pow(3)
        ret.y = 0;
        ret.x = 1.0f - ret.z;
        ret.w = 1;
    }

    // Global memory write
    *imagePtr = ret;
}

__global__ void kernelRender(glm::mat4x4 invProj, glm::mat4x4 invView, glm::vec3 camPos) {

    int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;

    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;

    float4* imgPtr = (float4*)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         1.0f - invHeight * (static_cast<float>(pixelY) + 0.5f));
    shadePixel(pixelCenterNorm, imgPtr, invProj, invView, camPos);
}

////////////////////////////////////////////////////////////////////////////////////////


CudaRenderer::CudaRenderer() {
    image = nullptr;
    cudaDeviceImageData = nullptr;
}

CudaRenderer::~CudaRenderer() {
    delete image;

    if (cudaDeviceImageData) {
        cudaFree(cudaDeviceImageData);
    }
}

const Image* CudaRenderer::getImage() {

    // Need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    //printf("Copying image data from device\n");

    cudaCheckError( cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost) );

    return image;
}

void CudaRenderer::loadScene(SceneName name) {
    sceneName = name;
    scene = SceneLoader::loadSceneCuda(sceneName, emitBytecode);
}

void CudaRenderer::setup() {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce GTX 1080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA GTX 1080.\n");
        printf("---------------------------------------------------------\n");
    }
    
    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy
    cudaCheckError(
        cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height)
    );

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.imageData = cudaDeviceImageData;
    params.perfVis = perfVis;

    params.background = PngImage();
    params.background.texture = nullptr;
    params.background.w = background.w;
    params.background.h = background.h;
    float *rgb_exp_device;
    if(background.rgb_exp) {
        cudaCheckError(
                cudaMalloc(&rgb_exp_device, sizeof(float) * 3 * background.w * background.h)
        );
        cudaCheckError(
                cudaMemcpy(rgb_exp_device, background.rgb_exp, sizeof(float) * 3 * background.w * background.h, cudaMemcpyHostToDevice)
        );
        params.background.rgb_exp = rgb_exp_device;
    } else {
        params.background.rgb_exp = nullptr;
    }
    rgb_exp_device = nullptr;

    params.lighting = PngImage();
    params.lighting.texture = nullptr;
    params.lighting.w = lighting.w;
    params.lighting.h = lighting.h;
    if(lighting.rgb_exp) {
        cudaCheckError(
                cudaMalloc(&rgb_exp_device, sizeof(float) * 3 * lighting.w * lighting.h)
        );
        cudaCheckError(
                cudaMemcpy(rgb_exp_device, lighting.rgb_exp, sizeof(float) * 3 * lighting.w * lighting.h, cudaMemcpyHostToDevice)
        );
        params.lighting.rgb_exp = rgb_exp_device;
    } else {
        params.lighting.rgb_exp = nullptr;
    }


    cudaCheckError(
        cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants))
    );

    scene->initCudaData();
}


/// Allocate buffer the renderer will render into.
void CudaRenderer::allocOutputImage(int width, int height) {
    delete image;
    image = new Image(width, height);
}

/// Clear the renderer's target image.  The state of the image after
/// the clear depends on the scene being rendered.
void CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    cudaCheckError(
        cudaDeviceSynchronize()
    );
}


void CudaRenderer::advanceAnimation() {
//    dim3 blockDim(256, 1);
//    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);
//
//    cudaDeviceSynchronize();
}

void CudaRenderer::render() {
    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
            (image->width + blockDim.x - 1) / blockDim.x,
            (image->height + blockDim.y - 1) / blockDim.y);

    static double begin = CycleTimer::currentSeconds();
    double cur = CycleTimer::currentSeconds();

    double elapsed_secs = cur - begin;
    //printf("Time elapsed: %f\n", elapsed_secs);

    glm::vec3 camPos(glm::sin(elapsed_secs) * 5.0f, 0.f, glm::cos(elapsed_secs) * 5.0f);
    glm::vec3 camLook(0.f, 0.f, 0.f);
    glm::vec3 camUp(0.f, 1.f, 0.f);

    static float aspect = float(image->width) / image->height;

    glm::mat4x4 invView = glm::inverse(glm::lookAt(camPos, camLook, camUp));
    glm::mat4x4 invProj = glm::inverse(glm::perspective(30.0f, aspect, 0.3f, 200.0f));

    kernelRender<<<gridDim, blockDim>>>(invProj, invView, camPos);
    cudaCheckError(
        cudaDeviceSynchronize()
    );
}
