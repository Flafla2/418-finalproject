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


////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {
    SceneName sceneName;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

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
            const float rt1_3 = 0.5773502692f;
            float ndotl = glm::dot(normal, -glm::vec3(rt1_3,-rt1_3,rt1_3));
            printf("Hit.  Center: (%d, %d) P: (%d, %d, %d) Normal: (%d, %d, %d)", pixelCenter.x, pixelCenter.y, p.x, p.y, p.z, normal.x, normal.y, normal.z);

            ret.x = ret.y = ret.z = ndotl;
            ret.w = 1.0f;

            break;
        } else if (t > 10.0f) {
            ret.x = (ray.x+1)/2;
            ret.y = (ray.y+1)/2;
            ret.z = (ray.z+1)/2;
            ret.w = 1.0;

            break;
        } else {
            t += sdf;
        }

    }

    if (march >= MAX_STEPS) {
        ret.x = (ray.x+1)/2;
        ret.y = (ray.y+1)/2;
        ret.z = (ray.z+1)/2;
        ret.w = 1.0;
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
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));
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

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void CudaRenderer::loadScene(SceneName name) {
    sceneName = name;
    scene = SceneLoader::loadSceneCuda(sceneName);
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
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

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

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

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
    
    cudaDeviceSynchronize();
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

    static std::clock_t begin = clock();
    std::clock_t cur = clock();

    double elapsed_secs = double(cur - begin) / CLOCKS_PER_SEC;

    glm::vec3 camPos(glm::sin(elapsed_secs) * 5.0f, 0.f, glm::cos(elapsed_secs) * 5.0f);
    glm::vec3 camLook(0.f, 0.f, 0.f);
    glm::vec3 camUp(0.f, 1.f, 0.f);

    static float aspect = float(image->width) / image->height;

    glm::mat4x4 invView = glm::inverse(glm::lookAt(camPos, camLook, camUp));
    glm::mat4x4 invProj = glm::inverse(glm::perspective(30.0f, aspect, 0.3f, 200.0f));

    kernelRender<<<gridDim, blockDim>>>(invProj, invView, camPos);
    cudaDeviceSynchronize();
}
