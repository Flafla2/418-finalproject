#ifndef RAYMARCHER_CUDASCENE_H
#define RAYMARCHER_CUDASCENE_H

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#endif

#include <vector>

#include "RefPrimitive.h"
#include "CudaPrimitive.h"

namespace CudaOpcodes {
    // Args: CudaSphere
    // Pushes sdf to float stack
    const char Sphere = 0x01;
    // Args: CudaBox
    // Pushes sdf to float stack
    const char Box = 0x02;
    // Args: CudaTorus
    // Pushes sdf to float stack
    const char Torus = 0x03;
    // Args: CudaCylinder
    // Pushes sdf to float stack
    const char Cylinder = 0x04;
    // Args: CudaCone
    // Pushes sdf to float stack
    const char Cone = 0x05;
    // Args: CudaPlane
    // Pushes sdf to float stack
    const char Plane = 0x06;

    // Transformation instructions take in one or more SDFs as input.

    // Args: byte CombineOp
    // CombineOp: 0x00 for union, 0x01 for diff, 0x02 for isect
    // Consumes 2 sdfs from float stack and pushes 1 combined sdf
    const char Combine = 0x07;
    // Args: byte CombineOp, float smoothing
    // CombineOp: 0x00 for union, 0x01 for diff, 0x02 for isect
    // smoothing: smoothing factor in world units
    // Consumes 2 sdfs from float stack and pushes 1 combined sdf
    const char CombineSmooth = 0x08;

    std::vector<char> refToBytecode(std::vector<RefPrimitive const*> prims);
}

class CudaScene {
public:
    explicit CudaScene(std::vector<RefPrimitive const*> const& prims);
    ~CudaScene();

    void initCudaData();
private:
    std::vector<char> bytecode;

    char *cudaDeviceBytecode;
};

#if defined(__CUDACC__)
__device__ float deviceSdf(glm::vec3 p);
__device__ glm::vec3 deviceNormal(glm::vec3 p);
#endif

#endif //RAYMARCHER_CUDASCENE_H
