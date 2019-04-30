
#ifndef RAYMARCHER_CUDAPRIMITIVE_H
#define RAYMARCHER_CUDAPRIMITIVE_H

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#else
#define __device__
#define __host__
#endif

#include <glm/vec3.hpp>

// CudaPrimitives need to be C++ POD types (same memory layout as C struct)
// https://stackoverflow.com/questions/4178175/what-are-aggregates-and-pods-and-how-why-are-they-special/7189821#7189821
// Note: We can use polymorphism here AS LONG AS the types are "trivial" (see above link)

struct CudaSphere {
    CudaSphere() = default; // needed for trivial type
    explicit CudaSphere(glm::vec3 center = glm::vec3(0,0,0), float radius = 1.f) :
        center(center), radius(radius) {}

    glm::vec3 center;
    float radius;
};

__device__ __host__
float SphereSDF(CudaSphere const& sphere, glm::vec3 p);

static_assert(sizeof(CudaSphere) == sizeof(glm::vec3) + sizeof(float), "CudaSphere is packed");

struct CudaBox {
    CudaBox() = default; // needed for trivial type
    explicit CudaBox(glm::vec3 center = glm::vec3(0,0,0), glm::vec3 dim = glm::vec3(1,1,1)) :
        center(center), dim(dim) {}

    glm::vec3 center;
    glm::vec3 dim;
};

__device__ __host__
float BoxSDF(CudaBox const& box, glm::vec3 p);

static_assert(sizeof(CudaBox) == 2 * sizeof(glm::vec3), "CudaBox is packed");


#endif //RAYMARCHER_CUDAPRIMITIVE_H
