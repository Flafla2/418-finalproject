
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

// Sphere //

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

// Box //

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

// Torus //

struct CudaTorus {
    CudaTorus() = default; // needed for trivial type
    explicit CudaTorus(glm::vec3 center = glm::vec3(0,0,0), glm::vec3 t = glm::vec3(1,1,1)) :
        center(center), radius(radius), thickness(thickness) {}

    glm::vec3 center;
    float radius;
    float thickness;
};

__device__ __host__
float TorusSDF(CudaTorus const& torus, glm::vec3 p);

static_assert(sizeof(CudaTorus) == sizeof(glm::vec3) + 2 * sizeof(float), "CudaTorus is packed");

// Cylinder //

struct CudaCylinder {
    CudaCylinder() = default; // needed for trivial type
    explicit CudaCylinder(glm::vec3 center = glm::vec3(0,0,0), glm::vec3 dim = glm::vec3(1,1,1)) :
        center(center), radius(radius) {}

    glm::vec3 center;
    float radius;
};

__device__ __host__
float CylinderSDF(CudaCylinder const& cylinder, glm::vec3 p);

static_assert(sizeof(CudaCylinder) == 2 * sizeof(glm::vec3), "CudaCylinder is packed");

// Cone //

struct CudaCone {
    CudaCone() = default; // needed for trivial type
    explicit CudaCone(glm::vec3 center = glm::vec3(0,0,0), glm::vec2 dir = glm::vec2(0,1)) :
        center(center), dir(dir) {}

    glm::vec3 center;
    glm::vec2 dir; // normalized direction of cone
};

__device__ __host__
float ConeSDF(CudaCone const& cone, glm::vec3 p);

static_assert(sizeof(CudaCone) == sizeof(glm::vec3) + sizeof(glm::vec2), "CudaCone is packed");

// Plane //

struct CudaPlane {
    CudaPlane() = default; // needed for trivial type
    explicit CudaPlane(glm::vec3 center = glm::vec3(0,0,0), glm::vec3 normal = glm::vec3(0,1,0), float offset = 0.0f) :
        center(center), normal(normal), offset(offset) {}

    glm::vec3 center;
    glm::vec3 normal; // normal of plane
    float offset; // offset along normal from origin
};

__device__ __host__
float PlaneSDF(CudaPlane const& plane, glm::vec3 p);

static_assert(sizeof(CudaPlane) == sizeof(glm::vec3) + sizeof(glm::vec4), "CudaPlane is packed");


#endif //RAYMARCHER_CUDAPRIMITIVE_H
