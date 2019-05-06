
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

#include "RefPrimitive.h"

// CudaPrimitives need to be C++ POD types (same memory layout as C struct)
// https://stackoverflow.com/questions/4178175/what-are-aggregates-and-pods-and-how-why-are-they-special/7189821#7189821
// Note: We can use polymorphism here AS LONG AS the types are "trivial" (see above link)

// Primitive //
struct CudaPrimitive {
    CudaPrimitive() = default; // needed for trivial type
    explicit CudaPrimitive(glm::mat4x4 const& world2local) : world2local(world2local) {}

    /// World space to local space transformation matrix.  Note that this matrix is the inverse
    /// of a traditional transformation matrix (which goes from local to world space)
    glm::mat4x4 world2local;
};

static_assert(sizeof(CudaPrimitive) == sizeof(glm::mat4x4), "CudaPrimitive is packed");

// Sphere //

struct CudaSphere : CudaPrimitive {
    CudaSphere() = default; // needed for trivial type
    explicit CudaSphere(
            float radius = 1.f,
            glm::mat4x4 const& world2local = glm::mat4x4(1.f)) :
        radius(radius), CudaPrimitive(world2local) {}

    CudaSphere(RefSphere const* ref) : radius(ref->radius), CudaPrimitive(ref->world2local) {}

    float radius;
};

__device__ __host__
float SphereSDF(CudaSphere const& sphere, glm::vec3 p);

static_assert(sizeof(CudaSphere) == sizeof(CudaPrimitive) + sizeof(float), "CudaSphere is packed");

// Box //

struct CudaBox : CudaPrimitive {
    CudaBox() = default; // needed for trivial type
    explicit CudaBox(
            glm::vec3 dim = glm::vec3(1,1,1),
            glm::mat4x4 const& world2local = glm::mat4x4(1.f)) :
        dim(dim), CudaPrimitive(world2local) {}

    CudaBox(RefBox const* ref) : dim(ref->dim), CudaPrimitive(ref->world2local) {}

    glm::vec3 dim;
};

__device__ __host__
float BoxSDF(CudaBox const& box, glm::vec3 p);

static_assert(sizeof(CudaBox) == sizeof(CudaPrimitive) + sizeof(glm::vec3), "CudaBox is packed");

// Torus //

struct CudaTorus : CudaPrimitive {
    CudaTorus() = default; // needed for trivial type
    explicit CudaTorus(
            float radius = 1.f,
            float thickness = .2f,
            glm::mat4x4 const& world2local = glm::mat4x4(1.f)) :
        radius(radius), thickness(thickness), CudaPrimitive(world2local) {}

    CudaTorus(RefTorus const* ref) : radius(ref->radius), thickness(ref->thickness), CudaPrimitive(ref->world2local) {}

    float radius;
    float thickness;
};

__device__ __host__
float TorusSDF(CudaTorus const& torus, glm::vec3 p);

static_assert(sizeof(CudaTorus) == sizeof(CudaPrimitive) + 2 * sizeof(float), "CudaTorus is packed");

// Cylinder //

struct CudaCylinder : CudaPrimitive {
    CudaCylinder() = default; // needed for trivial type
    explicit CudaCylinder(
            float radius,
            glm::mat4x4 const& world2local = glm::mat4x4(1.f)) :
        radius(radius), CudaPrimitive(world2local) {}

    CudaCylinder(RefCylinder const* ref) : radius(ref->radius), CudaPrimitive(ref->world2local) {}

    float radius;
};

__device__ __host__
float CylinderSDF(CudaCylinder const& cylinder, glm::vec3 p);

static_assert(sizeof(CudaCylinder) == sizeof(CudaPrimitive) + sizeof(float), "CudaCylinder is packed");

// Cone //

struct CudaCone : CudaPrimitive {
    CudaCone() = default; // needed for trivial type
    explicit CudaCone(
            glm::vec2 dir = glm::vec2(0,1),
            glm::mat4x4 const& world2local = glm::mat4x4(1.f)) :
        dir(dir), CudaPrimitive(world2local) {}

    CudaCone(RefCone const* ref) : dir(ref->dir), CudaPrimitive(ref->world2local) {}

    glm::vec2 dir; // normalized direction of cone
};

__device__ __host__
float ConeSDF(CudaCone const& cone, glm::vec3 p);

static_assert(sizeof(CudaCone) == sizeof(CudaPrimitive) + sizeof(glm::vec2), "CudaCone is packed");

// Plane //

struct CudaPlane : CudaPrimitive {
    CudaPlane() = default; // needed for trivial type
    explicit CudaPlane(
            glm::vec3 normal = glm::vec3(0,1,0),
            float offset = 0.0f,
            glm::mat4x4 const& world2local = glm::mat4x4(1.f)) :
        normal(normal), offset(offset), CudaPrimitive(world2local) {}

    CudaPlane(RefPlane const* ref) : normal(ref->normal), offset(ref->offset), CudaPrimitive(ref->world2local) {}

    glm::vec3 normal; // normal of plane
    float offset; // offset along normal from origin
};

__device__ __host__
float PlaneSDF(CudaPlane const& plane, glm::vec3 p);

static_assert(sizeof(CudaPlane) == sizeof(CudaPrimitive) + sizeof(glm::vec3) + sizeof(float), "CudaPlane is packed");


#endif //RAYMARCHER_CUDAPRIMITIVE_H
