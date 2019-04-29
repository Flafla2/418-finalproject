
#ifndef RAYMARCHER_CUDAPRIMITIVE_H
#define RAYMARCHER_CUDAPRIMITIVE_H

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#endif

#include <glm/vec3.hpp>

class CudaPrimitive {
public:
    /// Evaluates a Signed Distance field for this primitive.
    /// \param p Point to evaluate SDF at
    /// \return If \c p is outside this primitive, returns the shortest distance from p to the
    ///         outside of this primitive.  If \c p is inside the primitive, returns the *negative*
    ///         distance to the outside of this primitive.
    __device__ __host__
    virtual float sdf(glm::vec3 p) const = 0;
};

class CudaSphere : public CudaPrimitive {
public:
    explicit CudaSphere(glm::vec3 center = glm::vec3(0,0,0), float radius = 1.f) :
        center(center), radius(radius) {}

    __device__ __host__
    float sdf(glm::vec3 p) const override;

    glm::vec3 center;
    float radius;
};

class CudaBox : public CudaPrimitive {
public:
    explicit CudaBox(glm::vec3 center = glm::vec3(0,0,0), glm::vec3 dim = glm::vec3(1,1,1)) : center(center), dim(dim) {}

    __device__ __host__
    float sdf(glm::vec3 p) const override;

    glm::vec3 center;
    glm::vec3 dim;
};


#endif //RAYMARCHER_CUDAPRIMITIVE_H
