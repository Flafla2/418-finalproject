
#ifndef RAYMARCHER_PRIMITIVE_H
#define RAYMARCHER_PRIMITIVE_H

#include <glm/vec3.hpp>

class Primitive {
public:
    /// Evaluates a Signed Distance field for this primitive.
    /// \param p Point to evaluate SDF at
    /// \return If \c p is outside this primitive, returns the shortest distance from p to the
    ///         outside of this primitive.  If \c p is inside the primitive, returns the *negative*
    ///         distance to the outside of this primitive.
    virtual inline float sdf(glm::vec3 p) const = 0;
};

class Sphere : public Primitive {
public:
    explicit Sphere(glm::vec3 center = glm::vec3(0,0,0), float radius = 1.f) :
        center(center), radius(radius) {}

    inline float sdf(glm::vec3 p) const override;

    glm::vec3 center;
    float radius;
};


#endif //RAYMARCHER_PRIMITIVE_H
