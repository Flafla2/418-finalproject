
#ifndef RAYMARCHER_REFPRIMITIVE_H
#define RAYMARCHER_REFPRIMITIVE_H

#include <glm/vec3.hpp>

class RefPrimitive {
public:
    /// Evaluates a Signed Distance field for this primitive.
    /// \param p Point to evaluate SDF at
    /// \return If \c p is outside this primitive, returns the shortest distance from p to the
    ///         outside of this primitive.  If \c p is inside the primitive, returns the *negative*
    ///         distance to the outside of this primitive.
    virtual float sdf(glm::vec3 p) const = 0;
};

class RefSphere : public RefPrimitive {
public:
    explicit RefSphere(glm::vec3 center = glm::vec3(0,0,0), float radius = 1.f) :
        center(center), radius(radius) {}

    float sdf(glm::vec3 p) const override;

    glm::vec3 center;
    float radius;
};

class RefBox : public RefPrimitive {
public:
    explicit RefBox(glm::vec3 center = glm::vec3(0,0,0), glm::vec3 dim = glm::vec3(1,1,1)) : center(center), dim(dim) {}

    float sdf(glm::vec3 p) const override;

    glm::vec3 center;
    glm::vec3 dim;
};

class RefTorus : public RefPrimitive {
public:
    explicit RefTorus(glm::vec3 center = glm::vec3(0,0,0), glm::vec3 t = glm::vec3(1,1,1) : center(center), t(t)){}

    float sdf(glm::vec3 p) const override;

    glm::vec3 center;
    glm::vec3 t;
};

class RefCylinder : public RefPrimitive {
public:
    explicit RefCylinder(glm::vec3 center = glm::vec3(0,0,0), glm::vec3 dim = glm::vec3(1,1,1) : center(center), dim(dim)) {}

    float sdf(glm::vec3 p) const override;

    glm::vec3 center;
    glm::vec3 dim;
};

class RefCone : public RefPrimitive {
public:
    explicit RefCone(glm::vec3 center = glm::vec3(0,0,0), glm::vec2 dim = glm::vec2(1,1) : center(center), dim(dim)) {}

    float sdf(glm::vec3 p) const override;

    glm::vec3 center;
    glm::vec3 dim;
};

class RefPlane : public RefPrimitive {
public:
    explicit RefPlane(glm::vec3 center = glm::vec3(0,0,0), glm::vec4 dim = glm::vec4(1,1,1,1) : center(center), dim(dim))

    float sdf(glm::vec3 p) const override;

    glm::vec3 center;
    glm::vec4 dim;
};


#endif //RAYMARCHER_REFPRIMITIVE_H
