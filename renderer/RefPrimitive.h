
#ifndef RAYMARCHER_REFPRIMITIVE_H
#define RAYMARCHER_REFPRIMITIVE_H

#include <glm/glm.hpp>

// https://stackoverflow.com/questions/18290523/is-a-default-move-constructor-equivalent-to-a-member-wise-move-constructor
#define DEFAULT_MOVE(ClassName) \
    ClassName(const ClassName& mE)            = default;\
    ClassName(ClassName&& mE)                 = default;\
    ClassName& operator=(const ClassName& mE) = default;\
    ClassName& operator=(ClassName&& mE)      = default;

class RefPrimitive {
public:
    explicit RefPrimitive(glm::mat4x4 const& world2local = glm::mat4x4(1.f))
        : world2local(world2local) {}

    // Probably not needed but whatever
    DEFAULT_MOVE(RefPrimitive)

    /// Evaluates a Signed Distance field for this primitive.
    /// \param p Point to evaluate SDF at
    /// \return If \c p is outside this primitive, returns the shortest distance from p to the
    ///         outside of this primitive.  If \c p is inside the primitive, returns the *negative*
    ///         distance to the outside of this primitive.
    virtual float sdf(glm::vec3 const& p) const = 0;

    /// World space to local space transformation matrix.  Note that this matrix is the inverse
    /// of a traditional transformation matrix (which goes from local to world space)
    glm::mat4x4 world2local;
};

enum RefCombineOperation {
    UNION, ISECT, DIFF
};

// Combines two primitives with a smoothing factor
class RefCombineSmooth : public RefPrimitive {
public:
    explicit RefCombineSmooth(
            RefPrimitive const *const p1,
            RefPrimitive const *const p2,
            float smoothing,
            RefCombineOperation op = UNION,
            glm::mat4x4 const& world2local = glm::mat4x4(1.f)) :
            p1(p1), p2(p2), op(op), smoothing(smoothing),
            RefPrimitive(world2local) {}
    DEFAULT_MOVE(RefCombineSmooth)

    float sdf(glm::vec3 const& p) const override;

    float smoothing; // in world units
    RefPrimitive const * p1;
    RefPrimitive const * p2;
    RefCombineOperation op;
};

// Combines two primitives
class RefCombine : public RefPrimitive {
public:
    explicit RefCombine(
            RefPrimitive const *const p1,
            RefPrimitive const *const p2,
            RefCombineOperation op = UNION,
            glm::mat4x4 const& world2local = glm::mat4x4(1.f)) :
        p1(p1), p2(p2), op(op), RefPrimitive(world2local) {}

    DEFAULT_MOVE(RefCombine)

    float sdf(glm::vec3 const& p) const override;

    RefPrimitive const * p1;
    RefPrimitive const * p2;
    RefCombineOperation op;
};

class RefSphere : public RefPrimitive {
public:
    explicit RefSphere(
            float radius = 1.f,
            glm::mat4x4 const& world2local = glm::mat4x4(1.f)) :
        radius(radius), RefPrimitive(world2local) {}

    DEFAULT_MOVE(RefSphere)

    float sdf(glm::vec3 const& p) const override;

    float radius;
};

class RefBox : public RefPrimitive {
public:
    explicit RefBox(
            glm::vec3 dim = glm::vec3(1,1,1),
            glm::mat4x4 const& world2local = glm::mat4x4(1.f)) :
        dim(dim), RefPrimitive(world2local) {}

    DEFAULT_MOVE(RefBox)

    float sdf(glm::vec3 const& p) const override;

    glm::vec3 dim;
};

class RefTorus : public RefPrimitive {
public:
    explicit RefTorus(
            float radius = 1.f,
            float thickness = .1f,
            glm::mat4x4 const& world2local = glm::mat4x4(1.f)) :
        radius(radius), thickness(thickness), RefPrimitive(world2local) {}

    DEFAULT_MOVE(RefTorus)

    float sdf(glm::vec3 const& p) const override;

    float radius;
    float thickness;
};

class RefCylinder : public RefPrimitive {
public:
    explicit RefCylinder(
            float radius = 1.f,
            glm::mat4x4 const& world2local = glm::mat4x4(1.f)) :
        radius(radius), RefPrimitive(world2local) {}

    DEFAULT_MOVE(RefCylinder)

    float sdf(glm::vec3 const& p) const override;

    float radius;
};

class RefCone : public RefPrimitive {
public:
    explicit RefCone(
            glm::vec2 dir = glm::vec2(0,1),
            glm::mat4x4 const& world2local = glm::mat4x4(1.f)) :
        dir(dir), RefPrimitive(world2local) {}

    DEFAULT_MOVE(RefCone)

    float sdf(glm::vec3 const& p) const override;

    glm::vec2 dir; // normalized direction of cone
};

class RefPlane : public RefPrimitive {
public:
    explicit RefPlane(
            glm::vec3 normal = glm::vec3(0,1,0),
            float offset = 0.0f,
            glm::mat4x4 const& world2local = glm::mat4x4(1.f)) :
        normal(normal), offset(offset), RefPrimitive(world2local) {}

    DEFAULT_MOVE(RefPlane)

    float sdf(glm::vec3 const& p) const override;

    glm::vec3 normal; // normal of plane
    float offset; // offset along normal from origin
};


#endif //RAYMARCHER_REFPRIMITIVE_H
