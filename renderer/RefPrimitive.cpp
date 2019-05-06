#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include "RefPrimitive.h"

float RefCombine::sdf(glm::vec3 const& p_) const {
    glm::vec3 p = world2local * glm::vec4(p_, 1.0f);

    switch(op) {
    case UNION:
        return glm::min(p1->sdf(p), p2->sdf(p));
    case DIFF:
        return glm::max(-p1->sdf(p), p2->sdf(p));
    case ISECT:
        return glm::max(p1->sdf(p), p2->sdf(p));
    }
}

float RefCombineSmooth::sdf(glm::vec3 const& p_) const {
    glm::vec3 p = world2local * glm::vec4(p_, 1.0f);

    float d1 = p1->sdf(p);
    float d2 = p2->sdf(p);
    float h;
    switch(op) {
    case UNION:
        h = glm::clamp( 0.5 + 0.5 * (d2 - d1) / smoothing, 0.0, 1.0 );
        return glm::mix( d2, d1, h ) - smoothing * h * (1.0 - h);
    case DIFF:
        h = glm::clamp( 0.5 - 0.5 * (d2 + d1) / smoothing, 0.0, 1.0 );
        return glm::mix( d2, -d1, h ) + smoothing * h * (1.0 - h);
    case ISECT:
        h = glm::clamp( 0.5 - 0.5 * (d2 - d1) / smoothing, 0.0, 1.0 );
        return glm::mix( d2, d1, h ) + smoothing * h * (1.0 - h);
    }
}

float RefSphere::sdf(glm::vec3 const& p_) const {
    glm::vec3 p = world2local * glm::vec4(p_, 1.0f);
    return glm::length(p) - radius;
}

float RefBox::sdf(glm::vec3 const& p_) const {
    glm::vec3 p = world2local * glm::vec4(p_, 1.0f);

    glm::vec3 d = glm::abs(p) - dim;
    return glm::length(glm::max(d, glm::vec3(0.0))) + glm::min(glm::max(d.x,glm::max(d.y,d.z)), 0.0f);
}

float RefTorus::sdf(glm::vec3 const& p_) const {
    glm::vec3 p = world2local * glm::vec4(p_, 1.0f);
    glm::vec2 q = glm::vec2(glm::length(p.xz()) - radius, p.y);
    return glm::length(q) - thickness;
}

float RefCylinder::sdf(glm::vec3 const& p_) const{
    glm::vec3 p = world2local * glm::vec4(p_, 1.0f);
    return glm::length(p.xz()) - radius;
}

float RefCone::sdf(glm::vec3 const& p_) const{
    glm::vec3 p = world2local * glm::vec4(p_, 1.0f);
    float q = glm::length(p.xy());
    return glm::dot(dir, glm::vec2(q, p.z));
}

float RefPlane::sdf(glm::vec3 const& p_) const {
    glm::vec3 p = world2local * glm::vec4(p_, 1.0f);
    return glm::dot(p, normal) + offset;
}