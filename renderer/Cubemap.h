#ifndef RAYMARCHER_CUBEMAP_H
#define RAYMARCHER_CUBEMAP_H

#include <glm/glm.hpp>

struct PngImage {
    unsigned char const *texture;
    float const *rgb_exp;
    int w;
    int h;
    int n; // components per pixel

#ifdef __CUDACC__
    __host__ __device__
#endif
    glm::vec4 sample(glm::vec2 const& uv);
};

#ifndef __CUDACC__
PngImage load_image(char const*path);
#endif

#ifdef __CUDACC__
__host__ __device__
#endif
glm::vec4 sample_exp(PngImage const& img, glm::vec2 const& uv);

namespace Cubemap {
#ifdef __CUDACC__
    __host__ __device__
#endif
    glm::vec2 dir2uv(glm::vec3 dir);
}

#endif //RAYMARCHER_CUBEMAP_H
