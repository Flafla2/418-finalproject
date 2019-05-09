
#include <algorithm>

#if __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>

#include "Cubemap.h"

#ifndef __CUDACC__
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif


#if __CUDACC__ || !defined(WITH_CUDA)
#ifdef __CUDACC__
__host__ __device__
#endif
static inline glm::vec4 get_pixel_at(PngImage const& img, int x, int y) {
    int coord = img.n * (y * img.w + x);
    unsigned char const* cols = &img.texture[coord];
    return glm::vec4(cols[0], cols[1], cols[2], img.n > 3 ? cols[3] : 1.f) / 255.0f;
}

#ifdef __CUDACC__
__host__ __device__
#endif
static inline glm::vec4 get_exp_pixel_at(PngImage const& img, int x, int y) {
    int coord = 3 * (y * img.w + x);
    float const* cols = &img.rgb_exp[coord];
    return glm::vec4(cols[0], cols[1], cols[2], 1.f);
}

#ifdef __CUDACC__
__host__ __device__
#endif
glm::vec4 PngImage::sample(glm::vec2 const& uv) {
    // Basic bilinear sampling

    float tex_x = uv.x * w + 0.5f;
    float tex_y = uv.y * h + 0.5f;

    int x0 = glm::floor(tex_x);
    int y0 = glm::floor(tex_y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float wx1 = tex_x - (float)x0;
    float wy1 = tex_y - (float)y0;
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;

    glm::vec4 s00 = get_pixel_at(*this, x0, y0);
    glm::vec4 s10 = get_pixel_at(*this, x1, y0);
    glm::vec4 s01 = get_pixel_at(*this, x0, y1);
    glm::vec4 s11 = get_pixel_at(*this, x1, y1);

    return s00 * (wx0 * wy0) + s10 * (wx1 * wy0) + s01 * (wx0 * wy1) + s11 * (wx1 * wy1);
}

#ifdef __CUDACC__
__host__ __device__
#endif
glm::vec4 sample_exp(PngImage const& img, glm::vec2 const& uv) {
    // Basic bilinear sampling

    float tex_x = uv.x * img.w + 0.5f;

    float tex_y_blk_dec = glm::floor(uv.y);
    float tex_y_blk_frac = uv.y - tex_y_blk_dec;
    tex_y_blk_frac = tex_y_blk_frac * (1.f-2.f/img.w) + 1.f/img.w;
    float tex_y = (tex_y_blk_dec+tex_y_blk_frac) * img.w + 0.5f;

    int x0 = glm::clamp((int)glm::floor(tex_x), 0, img.w-1);
    int y0 = glm::clamp((int)glm::floor(tex_y), 0, img.h-1);
    int x1 = glm::clamp(x0 + 1, 0, img.w-1);
    int y1 = glm::clamp(y0 + 1, 0, img.h-1);


    float wx1 = glm::clamp(tex_x - (float)x0, 0.f, 1.f);
    float wy1 = glm::clamp(tex_y - (float)y0, 0.f, 1.f);
    float wx0 = 1.0f - wx1;
    float wy0 = 1.0f - wy1;

    glm::vec4 s00 = get_exp_pixel_at(img, x0, y0);
    glm::vec4 s10 = get_exp_pixel_at(img, x1, y0);
    glm::vec4 s01 = get_exp_pixel_at(img, x0, y1);
    glm::vec4 s11 = get_exp_pixel_at(img, x1, y1);

    return s00 * (wx0 * wy0) + s10 * (wx1 * wy0) + s01 * (wx0 * wy1) + s11 * (wx1 * wy1);
}

/// Converts a ray direction into the UVs of a loaded cubemap.
/// Assumes that the cubemap is laid out in [bot, top, left, right, fwd, back] from top to bottom
/// Adapted from https://www.gamedev.net/forums/topic/687535-implementing-a-cube-map-lookup-function/
/// \param dir Ray to sample
/// \return UVs in [0, 1] range.
#ifdef __CUDACC__
__host__ __device__
#endif
glm::vec2 Cubemap::dir2uv(glm::vec3 dir) {
    glm::vec3 vAbs = glm::abs(dir);
    float ma;
    float faceIndex;
    glm::vec2 uv;
    if(vAbs.z >= vAbs.x && vAbs.z >= vAbs.y)
    {
        faceIndex = dir.z < 0.0 ? 5.0 : 4.0;
        ma = 0.5 / vAbs.z;
        uv = glm::vec2(dir.z < 0.0 ? -dir.x : dir.x, -dir.y);
    }
    else if(vAbs.y >= vAbs.x)
    {
        faceIndex = dir.y < 0.0 ? 0.0 : 1.0;
        ma = 0.5 / vAbs.y;
        uv = glm::vec2(dir.x, dir.y < 0.0 ? -dir.z : dir.z);
    }
    else
    {
        faceIndex = dir.x < 0.0 ? 3.0 : 2.0;
        ma = 0.5 / vAbs.x;
        uv = glm::vec2(dir.x < 0.0 ? dir.z : -dir.z, -dir.y);
    }

    uv = uv * ma + 0.5f;

    uv = glm::clamp(uv, 0.f, 1.f);

    uv.y += faceIndex;

    return uv;
}

#endif

#ifndef __CUDACC__
PngImage load_image(char const*path) {
    PngImage ret = PngImage();
    ret.texture = stbi_load(path, &ret.w, &ret.h, &ret.n, 0);

    if (ret.n == 4) {
        int const siz = ret.w * ret.h;

        auto *rgbe = new float[siz * 3];
        for(int x = 0; x < siz; x++) {
            float *fpix = rgbe + (3 * x);
            unsigned char const* cpix = ret.texture + (4 * x);

            if (cpix[0] == 0 && cpix[1] == 0 && cpix[2] == 0 && cpix[3] == 0) {
                fpix[0] = 0;
                fpix[1] = 0;
                fpix[2] = 0;
                continue;
            }

            int exp = int(cpix[3]) - 128;
            fpix[0] = glm::pow(std::ldexp((float(cpix[0]) + 0.5f) / 256.0f, exp),0.45);
            fpix[1] = glm::pow(std::ldexp((float(cpix[1]) + 0.5f) / 256.0f, exp),0.45);
            fpix[2] = glm::pow(std::ldexp((float(cpix[2]) + 0.5f) / 256.0f, exp),0.45);
        }

        ret.rgb_exp = rgbe;
    } else {
        ret.rgb_exp = nullptr;
    }

    return ret;
}
#endif
