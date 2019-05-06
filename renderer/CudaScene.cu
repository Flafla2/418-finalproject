#include "CudaScene.h"

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#define DEBUG
#include "cuda_error.h"
#include "cuda_constants.h"

// Refer to the constants that are defined in cudaRenderer.cu
// See https://stackoverflow.com/questions/7959174/nvcc-combine-extern-and-constant
//     ^^ Second answer, because the primary was written pre-cuda 5.0
extern __constant__ GlobalConstants cuConstRendererParams;
extern __constant__ SceneConstants cudaConstSceneParams;


/// Appends the bytes that make up <c>data</c> to <c>vec</c>
/// \tparam T Type of <c>data</c>, so <c>sizeof(T)</c> bytes will be appended
/// \param vec Vector to append to
/// \param data Data to append
template<typename T>
static void appendStruct(std::vector<char> &vec, T const& data) {
    char const*raw = reinterpret_cast<char const*>(&data);
    vec.insert(vec.end(), raw, raw + sizeof(T));
}

static void makeAligned(std::vector<char> &vec, char align = 16) {
    int count = (align - vec.size() % align) % align;
    for (int x = 0; x < count; ++x)
        vec.push_back((char)0x00);
}

__host__
static void appendPrimitive(std::vector<char> &ret, RefPrimitive const* cur) {
    RefSphere const*sphere = dynamic_cast<RefSphere const*>(cur);
    if (sphere) {
        ret.push_back(CudaOpcodes::Sphere);
        makeAligned(ret);
        appendStruct(ret, CudaSphere(sphere));
        return;
    }

    RefBox const*box = dynamic_cast<RefBox const*>(cur);
    if (box) {
        ret.push_back(CudaOpcodes::Box);
        makeAligned(ret);
        appendStruct(ret, CudaBox(box));
        return;
    }

    RefTorus const*torus = dynamic_cast<RefTorus const*>(cur);
    if (torus) {
        ret.push_back(CudaOpcodes::Torus);
        makeAligned(ret);
        appendStruct(ret, CudaTorus(torus));
        return;
    }

    RefCylinder const*cylinder = dynamic_cast<RefCylinder const*>(cur);
    if (cylinder) {
        ret.push_back(CudaOpcodes::Cylinder);
        makeAligned(ret);
        appendStruct(ret, CudaCylinder(cylinder));
        return;
    }

    RefCone const*cone = dynamic_cast<RefCone const*>(cur);
    if (cone) {
        ret.push_back(CudaOpcodes::Cone);
        makeAligned(ret);
        appendStruct(ret, CudaCone(cone));
        return;
    }

    RefPlane const*plane = dynamic_cast<RefPlane const*>(cur);
    if (plane) {
        ret.push_back(CudaOpcodes::Plane);
        makeAligned(ret);
        appendStruct(ret, CudaPlane(plane));
        return;
    }

    RefCombine const*combine = dynamic_cast<RefCombine const*>(cur);
    if (combine) {
        appendPrimitive(ret, combine->p1);
        appendPrimitive(ret, combine->p2);

        ret.push_back(CudaOpcodes::Combine);
        switch(combine->op) {
            case UNION:
                ret.push_back(0x00);
                break;
            case DIFF:
                ret.push_back(0x01);
                break;
            case ISECT:
                ret.push_back(0x02);
                break;
        }
        return;
    }

    RefCombineSmooth const*combineSmooth = dynamic_cast<RefCombineSmooth const*>(cur);
    if (combineSmooth) {
        appendPrimitive(ret, combineSmooth->p1);
        appendPrimitive(ret, combineSmooth->p2);

        ret.push_back(CudaOpcodes::CombineSmooth);
        switch(combineSmooth->op) {
            case UNION:
                ret.push_back(0x00);
                break;
            case DIFF:
                ret.push_back(0x01);
                break;
            case ISECT:
                ret.push_back(0x02);
                break;
        }
        makeAligned(ret, sizeof(float));
        appendStruct<float>(ret, combineSmooth->smoothing);
        return;
    }
}

std::vector<char> CudaOpcodes::refToBytecode(std::vector<RefPrimitive *> const& prims) {
    std::vector<char> ret;

    for(int p = 0; p < prims.size(); ++p) {
        RefPrimitive const* cur = prims[p];

        appendPrimitive(ret, cur);
        if(p != 0) {
            // At the root level of iterating through RefPrimitives, there is an implicit
            // minimization of all SDFs.  Theoretically you could represent a vector of
            // RefPrimitives as a tree of RefCombine primitives with UNION operation
            ret.push_back(CudaOpcodes::Combine);
            ret.push_back(0x00); // union
        }

    }

    return ret;
}

CudaScene::CudaScene(std::vector<RefPrimitive *> const& prims) {
    bytecode = CudaOpcodes::refToBytecode(prims);
}

CudaScene::~CudaScene() = default;

__device__ static inline void alignPC(int &pc, char align = 16) {
    int count = (align - pc % align) % align;
    pc += count;
}

// Maximum depth of instruction stack
#define STACK_SIZE 64

__device__ float deviceSdf(glm::vec3 p) {
    char const*const bytecode = cudaConstSceneParams.bytecode;
    int const size = cudaConstSceneParams.bytecodeSize;

    int sc = -1; // stack counter
    int pc = 0; // program counter

    // Stack of SDFs that will be coalesced by combination instructions
    float stack[STACK_SIZE];

    while (pc < size) {
        const char instruction = bytecode[pc++];

        switch(instruction) {
            case CudaOpcodes::Sphere: {
                alignPC(pc);
                CudaSphere const *s = reinterpret_cast<CudaSphere const *>(&bytecode[pc]);
                stack[++sc] = SphereSDF(*s, p);
                pc += sizeof(CudaSphere);
                break;
            }
            case CudaOpcodes::Box: {
                alignPC(pc);
                CudaBox const *b = reinterpret_cast<CudaBox const*>(&bytecode[pc]);
                stack[++sc] = BoxSDF(*b, p);
                pc += sizeof(CudaBox);
                break;
            }
            case CudaOpcodes::Torus: {
                alignPC(pc);
                CudaTorus const *t = reinterpret_cast<CudaTorus const*>(&bytecode[pc]);
                stack[++sc] = TorusSDF(*t, p);
                pc += sizeof(CudaTorus);
                break;
            }
            case CudaOpcodes::Cylinder: {
                alignPC(pc);
                CudaCylinder const *c = reinterpret_cast<CudaCylinder const*>(&bytecode[pc]);
                stack[++sc] = CylinderSDF(*c, p);
                pc += sizeof(CudaTorus);
                break;
            }
            case CudaOpcodes::Cone: {
                alignPC(pc);
                CudaCone const *k = reinterpret_cast<CudaCone const*>(&bytecode[pc]);
                stack[++sc] = ConeSDF(*k, p);
                pc += sizeof(CudaCone);
                break;
            }
            case CudaOpcodes::Plane: {
                alignPC(pc);
                CudaPlane const *l = reinterpret_cast<CudaPlane const*>(&bytecode[pc]);
                stack[++sc] = PlaneSDF(*l, p);
                pc += sizeof(CudaPlane);
                break;
            }
            case CudaOpcodes::Combine: {
                float p1 = stack[sc--];
                float p2 = stack[sc--];
                switch(bytecode[pc]) {
                    case 0x00: // union
                        stack[++sc] = glm::min(p1, p2);
                        break;
                    case 0x01: // diff
                        stack[++sc] = glm::max(-p1, p2);
                        break;
                    case 0x02: // isect
                        stack[++sc] = glm::max(p1, p2);
                        break;
                    default: break;
                }
                pc++;
                break;
            }
            case CudaOpcodes::CombineSmooth: {
                float d1 = stack[sc--];
                float d2 = stack[sc--];
                char op = bytecode[pc++];
                alignPC(pc, sizeof(float));
                float smoothing = *(reinterpret_cast<float const*>(&bytecode[pc]));

                float h;
                switch(op) {
                    case 0x00: // union
                        h = glm::clamp( 0.5 + 0.5 * (d2 - d1) / smoothing, 0.0, 1.0 );
                        stack[++sc] = glm::mix( d2, d1, h ) - smoothing * h * (1.0 - h);
                        break;
                    case 0x01: // diff
                        h = glm::clamp( 0.5 - 0.5 * (d2 + d1) / smoothing, 0.0, 1.0 );
                        stack[++sc] = glm::mix( d2, -d1, h ) + smoothing * h * (1.0 - h);
                        break;
                    case 0x02: // isect
                        h = glm::clamp( 0.5 - 0.5 * (d2 - d1) / smoothing, 0.0, 1.0 );
                        stack[++sc] = glm::mix( d2, d1, h ) + smoothing * h * (1.0 - h);
                        break;
                    default: break;
                }
                pc += sizeof(float);
                break;
            }
        }
    }

    return stack[0];
}

__device__ glm::vec3 deviceNormal(glm::vec3 p) {
    const float eps = 0.001f;
    const glm::vec3 x = glm::vec3(eps, 0, 0);
    const glm::vec3 y = glm::vec3(0, eps, 0);
    const glm::vec3 z = glm::vec3(0, 0, eps);

    glm::vec3 ret(
            deviceSdf(p + x) - deviceSdf(p - x),
            deviceSdf(p + y) - deviceSdf(p - y),
            deviceSdf(p + z) - deviceSdf(p - z)
    );
    return glm::normalize(ret);
}

void CudaScene::initCudaData() {
    static bool cudaDataInitialized = false;
    if (cudaDataInitialized) {
        std::cerr << "ERROR: Scene Cuda data already initialized!  Exiting." << std::endl;
        exit(1);
    }
    cudaDataInitialized = true;

    cudaCheckError(
            cudaMalloc(&cudaDeviceBytecode, bytecode.size())
    );
    cudaCheckError(
            cudaMemcpy(cudaDeviceBytecode, bytecode.data(), bytecode.size(), cudaMemcpyHostToDevice)
    );

    SceneConstants params;
    params.bytecode = cudaDeviceBytecode;
    params.bytecodeSize = bytecode.size();

    cudaCheckError(
            cudaMemcpyToSymbol(cudaConstSceneParams, &params, sizeof(SceneConstants))
    );
}