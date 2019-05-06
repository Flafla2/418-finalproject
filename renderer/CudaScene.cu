#include "CudaScene.h"

#include <iostream>
#include <iomanip>
#include <typeinfo>

#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "cuda_error.h"
#include "cuda_constants.h"

// Refer to the constants that are defined in cudaRenderer.cu
// See https://stackoverflow.com/questions/7959174/nvcc-combine-extern-and-constant
//     ^^ Second answer, because the primary was written pre-cuda 5.0
extern __constant__ GlobalConstants cuConstRendererParams;
extern __constant__ SceneConstants cudaConstSceneParams;


#define LOG_BYTECODE_PER_STRUCT

/// Appends the bytes that make up <c>data</c> to <c>vec</c>
/// \tparam T Type of <c>data</c>, so <c>sizeof(T)</c> bytes will be appended
/// \param vec Vector to append to
/// \param data Data to append
template<typename T>
static void appendStruct(std::vector<char> &vec, T const& data) {
    char const*raw = reinterpret_cast<char const*>(&data);
    vec.insert(vec.end(), raw, raw + sizeof(T));
#if defined(DEBUG) && defined(LOG_BYTECODE_PER_STRUCT)
    std::cout << "Appending struct to byte stream: " << typeid(T).name();
    std::cout << std::hex;
    int count = 0;
    for(char const *c = raw; c < raw + sizeof(T); ++c) {
        if (count % 16 == 0)
            std::cout << std::endl;
        else
            std::cout << ' ';
        std::cout << std::setfill('0') << std::setw(2) << (static_cast<int>(*c) & 0xff);
        count++;
    }
    std::cout << std::dec;
    std::cout << std::endl;
#endif
}

static void makeAligned(std::vector<char> &vec, char align = 16) {
    int count = (align - vec.size() % align) % align;
    for (int x = 0; x < count; ++x)
        vec.push_back((char)0x00);
    __dbg_assert(vec.size() % align == 0);
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

#if defined(DEBUG)
    std::cout << "Loaded bytecode:" << std::endl << std::hex;
    for(int x = 0; x < ret.size(); ++x) {
        if (x > 0) {
            if (x % 16 == 0)
                std::cout << std::endl;
            else
                std::cout << ' ';
        }

        std::cout << std::setfill('0') << std::setw(2) << (static_cast<int>(ret[x]) & 0xff);
    }
    std::cout << std::dec << std::endl << "[EOF]" << std::endl;
#endif

    return ret;
}

CudaScene::CudaScene(std::vector<RefPrimitive *> const& prims) {
    bytecode = CudaOpcodes::refToBytecode(prims);
}

CudaScene::~CudaScene() = default;

__device__ static inline void alignPC(int &pc, char align = 16) {
    int count = (align - pc % align) % align;
    pc += count;
    __dbg_assert(pc % align == 0);
}

// Maximum depth of instruction stack
#define STACK_SIZE 64

// Uncomment to enable printf messages on the thread with threadIdx/blockIdx = (0,0,0)
// #define DEBUG_SINGLE_ITER

__device__ float deviceSdf(glm::vec3 p) {
    char const*const bytecode = cudaConstSceneParams.bytecode;
    int const size = cudaConstSceneParams.bytecodeSize;

    int sc = -1; // stack counter
    int pc = 0; // program counter

    // Stack of SDFs that will be coalesced by combination instructions
    float stack[STACK_SIZE];

#ifdef DEBUG_SINGLE_ITER
    bool master = blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0;
    if (!master) return 10000;
    printf("[%d,%d->%d,%d] Computing SDF at (%f, %f, %f) -- bytecode size = %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, p.x, p.y, p.z, size);
#endif

    while (pc < size) {

#ifdef DEBUG_SINGLE_ITER
        if (master)
            printf("Loading instruction with pc = %d...", pc);
#endif

        const char instruction = bytecode[pc++];

#ifdef DEBUG_SINGLE_ITER
        if (master)
            printf("Loaded. (%d)\n", (int)instruction);
#endif

        switch(instruction) {
            case CudaOpcodes::Sphere: {
                alignPC(pc);
                __dbg_assert(pc + sizeof(CudaSphere) < size);
                CudaSphere const *s = reinterpret_cast<CudaSphere const *>(&bytecode[pc]);
                stack[++sc] = SphereSDF(*s, p);
                __dbg_assert(sc >= 0);
                pc += sizeof(CudaSphere);
                break;
            }
            case CudaOpcodes::Box: {
                alignPC(pc);
                __dbg_assert(pc + sizeof(CudaBox) < size);
                CudaBox const *b = reinterpret_cast<CudaBox const*>(&bytecode[pc]);
                stack[++sc] = BoxSDF(*b, p);
                __dbg_assert(sc >= 0 && sc < STACK_SIZE);
                pc += sizeof(CudaBox);
                break;
            }
            case CudaOpcodes::Torus: {
                alignPC(pc);
                __dbg_assert(pc + sizeof(CudaTorus) < size);
                CudaTorus const *t = reinterpret_cast<CudaTorus const*>(&bytecode[pc]);
                stack[++sc] = TorusSDF(*t, p);
                __dbg_assert(sc >= 0 && sc < STACK_SIZE);
                pc += sizeof(CudaTorus);
                break;
            }
            case CudaOpcodes::Cylinder: {
                alignPC(pc);
                __dbg_assert(pc + sizeof(CudaCylinder) < size);
                CudaCylinder const *c = reinterpret_cast<CudaCylinder const*>(&bytecode[pc]);
                stack[++sc] = CylinderSDF(*c, p);
                __dbg_assert(sc >= 0 && sc < STACK_SIZE);
                pc += sizeof(CudaTorus);
                break;
            }
            case CudaOpcodes::Cone: {
                alignPC(pc);
                __dbg_assert(pc + sizeof(CudaCone) < size);
                CudaCone const *k = reinterpret_cast<CudaCone const*>(&bytecode[pc]);
                stack[++sc] = ConeSDF(*k, p);
                __dbg_assert(sc >= 0 && sc < STACK_SIZE);
                pc += sizeof(CudaCone);
                break;
            }
            case CudaOpcodes::Plane: {
                alignPC(pc);
                __dbg_assert(pc + sizeof(CudaPlane) < size);
                CudaPlane const *l = reinterpret_cast<CudaPlane const*>(&bytecode[pc]);
                stack[++sc] = PlaneSDF(*l, p);
                __dbg_assert(sc >= 0 && sc < STACK_SIZE);
                pc += sizeof(CudaPlane);
                break;
            }
            case CudaOpcodes::Combine: {
                __dbg_assert(sc >= 0 && sc < STACK_SIZE);
                float p1 = stack[sc--];
                __dbg_assert(sc >= 0 && sc < STACK_SIZE);
                float p2 = stack[sc--];
                __dbg_assert(sc >= -1 && sc < STACK_SIZE);
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
                __dbg_assert(sc >= 0 && sc < STACK_SIZE);
                float d1 = stack[sc--];
                __dbg_assert(sc >= 0 && sc < STACK_SIZE);
                float d2 = stack[sc--];
                __dbg_assert(sc >= -1 && sc < STACK_SIZE);
                char op = bytecode[pc++];
                alignPC(pc, sizeof(float));
                __dbg_assert(pc < size);
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

        __dbg_assert(pc <= size && pc >= 0);
        __dbg_assert(sc >= -1 && sc < STACK_SIZE);
    }

#ifdef DEBUG_SINGLE_ITER
    printf("Computed SDF at (%f,%f,%f) = %f\n", p.x, p.y, p.z, stack[0]);
#endif

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
    std::cout << "Uploading CUDA Scene data...";
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
    std::cout << "...done" << std::endl;
}