#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "refRenderer.h"
#if WITH_CUDA
#include "cudaRenderer.h"
#endif
#include "platformgl.h"


void startRendererWithDisplay(Renderer* renderer, bool printStats = true);
void startBenchmark(Renderer* renderer, int startFrame, int totalFrames, const std::string& frameFilename);
void CheckBenchmark(Renderer* ref_renderer, Renderer* cuda_renderer,
                        int benchmarkFrameStart, int totalFrames, const std::string& frameFilename);


void usage(const char* progname) {
    printf("Usage: %s [options] scenename\n", progname);
    printf("Valid scenenames are: test\n");
    printf("Program Options:\n");
    printf("  -b  --bench <START:END>    Benchmark mode, do not create display. Time frames [START,END)\n");
    printf("                             Requires CUDA-capable machine.\n");
    printf("  -c  --check                Check correctness of output.  Requires CUDA-capable machine.\n");
    printf("  -f  --file  <FILENAME>     Dump frames in benchmark mode (FILENAME_xxxx.ppm)\n");
    printf("  -r  --renderer <ref/cuda>  Select renderer: ref or cuda\n");
    printf("  -d  --frame-data           Print frame timing data (Clear/Advance/Render)");
    printf("  -w  --width  <INT>         Set width (default: 800px)\n");
    printf("  -h  --height <INT>         Set height (default: 600px)\n");
    printf("  -e  --emit-bytecode        Emit CUDA bytecode (requires running in CUDA mode)\n");
    printf("  -?  --help                 This message\n");
}


int main(int argc, char** argv)
{

    int benchmarkFrameStart = -1;
    int benchmarkFrameEnd = -1;
    int imageWidth = 800;
    int imageHeight = 600;

    std::string sceneNameStr;
    std::string frameFilename;
    SceneName sceneName;
    bool useRefRenderer = true;
    bool frameData = false;
    bool checkCorrectness = false;
    bool emitBytecode = false;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"help",     0, 0,  '?'},
        {"check",    0, 0,  'c'},
        {"bench",    1, 0,  'b'},
        {"file",     1, 0,  'f'},
        {"renderer", 1, 0,  'r'},
        {"width",    1, 0,  'w'},
        {"height",   1, 0,  'h'},
        {"frame-data", 1, 0, 'd'},
        {"emit-bytecode", 1, 0, 'e'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "b:f:r:w:h:c?de", long_options, nullptr)) != EOF) {

        switch (opt) {
        case 'b':
            if (sscanf(optarg, "%d:%d", &benchmarkFrameStart, &benchmarkFrameEnd) != 2) {
                fprintf(stderr, "Invalid argument to -b option\n");
                usage(argv[0]);
                exit(1);
            }
            break;
        case 'c':
            checkCorrectness = true;
            break;
        case 'f':
            frameFilename = optarg;
            break;
        case 'r':
            if (std::string(optarg).compare("cuda") == 0) {
                useRefRenderer = false;
            }
            break;
        case 'w':
            imageWidth = atoi(optarg);
            break;
        case 'h':
            imageHeight = atoi(optarg);
            break;
        case 'd':
            frameData = true;
            break;
        case 'e':
            emitBytecode = true;
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////


    if (optind + 1 > argc) {
        fprintf(stderr, "Error: missing scene name\n");
        usage(argv[0]);
        return 1;
    }

    sceneNameStr = argv[optind];

    if (sceneNameStr.compare("test1") == 0) {
        sceneName = TEST_SCENE1;
    }
    else if (sceneNameStr.compare("test2") == 0) {
        sceneName = TEST_SCENE2;
    }
    else if (sceneNameStr.compare("test3") == 0) {
        sceneName = TEST_SCENE3;
    }
    else if (sceneNameStr.compare("test4") == 0) {
        sceneName = TEST_SCENE4;
    }
    else if (sceneNameStr.compare("test5") == 0) {
        sceneName = TEST_SCENE5;
    }
    else {
        fprintf(stderr, "Unknown scene name (%s)\n", sceneNameStr.c_str());
        usage(argv[0]);
        return 1;
    }

    printf("Rendering to %dx%d image\n", imageWidth, imageHeight);

    Renderer* renderer;

    if (checkCorrectness) {
#if WITH_CUDA
        // Need both the renderers

        Renderer* ref_renderer;
        Renderer* cuda_renderer;

        ref_renderer = new RefRenderer();
        cuda_renderer = new CudaRenderer();

        ref_renderer->allocOutputImage(imageWidth, imageHeight);
        ref_renderer->loadScene(sceneName);
        ref_renderer->setup();
        cuda_renderer->allocOutputImage(imageWidth, imageHeight);
        cuda_renderer->loadScene(sceneName);
        cuda_renderer->setup();

        // Check the correctness
        CheckBenchmark(ref_renderer, cuda_renderer, 0, 1, frameFilename);
#else
        fprintf(stderr, "Checking correctness is not supported when compiling without CUDA.\n");
        usage(argv[0]);
        exit(1);
#endif
    }
    else {
#if WITH_CUDA
        if (useRefRenderer)
            renderer = new RefRenderer();
        else {
            CudaRenderer *cr = new CudaRenderer();
            cr->emitBytecode = emitBytecode;
            renderer = cr;
        }
#else
        if (!useRefRenderer) {
            fprintf(stderr, "Rendering with CUDA is not supported when compiling without CUDA.\n");
            exit(1);
        }
        
        renderer = new RefRenderer();
#endif

        renderer->allocOutputImage(imageWidth, imageHeight);
        renderer->loadScene(sceneName);
        renderer->setup();

        if (benchmarkFrameStart >= 0)
            startBenchmark(renderer, benchmarkFrameStart, benchmarkFrameEnd - benchmarkFrameStart, frameFilename);
        else {
            glutInit(&argc, argv);
            startRendererWithDisplay(renderer, frameData);
        }
    }

    return 0;
}
