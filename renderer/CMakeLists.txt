cmake_minimum_required(VERSION 3.14)

project(Raymarcher 	VERSION 0.1
          					DESCRIPTION "A dynamic parallel renderer using raymarching in CUDA."
          					LANGUAGES CXX)

include(CheckLanguage)

add_library( raymarcher-common INTERFACE )
target_sources( raymarcher-common INTERFACE renderer.h cycleTimer.h util.h sceneLoader.h image.h Scene.cpp Scene.h Primitive.cpp Primitive.h )

add_library( raymarcher-seq STATIC refRenderer.h refRenderer.cpp RefScene.cpp RefScene.h)
target_link_libraries( raymarcher-seq PUBLIC raymarcher-common )

target_compile_features(raymarcher-seq PUBLIC cxx_std_11)
set_target_properties(raymarcher-seq PROPERTIES CXX_EXTENSIONS OFF)

add_executable( raymarcher
  main.cpp
  benchmark.cpp
  display.cpp
  platformgl.h
  ppm.h
  ppm.cpp
  sceneLoader.cpp )
target_link_libraries( raymarcher PUBLIC raymarcher-seq)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLM REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} ${GLM_INCLUDE_DIRS} )

target_link_libraries( raymarcher PUBLIC ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA support detected.  Compiling with CUDA.")
  enable_language(CUDA)

  set_property(TARGET raymarcher
               PROPERTY CUDA_SEPARABLE_COMPILATION ON)

  add_compile_definitions(WITH_CUDA)

  add_library( raymarcher-cuda STATIC cudaRenderer.h cudaRenderer.cu CudaScene.cu CudaScene.h)
  target_link_libraries( raymarcher-cuda PUBLIC raymarcher-common )
  target_link_libraries( raymarcher PUBLIC raymarcher-cuda )

  target_compile_features(raymarcher-cuda PUBLIC cxx_std_11)
  set_target_properties(raymarcher-cuda PROPERTIES CXX_EXTENSIONS OFF)

  # https://devblogs.nvidia.com/building-cuda-applications-cmake/
  if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath, 
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET particle_test 
                 PROPERTY
                 BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  endif()

  add_compile_definitions(CUDA_ENABLED)

# I tried to add support for coriander (see here):
#     https://github.com/hughperkins/coriander
# But it wasn't working with the base code.  Try uncommenting this to see if you can get it working!

# elseif(EXISTS "$ENV{HOME}/coriander/share/cocl/cocl.cmake")
#   message(STATUS "CUDA->OpenCL support detected via coriander.  Compiling with cocl.")
#   include("$ENV{HOME}/coriander/share/cocl/cocl.cmake")

#   add_compile_definitions(WITH_CUDA WITH_COCL)
#   cocl_add_library( raymarcher-cocl cudaRenderer.h cudaRenderer.cu )
#   target_sources( raymarcher-cocl INTERFACE cudaRenderer.cu )

#   target_compile_features(raymarcher-cocl PUBLIC cxx_std_11)
#   set_target_properties(raymarcher-cocl PROPERTIES CXX_EXTENSIONS OFF)

#   target_link_libraries( raymarcher-cocl PUBLIC raymarcher-common )
#   target_link_libraries( raymarcher-cocl PUBLIC cocl clblast clew easycl )

#   target_link_libraries( raymarcher PUBLIC raymarcher-cocl )

#   set_target_properties(raymarcher-cocl PROPERTIES COMPILE_FLAGS "-DWITH_CUDA -DWITH_COCL")

else()
  message(WARNING "No CUDA support detected.  Compiling without CUDA.")
  add_custom_target( raymarcher-cuda SOURCES cudaRenderer.h cudaRenderer.cu )
endif()