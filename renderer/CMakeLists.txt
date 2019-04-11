cmake_minimum_required(VERSION 3.14)

project(Raymarcher 	VERSION 0.1
          					DESCRIPTION "A dynamic parallel renderer using raymarching in CUDA."
          					LANGUAGES CXX)

include(CheckLanguage)

add_library( raymarcher-common INTERFACE )
target_sources( raymarcher-common INTERFACE circleRenderer.h cycleTimer.h util.h sceneLoader.h image.h )

add_library( raymarcher-seq STATIC refRenderer.h refRenderer.cpp )
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
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

target_link_libraries( raymarcher PUBLIC ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA support detected.  Compiling with CUDA.")
  enable_language(CUDA)

  set_property(TARGET raymarcher
               PROPERTY CUDA_SEPARABLE_COMPILATION ON)

  add_library( raymarcher-cuda STATIC cudaRenderer.h cudaRenderer.cu )
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
else()
  message(WARNING "No CUDA support detected.  Compiling without CUDA.")
endif()