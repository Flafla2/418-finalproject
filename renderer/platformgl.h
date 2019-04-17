#ifndef __PLATFORM_GL_H__
#define __PLATFORM_GL_H__

#ifdef __APPLE__
// Silence OpenGL Deprecation warning on MacOS
#define GL_SILENCE_DEPRECATION

#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#endif

