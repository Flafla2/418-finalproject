# CMU 15-418, Spring 2019 Final Project
### Adrian Biagioli / Chiara Mrose

This is a CUDA-based renderer that uses raymarching of signed distance fields to realize complex primitive-based scene descriptions.

## Building

From the root directory, run:
```
$ make
```
This will create a new directory called `build` with the executable `raymarcher`

On MacOS, you can alternatively run:
```
$ make xcode-project
```

This will create an xcode project in a new directory called `xcode`.  To clean any build artifacts, run:
```
$ make clean
```

You will notice that the `Makefile` in the root directory is a thin layer on top of CMake.  Therefore you can build for other platforms / tools that are supported by `cmake`.

## Running

To run the current test scene, run
```
$ ./raymarcher test
```
