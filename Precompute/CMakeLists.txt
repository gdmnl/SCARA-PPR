cmake_minimum_required(VERSION 3.16)
project(featpush)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_OSX_ARCHITECTURES "x86_64")
set(CMAKE_CXX_FLAGS "-march=native -msse2 -msse -DHAVE_SSE2 -ffast-math -mfpmath=sse -pthread -std=c++14 -O3 -DNDEBUG")
#set(CMAKE_CXX_FLAGS "-march=native -ffast-math -use_fast_math -pthread -std=c++11 -O3 -DNDEBUG")
#set(CMAKE_CXX_FLAGS "-march=native -ffast-math -use_fast_math -pthread -std=c++11")

add_executable(featpush main.cpp
    fastPRNG.h npy.hpp
    BasicDefinition.h Graph.h BatchRandomWalk.h
    HelperFunctions.h HelperFunctions.cpp MyType.h
    FeatureOp.cpp FeatureOp.h SpeedPPR.h)
