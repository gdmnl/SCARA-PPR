cmake_minimum_required(VERSION 3.16)
project(featpush)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_OSX_ARCHITECTURES "x86_64")
set(CMAKE_CXX_FLAGS "-march=native -msse2 -msse -DHAVE_SSE2 -ffast-math -mfpmath=sse -pthread -lz -std=c++14 -O3 -DNDEBUG")
#set(CMAKE_CXX_FLAGS "-march=native -ffast-math -use_fast_math -pthread -std=c++11 -O3 -DNDEBUG")
#set(CMAKE_CXX_FLAGS "-march=native -ffast-math -use_fast_math -pthread -std=c++11")

set(SFMT_INCLUDE_DIR "SFMT/")
INCLUDE_DIRECTORIES(${SFMT_INCLUDE_DIR})
set(FAST_DOUBLE_PARSER_INCLUDE_DIR "fast_double_parser/include")
INCLUDE_DIRECTORIES(${FAST_DOUBLE_PARSER_INCLUDE_DIR})

add_executable(featpush main.cpp
    Graph.h BasicDefinition.h CleanGraph.h
    HelperFunctions.h HelperFunctions.cpp npy.hpp
    BatchRandomWalk.cpp BatchRandomWalk.h
    MyQueue.h MyRandom.cpp MyRandom.h
    SpeedPPR.cpp SpeedPPR.h
    SFMT/SFMT.c SFMT/SFMT.h SFMT/SFMT-sse2.h)
target_compile_definitions(featpush PRIVATE SFMT_MEXP=19937)

target_link_libraries(featpush
    z
)
