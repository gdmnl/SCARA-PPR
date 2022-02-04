cmake_minimum_required(VERSION 3.16)
project(featpush)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-march=native -msse2 -msse -DHAVE_SSE2 -ffast-math -mfpmath=sse -pthread -lcnpy -lz -std=c++14 -O3 -DNDEBUG")
#set(CMAKE_CXX_FLAGS "-march=native -msse2 -msse -DHAVE_SSE2 -ffast-math -mfpmath=sse -pthread -L${CMAKE_SOURCE_DIR}/cnpy/include/lib/ -lcnpy -lz -std=c++14 -O3 -DNDEBUG")
#set(CMAKE_CXX_FLAGS "-march=native -ffast-math -use_fast_math -pthread -std=c++11 -O3 -DNDEBUG")
#set(CMAKE_CXX_FLAGS "-march=native -ffast-math -use_fast_math -pthread -std=c++11")

set(SFMT_INCLUDE_DIR "SFMT-src-1.5.1/")
INCLUDE_DIRECTORIES(${SFMT_INCLUDE_DIR})
set(FAST_DOUBLE_PARSER_INCLUDE_DIR "fast_double_parser-master/include")
INCLUDE_DIRECTORIES(${FAST_DOUBLE_PARSER_INCLUDE_DIR})
set(CNPY_INCLUDE_DIR "cnpy-master/include")
INCLUDE_DIRECTORIES(${CNPY_INCLUDE_DIR})
link_directories(${CNPY_INCLUDE_DIR})

add_executable(featpush main.cpp
        Graph.h BasicDefinition.h
        HelperFunctions.h IteratedMethods.h HelperFunctions.cpp
        QueryLoader.h
        BatchRandomWalk.cpp BatchRandomWalk.h
        MyQueue.h MyRandom.cpp MyRandom.h
        SpeedPPR.cpp SpeedPPR.h
        SFMT-src-1.5.1/SFMT.c SFMT-src-1.5.1/SFMT.h SFMT-src-1.5.1/SFMT-sse2.h
        cnpy-master/cnpy.cpp
        CleanGraph.h QueryGenerator.h)
target_compile_definitions(featpush PRIVATE SFMT_MEXP=19937)

find_library (
    CNPY_LIB
    NAMES cnpy libcnpy
    HINTS "./cnpy-master/include/lib"
    NO_DEFAULT_PATH # do not search system default paths
)
target_link_libraries(featpush
    ${CNPY_LIB}
)