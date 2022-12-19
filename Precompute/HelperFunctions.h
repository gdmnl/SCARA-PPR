/*
  Interface and IO
  Author: nyLiao
*/
#ifndef SCARA_HELPERFUNCTIONS_H
#define SCARA_HELPERFUNCTIONS_H

#define MSG(...)  { cout << #__VA_ARGS__ << ": " << (__VA_ARGS__) << endl; }


#include <string>
#include <vector>
#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <unistd.h>
#include <malloc.h>
#include "BasicDefinition.h"
#ifdef __linux__
    #include <sys/resource.h>
#endif


// ==================== Runtime measurement
extern double getCurrentTime();

inline float get_proc_memory(){
    struct rusage r_usage;
    getrusage(RUSAGE_SELF,&r_usage);
    return r_usage.ru_maxrss/1000000.0;
}

inline float get_alloc_memory(){
    struct mallinfo mi = mallinfo();
    return mi.uordblks / 1000000000.0;
}

inline float get_stat_memory(){
    long rss;
    std::string ignore;
    std::ifstream ifs("/proc/self/stat", std::ios_base::in);
    ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore
            >> ignore >> ignore >> ignore >> rss;

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;
    return rss * page_size_kb / 1000000.0;
}

// ==================== Argument parsing
struct Param {
    std::string graph_file;
    std::string query_file;
    std::string feature_file;
    std::string algorithm = "featpush";
    std::string data_folder;
    std::string estimation_folder;
    unsigned int thread_num = 1;
    unsigned int seed = 0;
    float epsilon = 0.5;
    float alpha = 0.2;
    float gamma = 0.2;
    float base_ratio = 0.04;
    bool index = false;
    bool output_estimations = false;
};

extern Param param;

extern Param parseArgs(int nargs, char **args);

// ==================== IO
inline size_t load_query(IntVector &Vt_nodes, std::string query_path, const NInt &V_num){
    // By default use all nodes
    if (query_path.empty()) {
        Vt_nodes.resize(V_num);
        std::iota(Vt_nodes.begin(), Vt_nodes.end(), 0);
    } else {
        std::ifstream query_file(query_path);
        if (query_file.good() == false) {
            printf("File Not Exists.\n");
            exit(1);
        }
        for (NInt sid; (query_file >> sid);) {
            Vt_nodes.emplace_back(sid);
        }
        if (Vt_nodes.empty()) {
            printf("Error! Empty File\n");
        }
        query_file.close();
    }

    cout << "Query size: " << Vt_nodes.size() << endl;
    return Vt_nodes.size();
}

#endif //SCARA_HELPERFUNCTIONS_H
