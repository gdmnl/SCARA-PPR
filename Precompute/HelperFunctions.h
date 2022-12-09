#ifndef SCARA_HELPERFUNCTIONS_H
#define SCARA_HELPERFUNCTIONS_H

#define MSG(...)  { cout << #__VA_ARGS__ << ": " << (__VA_ARGS__) << endl; }


#include <string>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <sstream>
#include "BasicDefinition.h"
#include "MyType.h"
#include "npy.hpp"
#ifdef __linux__
    #include <sys/resource.h>
#endif


// ==================== Runtime measurement
extern double getCurrentTime();

inline long get_proc_memory(){
    struct rusage r_usage;
    getrusage(RUSAGE_SELF,&r_usage);
    return r_usage.ru_maxrss;
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
template<class FLT>
inline FLT vector_L1(std::vector<FLT> Vec){
    FLT sum = 0;
    for(FLT a : Vec)
        sum += fabs(a);
    return sum;
}

/*
Assign vector value from _data to _target:
    If _data.size == _target.size: allow swap() for fast assign if _data no longer used
    If _data.size <  _target.size: assign value according to _mapping, other values are set to 0
*/
template<class FLT>
inline void propagate_vector(std::vector<FLT> &_data, std::vector<FLT> &_target,
    const IntVector &_mapping, const NInt &target_size, bool swap = false) {
    if (target_size == _data.size()) {
        if (swap) {
            _data.swap(_target);
        } else {
            if (_target.empty()) {
                _target.reserve(target_size);
            }
            std::copy(_data.begin(), _data.end(), _target.begin());
        }
    } else {
        if (_target.empty()) {
            _target.resize(target_size, 0.0);
        } else {
            std::fill(_target.begin(), _target.end(), 0.0);
        }
        for (NInt j = 0; j < _data.size(); j++) {
            _target[_mapping[j]] = _data[j];
        }
    }
}

template<class T>
inline void show_vector(const std::string &_header, const std::vector<T> &_vec) {
    if (_vec.empty()) {
        cout << "Empty Vector." << endl;
    } else {
        cout << endl << _header;
        bool identical = true;
        const T &elem = _vec.front();
        std::for_each(_vec.begin(), _vec.end(), [&](const T &e) { identical &= (e == elem); });
        if (identical) {
            cout << "\tSize of the Vector: " << _vec.size() << "\t Value of Each Element: " << elem;
        } else {
            cout << endl;
            std::copy(begin(_vec), end(_vec), std::ostream_iterator<T>(cout, "\t"));
        }
        cout << endl;
    }
}

template<class T>
inline void output_vector(std::vector<T> Vec, std::string filename){
    std::ofstream file;
    file.open(filename, std::ios_base::app);
    for(auto a : Vec)
        file<<a<<"\t";
    file << "\n";
    file.close();
}

inline void output_feature(const FltVector &out_matrix, const std::string &_out_path,
               const unsigned long spt_size, const NInt &_node_num) {
    // Save to .npy file
    std::array<long unsigned, 2> res_shape {{spt_size, _node_num}};
    npy::SaveArrayAsNumpy(_out_path, false, res_shape.size(), res_shape.data(), out_matrix);
    cout<<"Saved "<<_out_path<<": "<<spt_size<<" "<<_node_num<<endl;
}


inline size_t load_query(IntVector &Vt_nodes, std::string query_path, const NInt &V_num){
    // By default use all nodes
    if (query_path.empty()) {
        for (NInt sid = 0; sid < V_num; sid++) {
            Vt_nodes.emplace_back(sid);
        }
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

inline size_t load_feature(IntVector &Vt_nodes, MyMatrix &feature_matrix,
    std::string feature_path) {
    NInt index = 0;
    std::vector<unsigned long> shape;
    bool fortran_order;
    FltVector arr_np;

    shape.clear();
    arr_np.clear();
    npy::LoadArrayFromNumpy(feature_path, shape, fortran_order, arr_np);
    auto feature_data = arr_np.data();
    NInt nrows = shape[0];   // node num Vt_num
    NInt ncols = shape[1];   // feature size F
    if (feature_matrix.empty())
        feature_matrix.allocate(ncols, Vt_nodes.size());  // use feature as rows
    cout<<"Input "<<feature_path<<": "<<nrows<<" "<<ncols<<endl;

    // Save each node vector (of length F) to feature_matrix
    for (NInt row = 0; row < nrows; row ++) {
        if (row == Vt_nodes[index]) {
            index++;
            FltVector feature_array(feature_data+row*ncols, feature_data+row*ncols+ncols);
            feature_matrix.set_col(index, feature_array);
        }
    }

    cout<<"Feature size: "<<feature_matrix.size()<<" "<<feature_matrix[0].size()<<endl;
    return feature_matrix.size();
}

#endif //SCARA_HELPERFUNCTIONS_H
