

#ifndef SPEEDPPR_HELPERFUNCTIONS_H
#define SPEEDPPR_HELPERFUNCTIONS_H

#define MSG(...)  { std::cout << #__VA_ARGS__ << ":" << (__VA_ARGS__) << std::endl; }


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
#include "npy.hpp"
#ifdef __linux__
    #include <sys/resource.h>
#endif

extern double getCurrentTime();


inline uint32_t parse_integer(const std::string &_str, size_t &_end) {
    uint32_t rtn = 0;
    for (_end = 0; !isdigit(_str[_end]);) { ++_end; }
    for (; isdigit(_str[_end]); ++_end) {
        rtn *= 10;
        rtn += _str[_end] - '0';
    }
    return rtn;
}

inline uint32_t parse_integer(const std::string &_str, const size_t &_start, size_t &_end) {
    uint32_t rtn = 0;
    for (_end = _start; !isdigit(_str[_end]);) { ++_end; }
    for (; isdigit(_str[_end]); ++_end) {
        rtn *= 10;
        rtn += _str[_end] - '0';
    }
    return rtn;
}

inline double parse_double(const std::string &_str, const size_t &_start, size_t &_end) {
    size_t rtn = 0, scale = 1;
    for (_end = _start; isspace(_str[_end]); ++_end);
    if (!isdigit(_str[_end])) { printf("Error in parsing double.\n"); }
    for (; isdigit(_str[_end]); ++_end) {
        rtn *= 10;
        rtn += _str[_end] - '0';
    }
    for (; isspace(_str[_end]);  ++_end);
    if (_str[_end++] != '.') {
        printf("Error in parsing double. Expecting Dot. \n");
        exit(1);
    }
    for (; isdigit(_str[_end]); ++_end, scale *= 10) {
        rtn *= 10;
        rtn += _str[_end] - '0';
    }
    return rtn / (double) scale;
}

struct Param {
    std::string graph_file;
    std::string query_file;
    std::string meta_file;
    std::string feature_file;
    std::string graph_binary_file;
    std::string algorithm = "featpush";
    std::string output_folder;
    std::string estimation_folder;
    unsigned int split_num = 1;
    unsigned int seed = 0;
    double epsilon = 0.5;
    double alpha = 0.2;
    double gamma = 0.2;
    double base_ratio = 0.04;
    bool is_undirected_graph = false;
    bool output_estimations = false;
};

extern Param param;

extern Param parseArgs(int nargs, char **args);

// ==================== IO
template<class T>
inline void show_vector(const std::string &_header, const std::vector<T> &_vec) {
    if (_vec.empty()) {
        std::cout << "Empty Vector." << std::endl;
    } else {
        std::cout << std::endl << _header;
        bool identical = true;
        const T &elem = _vec.front();
        std::for_each(_vec.begin(), _vec.end(), [&](const T &e) { identical &= (e == elem); });
        if (identical) {
            std::cout << "\tSize of the Vector: " << _vec.size() << "\t Value of Each Element: " << elem;
        } else {
            std::cout << std::endl;
            std::copy(begin(_vec), end(_vec), std::ostream_iterator<T>(std::cout, "\t"));
        }
        std::cout << std::endl;
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

template<class FLOAT_TYPE>
inline void
output_feature(const std::vector<FLOAT_TYPE> &_value_vec, std::vector<float> &out_matrix,
    const std::string &_out_path, const unsigned int split_num,
    const unsigned long idx_feat, const unsigned long feat_num, const VertexIdType &_node_num) {
    // Decide split
    unsigned long spt_size = feat_num / split_num;  // size per split
    unsigned long spt = idx_feat / spt_size;    // index of split
    unsigned long idxf = idx_feat % spt_size;   // index of feature in split
    // Save [F/split_num, n] array of all nodes to out_matrix
    for (VertexIdType id = 0; id < _node_num; ++id) {
        out_matrix[idxf*_node_num+id] = _value_vec[id];
    }

    // Save to .npy file
    if ((idxf + 1) % spt_size == 0 || idx_feat + 1 == feat_num) {
        std::string spt_path;
        if (split_num == 1) {
            spt_path = _out_path + ".npy";
        } else {
            spt_path = _out_path + "_" + std::to_string(spt) + ".npy";
        }
        std::array<long unsigned, 2> res_shape {{spt_size, _node_num}};
        npy::SaveArrayAsNumpy(spt_path, false, res_shape.size(), res_shape.data(), out_matrix);
        printf("Feature saved: %s\n", spt_path.c_str());
    }
}


inline unsigned int
load_query(std::vector<VertexIdType> &Vt_nodes, std::string query_path){
    std::ifstream query_file(query_path);
    if (query_file.good() == false) {
        printf("File Not Exists.\n");
        exit(1);
    }
    for (VertexIdType sid; (query_file >> sid);) {
        Vt_nodes.emplace_back(sid);
    }
    if (Vt_nodes.empty()) {
        printf("Error! Empty File\n");
    }
    query_file.close();
    std::cout << "Query size: " << Vt_nodes.size() << std::endl;
    return Vt_nodes.size();
}

inline void
load_feature(std::vector<VertexIdType> &Vt_nodes, std::vector<std::vector<float>> &feature_matrix,
    std::string feature_path, const unsigned int split_num) {
    VertexIdType index = 0;
    VertexIdType sumrow = 0;
    std::vector<unsigned long> shape;
    bool fortran_order;
    std::vector<float> arr_np;
    // feature_matrix = std::vector<std::vector<float>>(500,std::vector<float>(Vt_nodes.size()));

    for (int spt = 0; spt < split_num; spt ++) {
        std::string spt_path = feature_path;
        if (split_num > 1) {
            spt_path = spt_path.insert(feature_path.length() - 4, std::to_string(spt));
        }
        shape.clear();
        arr_np.clear();
        npy::LoadArrayFromNumpy(spt_path, shape, fortran_order, arr_np);
        auto feature_data = arr_np.data();
        int nrows = shape[0];   // node num
        int ncols = shape[1];   // feature num
        std::cout<<"Input shape: "<<nrows<<" "<<ncols<<std::endl;

        for (int row = 0; row <nrows; row ++) {
            if (sumrow + row == Vt_nodes[index]) {
                index++;
                std::vector<float> feature_array(feature_data+row*ncols, feature_data+row*ncols+ncols);
                feature_matrix.emplace_back(feature_array);
            }
        }
        sumrow += nrows;
        // std::cout << "  sumrow " << sumrow << " index " << index << std::endl;
    }
    std::cout<<"Feature size: "<<feature_matrix.size()<<" "<<feature_matrix[0].size()<<std::endl;
}

// ==================== Reuse
inline double
calc_L1_residue(std::vector<double> &V1, std::vector<double> &V2, double pace = 1.0){
    int index;
    double used_sum = 0;
    double theta;
    std::vector<std::pair<double, int>> ratio_and_index;
    for(int i = 0; i < V1.size(); i++){
        if(V2[i]!=0) {
            theta =  V1[i] / V2[i];
            ratio_and_index.push_back(std::make_pair( theta , i));
        }
    }
    std::sort ( ratio_and_index.begin(), ratio_and_index.end(),
                [](std::pair<double, int> a, std::pair<double, int> b) {
                    return abs(a.first) < abs(b.first); });
    for(int i = 0; i < ratio_and_index.size(); i+=1){
        theta = ratio_and_index[i].first;
        index = ratio_and_index[i].second;
        used_sum += abs(V2[index]);
        if(used_sum > 0.5 || index == -1)
            break;
    }

    double orig_sum = 0;
    double diff_sum = 0;
    for(int i = 0; i < V1.size(); i++){
        orig_sum += abs(V1[i]);
        diff_sum += abs(V1[i] - theta * V2[i] * pace);
    }
    if (diff_sum > orig_sum)
        return 0;

    for(int i = 0; i < V1.size(); i++){
        V1[i] = V1[i] - theta * V2[i] * pace;
    }
    // printf("theta: %.6f, residue: %.6f\n", theta, diff_sum);
    return theta * pace;
}

inline double
calc_L2_residue(std::vector<double> &V1, std::vector<double> &V2, double pace = 1.0){
    double prd = 0;
    double sum2 = 0;
    for (int i = 0; i < V1.size(); i++){
        prd += V1[i] * V2[i];
        sum2 += V2[i] * V2[i];
    }
    double theta = prd / sum2;

    double orig_sum = 0;
    double diff_sum = 0;
    for(int i = 0; i < V1.size(); i++){
        orig_sum += abs(V1[i]);
        diff_sum += abs(V1[i] - theta * V2[i] * pace);
    }
    if (diff_sum > orig_sum)
        return 0;

    for(int i = 0; i < V1.size(); i++){
        V1[i] = V1[i] - theta * V2[i] * pace;
    }
    // printf("theta: %.6f, residue: %.6f\n", theta, diff_sum);
    return theta * pace;
}

inline double
calc_L1_distance(std::vector<double> &V1, std::vector<double> &V2){
    double distance = 0;
    for (int i = 0; i < V1.size(); i++){
        distance += abs(V1[i] - V2[i]);
    }
    return distance;
}

inline double
calc_L2_distance(std::vector<double> &V1, std::vector<double> &V2){
    double distance = 0;
    double prd = 0;
    double sum1 = 0;
    double sum2 = 0;
    for (int i = 0; i < V1.size(); i++){
        prd += V1[i] * V2[i];
        sum1 += V1[i] * V1[i];
        sum2 += V2[i] * V2[i];
    }
    // distance = 1 - prd / (sqrt(sum1) * sqrt(sum2));
    distance = 1 - abs(prd / (sqrt(sum1) * sqrt(sum2)));
    return distance;
}

inline void
get_base_with_norm(std::vector<std::vector<double >> &seed_matrix, std::vector<std::vector<double>> &base_matrix,
                   std::vector<int> &base_vex, double base_ratio)
{
    std::vector<std::pair<int, double>> min_L1_counter(seed_matrix.size(), std::make_pair(0, 0));
    for (int i = 0; i < seed_matrix.size(); i++) {
        double L1_dis_min = seed_matrix.size();
        int min_L1_idx = -1;
        for (int j = 0; j < seed_matrix.size(); j++) {
            if(i!=j){
                double L1_dis = calc_L1_distance(seed_matrix[i], seed_matrix[j]);
                if (L1_dis_min > L1_dis){
                    L1_dis_min = L1_dis;
                    min_L1_idx = j;
                }
            }
        }
        // printf("id: %4d, dis: %.8f, tar: %4d\n", i, L1_dis_min, min_L1_idx);
        if(min_L1_idx < 0 || min_L1_idx > seed_matrix.size()) continue;
        min_L1_counter[min_L1_idx].first = min_L1_idx;
        min_L1_counter[min_L1_idx].second += 1 - L1_dis_min;
    }

    std::sort(min_L1_counter.begin(), min_L1_counter.end(), [](std::pair<int, double> a1, std::pair<int, double>a2){
        return a1.second > a2.second;
    });
    int base_size = seed_matrix.size() * base_ratio;
    if (base_size < 3) {
        base_size = 3;
    }
    MSG(base_size);
    for(int i = 0; i < base_size; i++){
        base_matrix.push_back(seed_matrix[min_L1_counter[i].first]);
        base_vex.push_back(min_L1_counter[i].first);
    }
}

inline std::vector<double>
feature_reuse(std::vector<double> &raw_seed, std::vector<std::vector<double >> &base_matrix){
    std::vector<double> base_weight(base_matrix.size(), 0.0);
    for(double delta = 1; delta <= 16; delta *= 2){
        double L1_dis_min = base_matrix.size();
        int min_L1_idx = 0;
        for(int j = 0; j < base_matrix.size(); j++){
            double L1_dis = calc_L1_distance(raw_seed, base_matrix[j]);
            if(L1_dis_min > L1_dis){
                L1_dis_min = L1_dis;
                min_L1_idx = j;
            }
        }
        double theta = calc_L1_residue(raw_seed, base_matrix[min_L1_idx], 1.0);
        if (abs(theta) / delta < 1 / 16) break;
        base_weight[min_L1_idx] += theta;
    }
    return base_weight;
}

template<class T>
inline double add_up_vector(std::vector<T> Vec){
    double sum = 0;
    for(auto a : Vec)
        sum += a;
    return sum;
}

inline long get_proc_memory(){
    struct rusage r_usage;
    getrusage(RUSAGE_SELF,&r_usage);
    return r_usage.ru_maxrss;
}
#endif //SPEEDPPR_HELPERFUNCTIONS_H
