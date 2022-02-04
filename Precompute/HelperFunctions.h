

#ifndef SPEEDPPR_HELPERFUNCTIONS_H
#define SPEEDPPR_HELPERFUNCTIONS_H

#define MSG(...)  { std::cout << #__VA_ARGS__ << ":" << (__VA_ARGS__) << std::endl; }


#include <string>
#include <vector>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <sstream>
#include "BasicDefinition.h"
#include"cnpy.h"

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


// inline void
// show(const std::vector<PageRankScoreType> &pi, const unsigned int &_numOfVertices, unsigned int &_num_to_show) {
//     if (_num_to_show > pi.size()) {
//         _num_to_show = pi.size();
//     }
//     std::priority_queue<PageRankScoreType,
//             std::vector<PageRankScoreType>,
//             std::greater<> > top_k_queue;
//     for (VertexIdType id = 0; id < _numOfVertices; ++id) {
//         if (top_k_queue.size() < _num_to_show) {
//             top_k_queue.push(pi[id]);
//         } else if (top_k_queue.top() < pi[id]) {
//             top_k_queue.pop();
//             top_k_queue.push(pi[id]);
//         }
//     }
//     printf("Top Page Rank Scores.\n");
//     std::vector<PageRankScoreType> top_k_list;
//     while (top_k_queue.empty() == false) {
//         top_k_list.emplace_back(top_k_queue.top());
//         top_k_queue.pop();
//     }
//     while (top_k_list.empty() == false) {
//         printf("%.13f\n", top_k_list.back());
//         top_k_list.pop_back();
//     }
// }

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
save_answer(const std::vector<FLOAT_TYPE> &_means, const VertexIdType &_numOfVertices, const std::string &_file_name) {
    std::ofstream file(_file_name);
    if (file.is_open()) {
        std::vector<IdScorePair<FLOAT_TYPE>> pairs(_numOfVertices, 0);
        for (VertexIdType id = 0; id < _numOfVertices; ++id) {
            pairs[id] = {id, _means[id]};
        }
        std::sort(pairs.begin(), pairs.end(), IdScorePairComparatorGreater<FLOAT_TYPE>());
        file.precision(std::numeric_limits<FLOAT_TYPE>::max_digits10);
        for (const auto &pair : pairs) {
            if (pair.score > 0) {
                file << pair.id << "\t" << pair.score << "\n";
            }
        }
        file.close();
    } else {
        printf("ERROR IN " __FILE__ " LINE %u\n", __LINE__);
        printf("FILE NOT EXISTS.\n%s\n", _file_name.c_str());
        exit(1);
    }
}

template<class FLOAT_TYPE>
inline void
save_answerk(const std::vector<FLOAT_TYPE> &_means, const VertexIdType &_numOfVertices, const std::string &_file_node, const std::string &_file_score) {
    // Save [query_size, k] top-k array to (node, score) .txt file pairs
    std::ofstream file;
    std::ofstream file_s;
    file.open( _file_node.c_str());
    file_s.open( _file_score.c_str());

    if (file.is_open() && file_s.is_open()) {
        std::vector<IdScorePair<FLOAT_TYPE>> pairs(_numOfVertices, 0);
        for (VertexIdType id = 0; id < _numOfVertices; ++id) {
            pairs[id] = {id, _means[id]};
        }
        std::sort(pairs.begin(), pairs.end(), IdScorePairComparatorGreater<FLOAT_TYPE>());
        file_s.precision(std::numeric_limits<FLOAT_TYPE>::max_digits10);
        int k = 0;
        for (const auto &pair : pairs) {
            if (pair.score > 0) {
                file << pair.id << "\t";
                file_s << pair.score << "\t";
            }
            k++;
            if (k >= 500) {
                break;
            }
        }
        file << "\n";
        file_s << "\n";
        file.close();
        file_s.close();
    } else {
        printf("ERROR IN " __FILE__ " LINE %u\n", __LINE__);
        printf("FILE NOT EXISTS.\n%s\n", _file_node.c_str());
        exit(1);
    }
}

template<class FLOAT_TYPE>
inline void
save_answer_Vt(const std::vector<FLOAT_TYPE> &_means, const VertexIdType &_numOfVertices, const std::string &_file_node, const std::string &_file_score, const std::vector<VertexIdType> Vt_nodes) {
    // Save [F, query_size] array to (node, score) .txt file pairs
    std::ofstream file;
    std::ofstream file_s;
    file.open( _file_node.c_str(), std::ios_base::app);
    file_s.open( _file_score.c_str(), std::ios_base::app);

    if (file.is_open() && file_s.is_open()) {
        std::vector<IdScorePair<FLOAT_TYPE>> pairs(_numOfVertices, 0);
        for (VertexIdType id = 0; id < _numOfVertices; ++id) {
            pairs[id] = {id, _means[id]};
        }
        //std::sort(pairs.begin(), pairs.end(), IdScorePairComparatorGreater<FLOAT_TYPE>());
        file_s.precision(std::numeric_limits<FLOAT_TYPE>::max_digits10);
        VertexIdType index = 0;
        for (const auto &pair : pairs) {
            if (pair.id == Vt_nodes[index]) {
                file << pair.id << "\t";
                file_s << pair.score << "\t";
                index++;
            }
        }
        file << "\n";
        file_s << "\n";
        file.close();
        file_s.close();
    } else {
        printf("ERROR IN " __FILE__ " LINE %u\n", __LINE__);
        printf("FILE NOT EXISTS.\n%s\n", _file_node.c_str());
        exit(1);
    }
}

template<class FLOAT_TYPE>
inline void
save_answer_Vt_np(const std::vector<FLOAT_TYPE> &_means, const VertexIdType &_numOfVertices,
    const std::string &_file_node, const std::string &_file_score,
    const std::vector<VertexIdType> Vt_nodes, std::vector<float> &res_matrix,
    const int idxOfFeature, const unsigned int numOfFeature) {
    // Save [F, n] array of all nodes to res_matrix
    unsigned int numOfQueries = Vt_nodes.size();
    VertexIdType index = 0;
    for (VertexIdType id = 0; id < _numOfVertices; ++id) {
        if (id == Vt_nodes[index]) {
            res_matrix[idxOfFeature*numOfQueries+index] = _means[id];
            index++;
        }
    }
    // Save to .npy file
    if (idxOfFeature == numOfFeature - 1) {
        cnpy::npy_save(_file_score.c_str(), &res_matrix[0], {numOfFeature, numOfQueries}, "w");
    }
}

template<class FLOAT_TYPE>
inline void
save_answer_n_np(const std::vector<FLOAT_TYPE> &_means, const VertexIdType &_numOfVertices,
    const std::string &_file_node, const std::string &_file_score,
    const std::vector<VertexIdType> Vt_nodes, std::vector<float> &res_matrix,
    const int idxOfFeature, const unsigned int numOfFeature) {
    // Save [F, n] array of all nodes to res_matrix
    VertexIdType index = 0;
    for (VertexIdType id = 0; id < _numOfVertices; ++id) {
        res_matrix[idxOfFeature*_numOfVertices+id] = _means[id];
    }
    // Save to .npy file
    if (idxOfFeature == numOfFeature - 1) {
        cnpy::npy_save(_file_score.c_str(), &res_matrix[0], {numOfFeature, _numOfVertices}, "w");
    }
}

template<class FLOAT_TYPE>
inline void
save_answer_n_np_dual(const std::vector<FLOAT_TYPE> &_means, const VertexIdType &_numOfVertices,
    const std::string &_file_node, const std::string &_file_score,
    const std::vector<VertexIdType> Vt_nodes, std::vector<float> &res_matrix,
    const int idxOfFeature, const unsigned int numOfFeature, const int flag) {
    // Save [F, n] array of all nodes to res_matrix
    VertexIdType index = 0;
    for (VertexIdType id = 0; id < _numOfVertices; ++id) {
        if (flag == 0){
            res_matrix[idxOfFeature*_numOfVertices+id] = -_means[id];
        }else{
            res_matrix[idxOfFeature*_numOfVertices+id] += _means[id];
        }
    }
    // Save to .npy file
    if (idxOfFeature == numOfFeature - 1 && flag == 1) {
        cnpy::npy_save(_file_score.c_str(), &res_matrix[0], {numOfFeature, _numOfVertices}, "w");
    }
}

struct Param {
    std::string graph_file;
    std::string query_file;
    std::string answer_folder;
    std::string index_file;
    std::string meta_file;
    std::string vt_file;
    std::string feature_file;
    std::string graph_binary_file;
    std::string algorithm = "PowItr";
    std::string output_folder;
    std::string estimation_folder;
    double epsilon = 0.5;
    double alpha = 0.2;
    double l1_error = 0.00000001;
    unsigned int num_top_k = 1;
    unsigned int query_size = 0;
    bool with_idx = false;
    bool is_top_k = false;
    bool is_undirected_graph = false;
    bool specified_l1_error = false;
    bool output_estimations = false;
};

extern Param param;

extern Param parseArgs(int nargs, char **args);

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

inline void
load_Vt_and_features(std::vector<VertexIdType> &Vt_nodes, std::vector<std::vector<float>> &feature_matrix, std::string query_file_path, std::string feature_file_path){
    std::ifstream vt_file(query_file_path);
    if (vt_file.good() == false) {
        printf("File Not Exists.\n");
        exit(1);
    }
    for (VertexIdType sid; (vt_file >> sid);) {
        Vt_nodes.emplace_back(sid);
    }
    if (Vt_nodes.empty()) {
        printf("Error. Empty File\n");
    }
    vt_file.close();

    std::ifstream file(feature_file_path);
    if (file.good() == false) {
        printf("File Not Exists.\n");
        exit(1);
    }
    std::string str;
    while(getline(file, str))
    {
        std::istringstream ss(str);
        std::vector<float> feature_array;
        float feature;
        while(ss >> feature)
        {
            feature_array.emplace_back(feature);
        }
        feature_matrix.emplace_back(feature_array);
    }
    if (Vt_nodes.empty()) {
        printf("Error. Empty File\n");
    }
    file.close();
    std::cout<<"Feature size: "<<feature_matrix[0].size()<<std::endl;

    //column normalization

    for(VertexIdType i = 0; i < feature_matrix[0].size(); i++){
        //find min

        VertexIdType min_index = 0;
        for(VertexIdType j = 1; j < feature_matrix.size(); j++){
            if(feature_matrix[j][i] < feature_matrix[min_index][i]){
                min_index = j;
            }
        }
        double min = abs(feature_matrix[min_index][i]);
        for(VertexIdType j = 0; j < feature_matrix.size(); j++){
            feature_matrix[j][i] = abs(feature_matrix[j][i]);
            //feature_matrix[j][i] = feature_matrix[j][i] + min;
        }
        double feature_sum = 0.0;
        for(VertexIdType j = 0; j < feature_matrix.size(); j++){
            feature_sum += abs(feature_matrix[j][i]);
        }
        for(VertexIdType j = 0; j < feature_matrix.size(); j++){
            feature_matrix[j][i] = feature_matrix[j][i] / feature_sum;
        }

    }
/*
    std::vector<double> vertex_sums;
    for(auto vertex_feature : feature_matrix){
        double vertex_sum = 0.0;
        for(auto r : vertex_feature){
            vertex_sum += r;
        }
        vertex_sums.push_back(vertex_sum);
    }
    std::sort(vertex_sums.rbegin(), vertex_sums.rend());
    double sum = 0;
    for(int i = 0; i < vertex_sums.size(); i++){
        MSG(i);
        MSG(vertex_sums[i]);
        sum += vertex_sums[i];
    }
    MSG(sum);
*/

}

inline void
load_Vt_and_features(std::vector<VertexIdType> &Vt_nodes, std::vector<std::vector<float>> &feature_matrix, std::string query_file_path, std::string feature_file_path, int flag){
    std::ifstream vt_file(query_file_path);
    if (vt_file.good() == false) {
        printf("File Not Exists.\n");
        exit(1);
    }
    for (VertexIdType sid; (vt_file >> sid);) {
        Vt_nodes.emplace_back(sid);
    }
    if (Vt_nodes.empty()) {
        printf("Error. Empty File\n");
    }
    vt_file.close();

    std::ifstream file(feature_file_path);
    if (file.good() == false) {
        printf("File Not Exists.\n");
        exit(1);
    }
    std::string str;
    while(getline(file, str))
    {
        std::istringstream ss(str);
        std::vector<float> feature_array;
        float feature;
        while(ss >> feature)
        {
            if(flag == 0 && feature > 0)
                feature_array.emplace_back(feature);
            else if(flag == 1 && feature < 0)
                feature_array.emplace_back(-feature);
            else
                feature_array.emplace_back(0);
        }
        feature_matrix.emplace_back(feature_array);
    }
    if (Vt_nodes.empty()) {
        printf("Error. Empty File\n");
    }
    file.close();
    std::cout<<"Feature size: "<<feature_matrix[0].size()<<std::endl;

    //column normalization

    for(VertexIdType i = 0; i < feature_matrix[0].size(); i++){
        //find min

        VertexIdType min_index = 0;
        for(VertexIdType j = 1; j < feature_matrix.size(); j++){
            if(feature_matrix[j][i] < feature_matrix[min_index][i]){
                min_index = j;
            }
        }
        double min = abs(feature_matrix[min_index][i]);
        for(VertexIdType j = 0; j < feature_matrix.size(); j++){
            feature_matrix[j][i] = feature_matrix[j][i];
            //feature_matrix[j][i] = feature_matrix[j][i] + min;
        }
        double feature_sum = 0.0;
        for(VertexIdType j = 0; j < feature_matrix.size(); j++){
            feature_sum += abs(feature_matrix[j][i]);
        }
        for(VertexIdType j = 0; j < feature_matrix.size(); j++){
            feature_matrix[j][i] = feature_matrix[j][i] / feature_sum;
        }

    }
}

inline void
load_Vt_and_features_np(std::vector<VertexIdType> &Vt_nodes, std::vector<std::vector<float>> &feature_matrix, std::string query_file_path, std::string feature_file_path){
    std::ifstream vt_file(query_file_path);
    if (vt_file.good() == false) {
        printf("File Not Exists.\n");
        exit(1);
    }
    for (VertexIdType sid; (vt_file >> sid);) {
        Vt_nodes.emplace_back(sid);
    }
    if (Vt_nodes.empty()) {
        printf("Error. Empty File\n");
    }
    vt_file.close();

    std::cout << "Loading feature... ";
    cnpy::NpyArray arr_np = cnpy::npz_load(feature_file_path, "arr_0");
    auto feature_data = arr_np.data<float>();
    int nrows = arr_np.shape [0];   // node size
    int ncols = arr_np.shape [1];   // feature size
    std::cout << nrows << ' ' << ncols << std::endl;
    std::sort(Vt_nodes.begin(),Vt_nodes.end());

    VertexIdType index = 0;
    for(int row = 0; row <nrows; row ++){
        if (row == Vt_nodes[index]) {
            index++;
            std::vector<float> feature_array(feature_data+row*ncols, feature_data+row*ncols+ncols);
            feature_matrix.emplace_back(feature_array);
            //output_vector(feature_array, "feat.txt");
        }
    }

    for(VertexIdType i = 0; i < feature_matrix[0].size(); i++){
        //find min
        VertexIdType min_index = 0;
        for(VertexIdType j = 1; j < feature_matrix.size(); j++){
            if(feature_matrix[j][i] < feature_matrix[min_index][i]){
                min_index = j;
            }
        }
        double feature_sum = 0.0;
        for(VertexIdType j = 0; j < feature_matrix.size(); j++){
            feature_matrix[j][i] = abs(feature_matrix[j][i]);
            feature_sum += abs(feature_matrix[j][i]);
        }
        for(VertexIdType j = 0; j < feature_matrix.size(); j++){
            if(feature_sum > 0)
                feature_matrix[j][i] = feature_matrix[j][i] / feature_sum;
        }

    }
    std::cout<<"Feature size: "<<feature_matrix[0].size()<<std::endl;
}

inline void
load_Vt_and_features_l(std::vector<VertexIdType> &Vt_nodes, std::vector<std::vector<float>> &feature_matrix,
        std::string query_file_path, std::string feature_file_path,
        const int Vt_size){
    std::ifstream vt_file(query_file_path);
    if (vt_file.good() == false) {
        printf("File Not Exists.\n");
        exit(1);
    }
    for (int i = 0; i < Vt_size; i++) {
        VertexIdType sid;
        vt_file >> sid;
        Vt_nodes.emplace_back(sid);
    }
    if (Vt_nodes.empty()) {
        printf("Error. Empty File\n");
    }
    vt_file.close();

    std::cout << "Loading feature... ";
    cnpy::NpyArray arr_np = cnpy::npz_load(feature_file_path, "arr_0");
    auto feature_data = arr_np.data<float>();
    int nrows = arr_np.shape [0];   // node size
    int ncols = arr_np.shape [1];   // feature size
    std::cout << nrows << ' ' << ncols << std::endl;
    std::sort(Vt_nodes.begin(),Vt_nodes.end());

    VertexIdType index = 0;
    for(int row = 0; row <nrows; row ++){
        if (row == Vt_nodes[index]) {
            index++;
            std::vector<float> feature_array(feature_data+row*ncols, feature_data+row*ncols+ncols);
            feature_matrix.emplace_back(feature_array);
            //output_vector(feature_array, "feat.txt");
        }
    }

    for(VertexIdType i = 0; i < feature_matrix[0].size(); i++){
        //find min
        VertexIdType min_index = 0;
        for(VertexIdType j = 1; j < feature_matrix.size(); j++){
            if(feature_matrix[j][i] < feature_matrix[min_index][i]){
                min_index = j;
            }
        }
        double feature_sum = 0.0;
        for(VertexIdType j = 0; j < feature_matrix.size(); j++){
            feature_matrix[j][i] = abs(feature_matrix[j][i]);
            feature_sum += abs(feature_matrix[j][i]);
        }
        for(VertexIdType j = 0; j < feature_matrix.size(); j++){
            if(feature_sum > 0)
                feature_matrix[j][i] = feature_matrix[j][i] / feature_sum;
        }

    }
    std::cout<<"Feature size: "<<feature_matrix[0].size()<<std::endl;
}

inline void
load_Vt_and_features_np(std::vector<VertexIdType> &Vt_nodes, std::vector<std::vector<float>> &feature_matrix, std::string query_file_path, std::string feature_file_path, int flag){
    std::ifstream vt_file(query_file_path);
    if (vt_file.good() == false) {
        printf("File Not Exists.\n");
        exit(1);
    }
    for (VertexIdType sid; (vt_file >> sid);) {
        Vt_nodes.emplace_back(sid);
    }
    if (Vt_nodes.empty()) {
        printf("Error. Empty File\n");
    }
    vt_file.close();

    std::cout << "Loading feature... ";
    cnpy::NpyArray arr_np = cnpy::npz_load(feature_file_path, "arr_0");
    auto feature_data = arr_np.data<float>();
    int nrows = arr_np.shape [0];   // node size
    int ncols = arr_np.shape [1];   // feature size
    std::cout << nrows << ' ' << ncols << std::endl;

    VertexIdType index = 0;
    for(int row = 0; row <nrows; row ++){
        if (row == Vt_nodes[index]) {
            index++;
            std::vector<float> feature_array(feature_data+row*ncols, feature_data+row*ncols+ncols);
            for(int i = 0; i < feature_array.size(); i++){
                if(feature_array[i] <= 0 && flag == 0){
                    feature_array[i] = 0;
                }else if(feature_array[i] >= 0 && flag == 1){
                    feature_array[i] = 0;
                }
            }
            feature_matrix.emplace_back(feature_array);
        }
    }
    std::cout<<"Feature size: "<<feature_matrix[0].size()<<std::endl;
}

inline double
calc_left_residue(std::vector<double> &V1, std::vector<double> &V2, double pace = 1.0){
    int index;
    std::vector<int> used_index;
    double used_sum = 0;
    double theta;
    while(1){
        theta = 1.0;
        index = -1;
        for(int i = 0; i < V1.size(); i++){
            if(theta > V1[i] / V2[i] && std::find(used_index.begin(), used_index.end(), i) == used_index.end()){
                theta = V1[i] / V2[i];
                index = i;
            }
        }
        if(used_sum + V2[index] > 0.5 || index == -1)
            break;
        used_sum += V2[index];
        used_index.push_back(index);
    }
    double diff_sum = 0;
    for(int i = 0; i < V1.size(); i++){
        diff_sum += abs(V1[i] - theta * V2[i] * pace);
        V1[i] = V1[i] - theta * V2[i] * pace;
    }
    std::cout<<"gamma: "<<theta<<"\tleft residue:"<<diff_sum<<std::endl;
    return theta;
}

inline double
calc_left_rsum(std::vector<double> &V1, std::vector<double> &V2){
    int index;
    std::vector<int> used_index;
    std::vector<double> left_res;
    double used_sum = 0;
    double gamma;
    while(1){
        gamma = 1.0;
        for(int i = 0; i < V1.size(); i++){
            if(gamma > V1[i] / V2[i] && std::find(used_index.begin(), used_index.end(), i) == used_index.end()){
                gamma = V1[i] / V2[i];
                index = i;
            }
        }
        if(used_sum + V2[index] > 0.5)
            break;
        used_sum += V2[index];
        used_index.push_back(index);
    }

    double diff_sum = 0;
    for(int i = 0; i < V1.size(); i++){
        diff_sum += abs(V1[i] - gamma * V2[i]);
        left_res.push_back(V1[i] - gamma * V2[i]);
    }
    return diff_sum;
}


inline double
calc_L1_distance(std::vector<double> &V1, std::vector<double> &V2){

    double distance = 0;
    double rsum_1 = 0;
    for(double v : V1)
        rsum_1 += v;

    double rsum_2 = 0;
    for(double v : V2)
        rsum_2 += v;


    for(int i = 0; i < V1.size(); i++){
        distance += abs(V1[i]/rsum_1 - V2[i]/rsum_2);
    }
    return distance;
}

inline void
get_base_with_L1(std::vector<std::vector<double >> &seed_matrix, std::vector<std::vector<double>> &base_matrix, std::vector<int> &base_vex)
{
    std::vector<std::pair<int, double>> min_L1_counter(seed_matrix.size(), std::make_pair(0, 0));
    for (int i = 0; i < seed_matrix.size(); i++) {
        double L1_dis_min = 10.0;
        int min_L1_idx;
        double sum = 0;
        for(double r : seed_matrix[i])
            sum += r;
        MSG(sum);
        for (int j = 0; j < seed_matrix.size(); j++) {
            if(i!=j){
                double L1_dis = calc_L1_distance(seed_matrix[i], seed_matrix[j]);
                if(L1_dis_min > L1_dis){
                    L1_dis_min = L1_dis;
                    min_L1_idx = j;
                }
            }
        }
        std::cout<<i<<": "<<L1_dis_min<<": "<<min_L1_idx<<std::endl;
        /*
        if(i > 2)
            feature_matrix_new.push_back(calc_left_residue(seed_matrix[i], seed_matrix[min_L1_idx]));
        else
            feature_matrix_new.push_back(seed_matrix[i]);*/
        if(min_L1_idx < 0 || min_L1_idx > seed_matrix.size()) continue;
        min_L1_counter[min_L1_idx].first = min_L1_idx;
        min_L1_counter[min_L1_idx].second += 1 - L1_dis_min;
    }

    std::sort(min_L1_counter.begin(), min_L1_counter.end(), [](std::pair<int, double> a1, std::pair<int, double>a2){
        return a1.second > a2.second;
    });
    for(int i = 0; i < seed_matrix.size()/30; i++){
        base_matrix.push_back(seed_matrix[min_L1_counter[i].first]);
        base_vex.push_back(min_L1_counter[i].first);
    }
}

inline std::vector<double> feature_reuse(std::vector<double> &raw_seed, std::vector<std::vector<double >> &base_matrix){
    std::vector<double> base_weight(base_matrix.size(), 0.0);;
    for(double pace = 1; pace <= 1; pace+= 1){
        double L1_dis_min = 10.0;
        int min_L1_idx;
        for(int j = 0; j < base_matrix.size(); j++){
            double L1_dis = calc_L1_distance(raw_seed, base_matrix[j]);
            if(L1_dis_min > L1_dis){
                L1_dis_min = L1_dis;
                min_L1_idx = j;
            }
        }
        base_weight[min_L1_idx] += calc_left_residue(raw_seed, base_matrix[min_L1_idx], pace);
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
