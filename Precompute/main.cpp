// Ref: https://github.com/wuhao-wu-jiang/Personalized-PageRank
// article{WGWZ21,
//     title={Unifying the Global and Local Approaches: An Efficient Power Iteration with Forward Push},
//     author={Wu, Hao and Gan, Junhao and Wei, Zhewei and Zhang, Rui},
//     journal={arXiv preprint arXiv:2101.03652}}
#include "HelperFunctions.h"
#include "Graph.h"
#include "BatchRandomWalk.h"
#include "SpeedPPR.h"
#include "CleanGraph.h"
#include <unistd.h>
#include <thread>

// void featpush(
//     unsigned long feat_left, unsigned long feat_right,
//     const VertexIdType NumOfVertices) {
// }

int main(int argc, char **argv) {
    param = parseArgs(argc, argv);
    SFMT64::initialize(param.seed);
    Graph graph;
    graph.set_alpha(param.alpha);

    if (param.algorithm == "clean_graph") {
        CleanGraph cleaner;
        if (param.is_undirected_graph) {
            std::string output_file = param.output_folder + "/" + "edge_duplicated_graph.txt";
            cleaner.duplicate_edges(param.graph_file, output_file);
            cleaner.clean_graph(output_file, param.output_folder);
        } else {
            cleaner.clean_graph(param.graph_file, param.output_folder);
        }
        return 0;
    } else if (!param.graph_binary_file.empty()) {
        graph.read_binary(param.meta_file, param.graph_binary_file);
        assert(graph.get_neighbor_list_start_pos(graph.get_dummy_id()) ==
               graph.get_neighbor_list_start_pos(graph.get_dummy_id() + 1));
    } else {
        printf("Error in" __FILE__ " LINE %d. Fail to load the graph.\n", __LINE__);
        return 0;
    }

    if (param.algorithm == "featpush"){
        std::vector<VertexIdType> Vt_nodes; // list of queried nodes
        std::vector<std::vector<float>> feature_matrix;
        size_t Vt_num = load_query(Vt_nodes, param.query_file);
        size_t feat_size = load_feature(Vt_nodes, feature_matrix, param.feature_file, param.split_num); // feature size
        double total_time = 0;
        graph.reset_set_dummy_neighbor();
        graph.fill_dead_end_neighbor_with_id();
        VertexIdType V_num = graph.getNumOfVertices();
        class SpeedPPR speedPPR(graph);
        WalkCache walkCache(graph);
        size_t spt_size = (feat_size + param.split_num - 1) / param.split_num;    // feature size per split (ceiling)
        size_t out_size = spt_size * V_num; // length of output matrix
        std::vector<float> out_matrix(out_size);
        printf("Result size: %ld \n", out_matrix.size());

        for (size_t spt_left = 0; spt_left < feat_size; spt_left += spt_size) {
            size_t spt_right = std::min(feat_size, spt_left + spt_size);
            for (size_t i = spt_left; i < spt_right; i++) {
                // printf("ID: %4d \n", i);
                SpeedPPR::WHOLE_GRAPH_STRUCTURE<double> whole_graph_structure(V_num);
                std::vector<double> seed(V_num, 0.0);
                for (size_t j = 0; j < Vt_num; j++) {
                    seed[Vt_nodes[j]] = feature_matrix[j][i];
                }

                double time_start = getCurrentTime();
                speedPPR.compute_approximate_page_rank_3(whole_graph_structure, seed, param.epsilon, param.alpha,
                                                         1.0 / V_num, walkCache);
                total_time += getCurrentTime() - time_start;

                // Save [F/split_num, n] array of all nodes to out_matrix
                size_t idxf = i % spt_size;   // index of feature in split
                for (size_t j = 0; j < V_num; j++) {
                    out_matrix[idxf*V_num+j] = whole_graph_structure.means[j];
                }
            }

            if (param.output_estimations) {
                size_t spt = spt_left / spt_size;    // index of split
                std::stringstream res_file;
                if (param.split_num <= 1) {
                    res_file << param.estimation_folder << "/score_" << param.alpha << '_' << param.epsilon << ".npy";
                } else {
                    res_file << param.estimation_folder << "/score_" << param.alpha << '_' << param.epsilon << "_" << spt << ".npy";
                }
                output_feature(out_matrix, res_file.str(), spt_right - spt_left, V_num);
            }
        }

        printf("Mem: %ld MB\n", get_proc_memory()/1000);
        printf("Total Time: %.6f, Average: %.12f\n", total_time, total_time / feat_size);
    } else if (param.algorithm == "featreuse"){
        std::vector<VertexIdType> Vt_nodes; // list of queried nodes
        std::vector<std::vector<float>> feature_matrix;
        size_t Vt_num = load_query(Vt_nodes, param.query_file);
        size_t feat_size = load_feature(Vt_nodes, feature_matrix, param.feature_file, param.split_num); // feature size
        double total_time = 0;
        graph.reset_set_dummy_neighbor();
        graph.fill_dead_end_neighbor_with_id();
        VertexIdType V_num = graph.getNumOfVertices();
        class SpeedPPR speedPPR(graph);
        WalkCache walkCache(graph);
        size_t spt_size = (feat_size + param.split_num - 1) / param.split_num;    // feature size per split (ceiling)
        size_t out_size = spt_size * V_num; // length of output matrix
        std::vector<float> out_matrix(out_size);
        printf("Result size: %ld \n", out_matrix.size());

        // Select base
        std::vector<std::vector<double>> seed_matrix;
        for (int i = 0; i < feature_matrix[0].size(); i++) {
            std::vector<double> seed;
            for(int j = 0; j < feature_matrix.size(); j++){
                seed.push_back(feature_matrix[j][i]);
            }
            seed_matrix.push_back(seed);
        }
        std::vector<int> base_nodes;
        std::vector<std::vector<double>> base_matrix;
        size_t base_size = select_base(seed_matrix, base_matrix, base_nodes, param.base_ratio);
        MSG(base_size);

        std::vector<std::vector<double>> base_result;
        double avg_tht = 0;     // base theta
        double avg_res = 0;     // reuse residue
        int re_feat_num = 0;    // number of reused features

        // Calculate base PPR
        for(size_t i = 0; i < base_size; i++){
            SpeedPPR::WHOLE_GRAPH_STRUCTURE<double> whole_graph_structure(V_num);
            std::vector<double> seed(V_num, 0.0);
            for(size_t j = 0; j < Vt_num; j++){
                seed[Vt_nodes[j]] = base_matrix[i][j];
            }
            double time_start = getCurrentTime();
            speedPPR.compute_approximate_page_rank_3(whole_graph_structure, seed, param.epsilon, param.alpha,
                                                     1.0 / V_num, walkCache, param.gamma);
            total_time += getCurrentTime() - time_start;
            base_result.push_back(whole_graph_structure.means);
        }
        printf("Time Used on Base %.6f\n", total_time);

        // Calculate residue PPR
        for (size_t i = 0; i < feat_size; i++) {
            SpeedPPR::WHOLE_GRAPH_STRUCTURE<double> whole_graph_structure(V_num);
            bool is_base = false;
            for (size_t idx = 0; idx < base_size; idx++) {
                if (base_nodes[idx] == i){
                    // printf("ID: %4d  is base\n", i);
                    is_base = true;
                    for (size_t j = 0; j < V_num; j++)
                        whole_graph_structure.means[j] = base_result[idx][j];
                    break;
                }
            }
            if (!is_base){
                std::vector<double> raw_seed = seed_matrix[i];
                std::vector<double> base_weight = reuse_weight(raw_seed, base_matrix);

                double theta_sum = vector_L1(base_weight);
                avg_tht += theta_sum;
                avg_res += vector_L1(raw_seed);
                // printf("ID: %4d, theta_sum: %.6f, residue_sum: %.6f\n", i, theta_sum, vector_L1(raw_seed));
                // Ignore less relevant features
                // if (theta_sum < 1.6) continue;
                re_feat_num++;

                std::vector<double> seed(V_num, 0.0);
                for (size_t j = 0; j < Vt_num; j++) {
                    seed[Vt_nodes[j]] = raw_seed[j];
                }

                double time_start = getCurrentTime();
                speedPPR.compute_approximate_page_rank_3(whole_graph_structure, seed, param.epsilon, param.alpha,
                                                         1.0 / V_num, walkCache, 2 - theta_sum * param.gamma);
                total_time += getCurrentTime() - time_start;

                for (size_t idx = 0; idx < base_size; idx++){
                    if (base_weight[idx] != 0) {
                        for (size_t j = 0; j < V_num; j++)
                            whole_graph_structure.means[j] += base_result[idx][j] * base_weight[idx];
                    }
                }
            }

            if (param.output_estimations) {
                // Save [F/split_num, n] array of all nodes to out_matrix
                size_t idxf = i % spt_size;   // index of feature in split
                for (size_t j = 0; j < V_num; j++) {
                    out_matrix[idxf*V_num+j] = whole_graph_structure.means[j];
                }

                size_t spt = i / spt_size;    // index of split
                if (idxf+1 == spt_size || i+1 == feat_size) {
                    std::stringstream res_file;
                    if (param.split_num == 1) {
                        res_file << param.estimation_folder << "/score_" << param.alpha << '_' << param.epsilon << ".npy";
                    } else {
                        res_file << param.estimation_folder << "/score_" << param.alpha << '_' << param.epsilon << "_" << spt << ".npy";
                    }
                    output_feature(out_matrix, res_file.str(), idxf+1, Vt_num);
                }
            }
        }

        avg_tht /= re_feat_num;
        avg_res /= re_feat_num;
        MSG(avg_tht);
        MSG(avg_res);
        MSG(re_feat_num);
        printf("Mem: %ld MB\n", get_proc_memory()/1000);
        printf("Total Time: %.6f, Average: %.12f\n", total_time, total_time / feat_size);
    }
    printf("%s\n", std::string(80, '-').c_str());
    return 0;
}
