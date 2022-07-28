// Ref: https://github.com/wuhao-wu-jiang/Personalized-PageRank
// article{WGWZ21,
//     title={Unifying the Global and Local Approaches: An Efficient Power Iteration with Forward Push},
//     author={Wu, Hao and Gan, Junhao and Wei, Zhewei and Zhang, Rui},
//     journal={arXiv preprint arXiv:2101.03652}}
#include <random>
#include "HelperFunctions.h"
#include "Graph.h"
#include "BatchRandomWalk.h"
#include "SpeedPPR.h"
#include "CleanGraph.h"
#include <unistd.h>

int main(int argc, char **argv) {
    SFMT64::initialize();
    param = parseArgs(argc, argv);
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
        unsigned int node_num = load_query(Vt_nodes, param.query_file);
        load_feature(Vt_nodes, feature_matrix, param.feature_file, param.split_num);
        class SpeedPPR speedPPR(graph);
        double total_time = 0;
        double time_start;
        double time_end;
        graph.reset_set_dummy_neighbor();
        WalkCache walkCache(graph);
        unsigned long feat_num = feature_matrix[0].size(); // feature size
        unsigned long out_size = feat_num * node_num;
        std::vector<float> out_matrix(out_size);
        printf("Result size: %ld \n", out_matrix.size());

        SpeedPPR::WHOLE_GRAPH_STRUCTURE<double> whole_graph_structure(graph.getNumOfVertices());
        for (int i = 0; i < feat_num; i++) {
            // printf("Query ID: %d ... \n", i);
            std::vector<double> seed(graph.getNumOfVertices(), 0.0);
            for(int j = 0; j < Vt_nodes.size(); j++){
                seed[Vt_nodes[j]] = feature_matrix[j][i];
            }

            graph.fill_dead_end_neighbor_with_id();
            time_start = getCurrentTime();
            speedPPR.compute_approximate_page_rank_3(whole_graph_structure, seed, param.epsilon, param.alpha,
                                                     1.0 / graph.getNumOfVertices(), walkCache);
            time_end = getCurrentTime();
            total_time += time_end - time_start;

            if (param.output_estimations) {
                std::stringstream res_file;
                res_file << param.estimation_folder << "/score_" << param.alpha << '_' << param.epsilon;
                output_feature(whole_graph_structure.means, out_matrix,
                               res_file.str(), param.split_num,
                               i, feat_num, node_num);
            }
        }

        printf("Mem: %ld MB\n", get_proc_memory()/1000);
        printf("Total Time: %.6f, Average: %.12f\n", total_time, total_time / feat_num);
    } else if (param.algorithm == "featreuse"){
        std::vector<VertexIdType> Vt_nodes;
        std::vector<std::vector<float>> feature_matrix;
        unsigned int node_num = load_query(Vt_nodes, param.query_file);
        load_feature(Vt_nodes, feature_matrix, param.feature_file, param.split_num);
        class SpeedPPR speedPPR(graph);
        double total_time = 0;
        double time_start;
        double time_end;
        graph.reset_set_dummy_neighbor();
        WalkCache walkCache(graph);
        unsigned long feat_num = feature_matrix[0].size(); // feature size
        unsigned long out_size = feat_num * node_num;
        std::vector<float> out_matrix(out_size);
        printf("Result size: %ld \n", out_matrix.size());

        // Select base
        std::vector<std::vector<double>> seed_matrix;
        for (int i = 0; i < feature_matrix[0].size(); i++) {
            std::vector<double> seed;
            for(int j = 0; j < feature_matrix.size(); j++){
                seed.push_back(feature_matrix[j][i]);
            }
            if(add_up_vector(seed) != 0)
                seed_matrix.push_back(seed);
        }
        std::vector<int> base_vex;
        std::vector<std::vector<double>> base_matrix;
        get_base_with_norm(seed_matrix, base_matrix, base_vex, param.base_ratio);

        SpeedPPR::WHOLE_GRAPH_STRUCTURE<double> whole_graph_structure(graph.getNumOfVertices());
        std::vector<std::vector<double>> base_result;
        graph.fill_dead_end_neighbor_with_id();
        double avg_tht = 0; // base theta
        double avg_res = 0; // reuse residue
        int re_feat_num = 0;

        // Calculate base PPR
        for(int i = 0; i < base_matrix.size(); i++){
            std::vector<double> seed(graph.getNumOfVertices(), 0.0);
            for(int j = 0; j < feature_matrix.size(); j++){
                seed[Vt_nodes[j]] = base_matrix[i][j];
            }
            time_start = getCurrentTime();
            speedPPR.compute_approximate_page_rank_3(whole_graph_structure, seed, param.epsilon, param.alpha,
                                                     1.0 / graph.getNumOfVertices(), walkCache, param.gamma);
            time_end = getCurrentTime();
            total_time += time_end - time_start;
            base_result.push_back(whole_graph_structure.means);
        }
        printf("Time Used on Base %.6f\n", total_time);

        // Calculate residue PPR
        for (int i = 0; i < seed_matrix.size(); i++) {
            // printf("Query ID: %d\n", i);
            bool is_base = false;
            int base_idx;
            for (base_idx = 0; base_idx < base_vex.size(); base_idx++){
                if (base_vex[base_idx] == i){
                    is_base = true;
                    // printf("ID: %4d  is base\n", i);
                    break;
                }
            }
            if (!is_base){
                std::vector<double> raw_seed = seed_matrix[i];
                std::vector<double> base_weight = feature_reuse(raw_seed, base_matrix);

                double theta_sum = 0;
                for(double t : base_weight)   theta_sum+=abs(t);
                avg_tht += theta_sum;
                double rsum = 0;
                for(double t : raw_seed)   rsum+=abs(t);
                avg_res += rsum;
                // printf("ID: %4d, theta_sum: %.6f, residue_sum: %.6f\n", i, theta_sum, rsum);
                // Ignore less relevant features
                // if (theta_sum < 1.6) continue;
                re_feat_num++;

                std::vector<double> seed(graph.getNumOfVertices(), 0.0);
                for (int j = 0; j < feature_matrix.size(); j++){
                    seed[Vt_nodes[j]] = raw_seed[j];
                }

                graph.fill_dead_end_neighbor_with_id();
                time_start = getCurrentTime();
                speedPPR.compute_approximate_page_rank_3(whole_graph_structure, seed, param.epsilon, param.alpha,
                                                         1.0 / graph.getNumOfVertices(), walkCache, 2 - theta_sum * param.gamma);
                time_end = getCurrentTime();
                total_time += time_end - time_start;

                for (int j = 0; j < base_weight.size(); j++){
                    if (base_weight[j] != 0){
                        for (int k = 0; k < whole_graph_structure.means.size(); k++)
                            whole_graph_structure.means[k] += base_result[j][k] * base_weight[j];
                    }
                }
            }
            if (param.output_estimations) {
                std::stringstream res_file;
                res_file << param.estimation_folder << "/score_" << param.alpha << '_' << param.epsilon;
                output_feature(whole_graph_structure.means, out_matrix,
                               res_file.str(), param.split_num,
                               i, feat_num, node_num);
            }
        }

        avg_tht /= re_feat_num;
        avg_res /= re_feat_num;
        MSG(avg_tht);
        MSG(avg_res);
        MSG(re_feat_num);
        printf("Mem: %ld MB\n", get_proc_memory()/1000);
        printf("Total Time: %.6f, Average: %.12f\n", total_time, total_time / feat_num);
    }
    printf("%s\n", std::string(80, '-').c_str());
    return 0;
}
