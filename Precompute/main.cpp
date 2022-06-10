// Ref: https://github.com/wuhao-wu-jiang/Personalized-PageRank
// article{WGWZ21,
//     title={Unifying the Global and Local Approaches: An Efficient Power Iteration with Forward Push},
//     author={Wu, Hao and Gan, Junhao and Wei, Zhewei and Zhang, Rui},
//     journal={arXiv preprint arXiv:2101.03652}}
#include <random>
#include "HelperFunctions.h"
#include "Graph.h"
#include "IteratedMethods.h"
#include "QueryLoader.h"
#include "BatchRandomWalk.h"
#include "SpeedPPR.h"
#include "CleanGraph.h"
#include "QueryGenerator.h"
#include <unistd.h>

int main(int argc, char **argv) {
//
    SFMT64::initialize();
    param = parseArgs(argc, argv);
    Graph graph;
    graph.set_alpha(param.alpha);

    if (param.algorithm == "CLEAN_GRAPH") {
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

    if (param.algorithm == "GEN_QUERY") {
        QueryGenerator::generate(graph, param.query_file);
        return 0;
    }

    if (param.algorithm == "BUILD_INDEX") {
        if (!param.index_file.empty()) {
            graph.set_dummy_neighbor(graph.get_dummy_id());
            WalkCache walkCache(graph);
            const double time_start = getCurrentTime();
            walkCache.generate();
            const double time_end = getCurrentTime();
            printf("Time Used %.12f\n", time_end - time_start);
            walkCache.save(param.index_file);
            graph.reset_set_dummy_neighbor();
            return 0;
        } else {
            printf("Error in" __FILE__ " LINE %d." "File Not Exists\n", __LINE__);
            return 0;
        }
    }

    if (param.query_size == 0) {
        printf("Error. Query Size Not Specified.\n");
        return 0;
    }

    if (param.algorithm == "FEATPUSH"){
        std::vector<VertexIdType> Vt_nodes;
        std::vector<std::vector<float>> feature_matrix;
        load_features(Vt_nodes, feature_matrix, param.query_file, param.feature_file);
        class SpeedPPR speedPPR(graph);
        double total_time = 0;
        double time_start;
        double time_end;
        graph.reset_set_dummy_neighbor();
        WalkCache walkCache(graph);
        if (param.with_idx) {
            walkCache.load(param.index_file);
        }
        unsigned long num_queries = feature_matrix[0].size(); // feature size
        unsigned long res_size = num_queries * param.query_size;
        MSG(num_queries);
        MSG(res_size);
        std::vector<float> res_matrix(res_size);
        printf("Res size: %ld \n", res_matrix.size());

        SpeedPPR::WHOLE_GRAPH_STRUCTURE<double> whole_graph_structure(graph.getNumOfVertices());
        for (int i = 0; i < num_queries; i++) {
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
                std::stringstream ss;
                ss << param.epsilon;
                output_matrix(whole_graph_structure.means, param.query_size,
                                param.estimation_folder + "/" + "score" + '_' + ss.str() + ".npy",
                                res_matrix, i, num_queries);
            }
        }

        printf("Mem: %ld\n", get_proc_memory());
        printf("Total Time: %.12f, Average: %.12f\n", total_time, total_time / num_queries);
    } else if (param.algorithm == "FEATREUSE"){
        std::vector<VertexIdType> Vt_nodes;
        std::vector<std::vector<float>> feature_matrix;
        load_features(Vt_nodes, feature_matrix, param.query_file, param.feature_file);
        class SpeedPPR speedPPR(graph);
        double total_time = 0;
        double time_start;
        double time_end;
        graph.reset_set_dummy_neighbor();
        WalkCache walkCache(graph);
        if (param.with_idx) {
            walkCache.load(param.index_file);
        }
        unsigned long num_queries = feature_matrix[0].size(); // feature size
        unsigned long res_size = num_queries * param.query_size;
        MSG(num_queries);
        MSG(res_size);
        std::vector<float> res_matrix(res_size);
        printf("Res size: %ld \n", res_matrix.size());

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
        get_base_with_L1(seed_matrix, base_matrix, base_vex, param.base_ratio);

        SpeedPPR::WHOLE_GRAPH_STRUCTURE<double> whole_graph_structure(graph.getNumOfVertices());
        std::vector<std::vector<double>> base_result;
        graph.fill_dead_end_neighbor_with_id();
        double avg_tht = 0;
        double avg_res = 0;
        int valid_feat_num = 0;

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
        printf("Time Used on Base %.12f\n", total_time);

        for (int i = 0; i < seed_matrix.size(); i++) {
            // printf("Query ID: %d\n", i);
            bool is_base = false;
            int base_idx;
            for (base_idx = 0; base_idx < base_vex.size(); base_idx++){
                if (base_vex[base_idx] == i){
                    is_base = true;
                    printf("ID: %4d  is base\n", i);
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
                printf("ID: %4d, theta_sum: %.6f, residue_sum: %.6f\n", i, theta_sum, rsum);
                // Ignore less relevant features
                // if (theta_sum < 1.6) continue;
                valid_feat_num++;

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
                std::stringstream ss;
                ss << param.epsilon;
                output_matrix(whole_graph_structure.means, param.query_size,
                                param.estimation_folder + "/" + "score" + '_' + ss.str() + ".npy",
                                res_matrix, i, num_queries);
            }
        }

        avg_tht /= valid_feat_num;
        avg_res /= valid_feat_num;
        MSG(avg_tht);
        MSG(avg_res);
        MSG(valid_feat_num);
        printf("Mem: %ld\n", get_proc_memory());
        printf("Total Time: %.12f, Average: %.12f\n", total_time, total_time / num_queries);
    }
    return 0;
}
