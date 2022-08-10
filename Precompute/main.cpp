#include "HelperFunctions.h"
#include "Graph.h"
#include "BatchRandomWalk.h"
#include "SpeedPPR.h"
#include "CleanGraph.h"
#include "FeatureOp.h"
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
        // Process graph
        graph.reset_set_dummy_neighbor();
        graph.fill_dead_end_neighbor_with_id();
        VertexIdType V_num = graph.getNumOfVertices();
        class SpeedPPR speedPPR(graph);
        WalkCache walkCache(graph);
        double total_time = 0;
        double total_time2 = 0;
        double total_time3 = 0;

        // Process feature
        std::vector<VertexIdType> Vt_nodes; // list of queried nodes
        MyMatrix feature_matrix;
        VertexIdType Vt_num = load_query(Vt_nodes, param.query_file);
        VertexIdType feat_size = load_feature(Vt_nodes, feature_matrix, param.feature_file, param.split_num); // feature size
        VertexIdType spt_size = (feat_size + param.split_num - 1) / param.split_num;    // feature size per split (ceiling)
        VertexIdType out_size = spt_size * V_num; // length of output matrix
        std::vector<PageRankScoreType> out_matrix(out_size);
        printf("Result size: %ld \n", out_matrix.size());

        for (VertexIdType spt_left = 0; spt_left < feat_size; spt_left += spt_size) {
            VertexIdType spt_right = std::min(feat_size, spt_left + spt_size);
            for (VertexIdType i = spt_left; i < spt_right; i++) {
                // printf("ID: %4ld \n", i);
                VertexIdType idxf = i % spt_size;   // index of feature in split
                double time_start = getCurrentTime();
                SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> graph_structure(V_num);
                std::vector<PageRankScoreType> seed;
                if (Vt_num == V_num) {
                    seed.swap(feature_matrix[i]);
                } else {
                    seed.resize(V_num, 0.0);
                    for (VertexIdType j = 0; j < Vt_num; j++) {
                        seed[Vt_nodes[j]] = feature_matrix[i][j];
                    }
                }
                total_time2 += getCurrentTime() - time_start;

                time_start = getCurrentTime();
                speedPPR.compute_approximate_page_rank_3(graph_structure, seed, param.epsilon, param.alpha,
                                                         1.0 / V_num, walkCache);
                total_time += getCurrentTime() - time_start;

                // Save embedding vector of feature i on all nodes to out_matrix
                time_start = getCurrentTime();
                std::swap_ranges(graph_structure.means.begin(), graph_structure.means.end(),
                                 out_matrix.begin() + idxf*V_num);
                total_time3 += getCurrentTime() - time_start;
            }

            if (param.output_estimations) {
                VertexIdType spt = spt_left / spt_size;    // index of split
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
        printf("Total Time2: %.6f, Average: %.12f\n", total_time2, total_time2 / feat_size);
        printf("Total Time3: %.6f, Average: %.12f\n", total_time3, total_time3 / feat_size);
    } else if (param.algorithm == "featreuse"){
        // Process graph
        graph.reset_set_dummy_neighbor();
        graph.fill_dead_end_neighbor_with_id();
        VertexIdType V_num = graph.getNumOfVertices();
        class SpeedPPR speedPPR(graph);
        WalkCache walkCache(graph);
        double total_time = 0;
        double total_time2 = 0;
        double total_time3 = 0;

        // Process feature
        std::vector<VertexIdType> Vt_nodes; // list of queried nodes
        MyMatrix feature_matrix;
        VertexIdType Vt_num = load_query(Vt_nodes, param.query_file);
        VertexIdType feat_size = load_feature(Vt_nodes, feature_matrix, param.feature_file, param.split_num); // feature size
        VertexIdType spt_size = (feat_size + param.split_num - 1) / param.split_num;    // feature size per split (ceiling)
        VertexIdType out_size = spt_size * V_num; // length of output matrix
        std::vector<PageRankScoreType> out_matrix(out_size);
        printf("Result size: %ld \n", out_matrix.size());

        // Select base
        MyMatrix base_matrix;
        VertexIdType base_size = feat_size * param.base_ratio;
        base_size = std::max(3u, base_size);
        std::vector<VertexIdType> base_nodes = select_base(feature_matrix, base_matrix, base_size);
        MSG(base_size);

        MyMatrix base_result(base_size, V_num);
        double avg_tht = 0;     // base theta
        double avg_res = 0;     // reuse residue
        VertexIdType re_feat_num = 0;    // number of reused features

        // Calculate base PPR
        for(VertexIdType i = 0; i < base_size; i++){
            double time_start = getCurrentTime();
            SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> graph_structure(V_num);
            std::vector<PageRankScoreType> seed(V_num, 0.0);
            for(VertexIdType j = 0; j < Vt_num; j++){
                seed[Vt_nodes[j]] = base_matrix[i][j];
            }
            total_time2 += getCurrentTime() - time_start;

            time_start = getCurrentTime();
            speedPPR.compute_approximate_page_rank_3(graph_structure, seed, param.epsilon, param.alpha,
                                                     1.0 / V_num, walkCache, param.gamma);
            total_time += getCurrentTime() - time_start;

            time_start = getCurrentTime();
            base_result.set_row(i, graph_structure.means);
            total_time3 += getCurrentTime() - time_start;
        }
        printf("Time Used on Base %.6f\n", total_time);

        // Calculate residue PPR
        for (VertexIdType spt_left = 0; spt_left < feat_size; spt_left += spt_size) {
            VertexIdType spt_right = std::min(feat_size, spt_left + spt_size);
            for (VertexIdType i = spt_left; i < spt_right; i++) {
                VertexIdType idxf = i % spt_size;   // index of feature in split
                SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> graph_structure(V_num);
                bool is_base = false;
                for (VertexIdType idx = 0; idx < base_size; idx++) {
                    if (base_nodes[idx] == i) {
                        // printf("ID: %4ld  is base\n", i);
                        is_base = true;
                        double time_start = getCurrentTime();
                        std::copy(base_result[idx].begin(), base_result[idx].end(), out_matrix.begin() + idxf*V_num);
                        total_time3 += getCurrentTime() - time_start;
                        break;
                    }
                }
                if (!is_base) {
                    std::vector<PageRankScoreType> raw_seed;
                    raw_seed.swap(feature_matrix[i]);
                    std::vector<PageRankScoreType> base_weight = reuse_weight(raw_seed, base_matrix);
                    double theta_sum = vector_L1(base_weight);
                    avg_tht += theta_sum;
                    avg_res += vector_L1(raw_seed);
                    // printf("ID: %4ld, theta_sum: %.6f, residue_sum: %.6f\n", i, theta_sum, vector_L1(raw_seed));
                    // Ignore less relevant features
                    // if (theta_sum < 1.6) continue;
                    re_feat_num++;

                    double time_start = getCurrentTime();
                    std::vector<PageRankScoreType> seed;
                    if (Vt_num == V_num) {
                        seed.swap(raw_seed);
                    } else {
                        seed.resize(V_num, 0.0);
                        for (VertexIdType j = 0; j < Vt_num; j++) {
                            seed[Vt_nodes[j]] = raw_seed[j];
                        }
                    }
                    total_time2 += getCurrentTime() - time_start;

                    time_start = getCurrentTime();
                    speedPPR.compute_approximate_page_rank_3(graph_structure, seed, param.epsilon, param.alpha,
                                                            1.0 / V_num, walkCache, 2 - theta_sum * param.gamma);
                    total_time += getCurrentTime() - time_start;

                    for (VertexIdType idx = 0; idx < base_size; idx++){
                        if (base_weight[idx] != 0) {
                            for (VertexIdType j = 0; j < V_num; j++)
                                graph_structure.means[j] += base_result[idx][j] * base_weight[idx];
                        }
                    }
                    // Save embedding vector of feature i on all nodes to out_matrix
                    time_start = getCurrentTime();
                    std::swap_ranges(graph_structure.means.begin(), graph_structure.means.end(),
                                     out_matrix.begin() + idxf*V_num);
                    total_time3 += getCurrentTime() - time_start;
                }
            }

            if (param.output_estimations) {
                VertexIdType spt = spt_left / spt_size;    // index of split
                std::stringstream res_file;
                if (param.split_num == 1) {
                    res_file << param.estimation_folder << "/score_" << param.alpha << '_' << param.epsilon << ".npy";
                } else {
                    res_file << param.estimation_folder << "/score_" << param.alpha << '_' << param.epsilon << "_" << spt << ".npy";
                }
                output_feature(out_matrix, res_file.str(), spt_right - spt_left, V_num);
            }
        }

        avg_tht /= re_feat_num;
        avg_res /= re_feat_num;
        MSG(avg_tht);
        MSG(avg_res);
        MSG(re_feat_num);
        printf("Mem: %ld MB\n", get_proc_memory()/1000);
        printf("Total Time: %.6f, Average: %.12f\n", total_time, total_time / feat_size);
        printf("Total Time2: %.6f, Average: %.12f\n", total_time2, total_time2 / feat_size);
        printf("Total Time3: %.6f, Average: %.12f\n", total_time3, total_time3 / feat_size);
    }
    printf("%s\n", std::string(80, '-').c_str());
    return 0;
}
