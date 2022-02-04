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
//            graph.fill_dead_end_neighbor_with_id(graph.get_dummy_id());
            // must be called before generating index
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

    std::unordered_set<std::string> iterated_algorithms{"PowItr", "PowForPush", "FwdPush"};

    if (param.algorithm == "GROUND_TRUTH") {
        QueryLoader queryLoader(param.query_file, param.query_size);
        // this command must be called for this section.
        graph.set_dummy_out_degree_zero();
        const double l1_error = 1e-9;
        MSG(l1_error)
        IteratedMethods iterated_methods(graph, l1_error);
        const double time_start = getCurrentTime();
        iterated_methods.multi_thread_ground_truth_iteration(queryLoader.source_vertices());
        const double time_end = getCurrentTime();
        std::cout << "Time Used Total: " << time_end - time_start << std::endl;
    } else if (iterated_algorithms.count(param.algorithm)) {
        QueryLoader queryLoader(param.query_file, param.query_size);
        const double l1_error = param.specified_l1_error ? param.l1_error : 1.0 / std::max(1u, graph.getNumOfEdges());
        IteratedMethods iterated_methods(graph, l1_error);
        double total_time = 0;
        std::vector<double> pi;
        std::vector<double> residuals;
        FwdPushStructure fwdPushStructure(graph.getNumOfVertices());
        std::vector<std::pair<VertexIdType, double>> sid_time;
        for (const VertexIdType &sid : queryLoader.source_vertices()) {
            printf("%s\n", std::string(110, '=').c_str());
            printf("Vertex ID: %d\n", sid);
            if (param.algorithm == "PowItr") {
                const double time_start = getCurrentTime();
                graph.fill_dead_end_neighbor_with_id(sid);
                iterated_methods.naive_power_iteration(sid, pi, residuals);
                const double time_end = getCurrentTime();
                sid_time.emplace_back(sid, time_end - time_start);
            } else if (param.algorithm == "FwdPush") {
                const double time_start = getCurrentTime();
                graph.fill_dead_end_neighbor_with_id(sid);
                iterated_methods.forward(sid, pi, residuals);
                const double time_end = getCurrentTime();
                sid_time.emplace_back(sid, time_end - time_start);
            } else if (param.algorithm == "PowForPush") {
                const double time_start = getCurrentTime();
                graph.change_in_neighbors_adj(sid, graph.get_dummy_id());
                iterated_methods.forward_iteration(sid, pi, residuals, fwdPushStructure);
                graph.restore_neighbors_adj(sid);
                const double time_end = getCurrentTime();
                sid_time.emplace_back(sid, time_end - time_start);
            }
            printf("Time Used %.12f\n", sid_time.back().second);
            total_time += sid_time.back().second;
            if (param.output_estimations) {
                save_answer(pi, graph.getNumOfVertices(),
                            param.estimation_folder + "/" + param.algorithm + "_" + std::to_string(sid) + ".txt");
            }
            printf("%s\n", std::string(50, '-').c_str());
        }
        auto num_queries = queryLoader.source_vertices().size();
        printf("Average Time Used %.12f\n", total_time / num_queries);
        graph.fill_dead_end_neighbor_with_id(graph.get_dummy_id());
        {
            // report median time
            std::sort(sid_time.begin(), sid_time.end(), [](const auto &left, const auto &right) {
                return left.second < right.second;
            });
            auto median_index = num_queries / 2;
            printf("Sid with Median Query Time: %u\t Time: %.12f\n", sid_time[median_index].first,
                   sid_time[median_index].second);
        }
    } else if(param.algorithm == "FEATPUSH"){
        std::vector<VertexIdType> Vt_nodes;
        std::vector<std::vector<float>> feature_matrix;
        load_Vt_and_features_l(Vt_nodes, feature_matrix, param.query_file, param.feature_file, param.query_size);
        class SpeedPPR speedPPR(graph);
        double total_time = 0;
        unsigned int num_queries = feature_matrix[0].size();
        graph.reset_set_dummy_neighbor();
        WalkCache walkCache(graph);
        if (param.with_idx) {
            walkCache.load(param.index_file);
        }
        std::vector<float> res_matrix(num_queries*param.query_size);
        const uint32_t max_num_vertices = 1u;

        std::vector<std::vector<double>> seed_matrix;
        MSG(Vt_nodes.size());
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
        get_base_with_L1(seed_matrix, base_matrix, base_vex);

        if (graph.getNumOfVertices() > max_num_vertices) {
            SpeedPPR::WHOLE_GRAPH_STRUCTURE<double> whole_graph_structure(graph.getNumOfVertices());
            double time_start;
            double time_end;
            double avg_res = 0;
            int valid_feat_num = 0;
            for (int i = 0; i < num_queries; i++) {
                std::vector<double> raw_seed = seed_matrix[i];
                std::vector<double> base_weight = feature_reuse(raw_seed, base_matrix);
                double rsum = 0;
                for(double r : raw_seed)   rsum+=abs(r);
                if(rsum > 0.5) continue;
                valid_feat_num++;

                std::vector<double> seed(graph.getNumOfVertices(), 0.0);
                for(int j = 0; j < feature_matrix.size(); j++){
                    seed[Vt_nodes[j]] = feature_matrix[j][i];
                }

                time_start = getCurrentTime();
                speedPPR.compute_approximate_page_rank_3(whole_graph_structure, seed, param.epsilon, param.alpha,
                                                         1.0 / graph.getNumOfVertices(), walkCache);
                time_end = getCurrentTime();
                total_time += time_end - time_start;
            }
            MSG(valid_feat_num);

        }
        struct rusage r_usage;
        getrusage(RUSAGE_SELF,&r_usage);
        std::cout<<"Total Memory: "<<r_usage.ru_maxrss/1000<<" MB"<<std::endl;
        printf("Total Time: %.12f, Average: %.12f\n", total_time, total_time / num_queries);
    } else if(param.algorithm == "FEATPUSH_REUSE"){
        std::vector<VertexIdType> Vt_nodes;
        std::vector<std::vector<float>> feature_matrix;
        std::vector<std::vector<double>> feature_matrix_new;
        // load_Vt_and_features_np(Vt_nodes, feature_matrix, param.query_file, param.feature_file);
        load_Vt_and_features_l(Vt_nodes, feature_matrix, param.query_file, param.feature_file, param.query_size);
        class SpeedPPR speedPPR(graph);
        double total_time = 0;
        unsigned int num_queries = feature_matrix[0].size();
        graph.reset_set_dummy_neighbor();
        WalkCache walkCache(graph);
        if (param.with_idx) {
            walkCache.load(param.index_file);
        }
        std::vector<float> res_matrix(num_queries*param.query_size);
        const uint32_t max_num_vertices = 1u;

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
        get_base_with_L1(seed_matrix, base_matrix, base_vex);

        if (graph.getNumOfVertices() > max_num_vertices) {
            SpeedPPR::WHOLE_GRAPH_STRUCTURE<double> whole_graph_structure(graph.getNumOfVertices());
            std::vector<std::vector<double>> base_result;
            double time_start;
            double time_end;
            double avg_res = 0;
            int valid_feat_num = 0;
            double gamma = 0.2;
            graph.fill_dead_end_neighbor_with_id();
            for(int i = 0; i < base_matrix.size(); i++){
                std::vector<double> seed(graph.getNumOfVertices(), 0.0);
                for(int j = 0; j < feature_matrix.size(); j++){
                    seed[Vt_nodes[j]] = base_matrix[i][j];
                }
                time_start = getCurrentTime();
                speedPPR.compute_approximate_page_rank_3(whole_graph_structure, seed, param.epsilon, param.alpha,
                                                         1.0 / graph.getNumOfVertices(), walkCache, gamma);
                time_end = getCurrentTime();
                total_time += time_end - time_start;
                base_result.push_back(whole_graph_structure.means);
            }
            printf("Time Used on Base %.12f\n", total_time);
            for (int i = 0; i < seed_matrix.size(); i++) {
                // printf("%s\n", std::string(50, '-').c_str());
                // printf("Vertex ID: %d\n", i);
                bool is_base = false;
                int base_idx;
                for(base_idx = 0; base_idx < base_vex.size(); base_idx++){
                    if(base_vex[base_idx] == i){
                        is_base = true;
                        break;
                    }
                }
                if(!is_base){
                    std::vector<double> raw_seed = seed_matrix[i];
                    std::vector<double> base_weight = feature_reuse(raw_seed, base_matrix);

                    double rsum = 0;
                    for(double r : raw_seed)   rsum+=abs(r);
                    MSG(rsum);
                    if(rsum > 0.5) continue;
                    avg_res += rsum;
                    valid_feat_num++;

                    std::vector<double> seed(graph.getNumOfVertices(), 0.0);
                    for(int j = 0; j < feature_matrix.size(); j++){
                        seed[Vt_nodes[j]] = raw_seed[j];
                        //seed[Vt_nodes[j]] = feature_matrix[j][i];
                        //seed[Vt_nodes[j]] = feature_matrix_new[i][j];
                    }

                    graph.fill_dead_end_neighbor_with_id();
                    time_start = getCurrentTime();
                    speedPPR.compute_approximate_page_rank_3(whole_graph_structure, seed, param.epsilon, param.alpha,
                                                                1.0 / graph.getNumOfVertices(), walkCache, 1 - (1 - rsum) * gamma);
                    time_end = getCurrentTime();
                    total_time += time_end - time_start;

                    for(int j = 0; j < base_weight.size(); j++){
                        if(base_weight[j] > 0){
                            for(int k = 0; k < whole_graph_structure.means.size(); k++)
                                whole_graph_structure.means[k] += base_result[j][k] * base_weight[j];
                        }
                    }
                }
            }
            avg_res /= valid_feat_num;
            MSG(avg_res);
            MSG(valid_feat_num);
        }
        printf("Mem: %.12f", get_proc_memory());
        printf("Total Time: %.12f, Average: %.12f\n", total_time, total_time / num_queries);
    }
    return 0;
}
