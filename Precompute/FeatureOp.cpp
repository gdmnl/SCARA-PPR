#include <thread>
#include "Graph.h"
#include "BatchRandomWalk.h"
#include "SpeedPPR.h"
#include "HelperFunctions.h"
#include "FeatureOp.h"

// Wrapper class
class Base {

protected:
    Param &param;
    Graph &graph;
    class SpeedPPR ppr;
    WalkCache walkCache;
    std::vector<VertexIdType> Vt_nodes; // list of queried nodes
    MyMatrix feature_matrix;

    PageRankScoreType epsilon;
    PageRankScoreType alpha;
    PageRankScoreType lower_threshold;

public:
    VertexIdType V_num;                 // number of vertices
    VertexIdType Vt_num;                // number of queried nodes
    VertexIdType feat_size;             // size of feature
    VertexIdType spt_size;              // size of feature per split (ceiling)
    // statistics
    double total_time = 0;
    double total_time2 = 0;
    double total_time3 = 0;

    std::vector<PageRankScoreType> out_matrix;

public:

    Base(Graph &_graph, Param &_param) :
            V_num(_graph.getNumOfVertices()),
            ppr(graph),
            walkCache(graph),
            epsilon(_param.epsilon),
            alpha(_param.alpha),
            lower_threshold(1.0 / _graph.getNumOfVertices()),
            graph(_graph),
            param(_param) {
        Vt_num = load_query(Vt_nodes, param.query_file);
        feat_size = load_feature(Vt_nodes, feature_matrix, param.feature_file, param.split_num);
        spt_size = (feat_size + param.split_num - 1) / param.split_num;
        out_matrix.resize(spt_size * V_num);   // spt_size rows, V_num columns
        printf("Result size: %ld \n", out_matrix.size());
        graph.reset_set_dummy_neighbor();
        graph.fill_dead_end_neighbor_with_id();
    }

    void push_one(const VertexIdType i) {
        // printf("ID: %4" IDFMT "\n", i);
        double time_start = getCurrentTime();
        VertexIdType idxf = i % spt_size;     // index of feature in split
        SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> graph_structure(V_num);
        std::vector<PageRankScoreType> seed;
        propagate_vector(feature_matrix[i], seed, Vt_nodes, V_num, true);
        total_time2 += getCurrentTime() - time_start;

        time_start = getCurrentTime();
        ppr.compute_approximate_page_rank_3(graph_structure, seed, epsilon, alpha,
                                            lower_threshold, walkCache);
        total_time += getCurrentTime() - time_start;

        // Save embedding vector of feature i on all nodes to out_matrix
        time_start = getCurrentTime();
        std::swap_ranges(graph_structure.means.begin(), graph_structure.means.end(),
                         out_matrix.begin() + idxf*V_num);
        total_time3 += getCurrentTime() - time_start;
    }

    void save_output(const VertexIdType feat_left, const VertexIdType feat_right) {
        if (param.output_estimations) {
            VertexIdType spt = feat_left / spt_size;    // index of split
            std::stringstream res_file;
            if (param.split_num <= 1) {
                res_file << param.estimation_folder << "/score_" << param.alpha << '_' << param.epsilon << ".npy";
            } else {
                res_file << param.estimation_folder << "/score_" << param.alpha << '_' << param.epsilon << "_" << spt << ".npy";
            }
            output_feature(out_matrix, res_file.str(), feat_right - feat_left, V_num);
        }
    }

    void show_statistics() {
        printf("Total Time : %.6f, Average: %.12f\n", total_time, total_time / feat_size);
        printf("Total Time2: %.6f, Average: %.12f\n", total_time2, total_time2 / feat_size);
        printf("Total Time3: %.6f, Average: %.12f\n", total_time3, total_time3 / feat_size);
    }

    void push() {
        for (VertexIdType spt_left = 0; spt_left < feat_size; spt_left += spt_size) {
            VertexIdType spt_right = std::min(feat_size, spt_left + spt_size);
            // TODO: 1. parallel function
            for (VertexIdType i = spt_left; i < spt_right; i++) {
                push_one(i);
            }
            save_output(spt_left, spt_right);
        }
    }

};

class Base_reuse : public Base {

public:
    VertexIdType base_size;
    // statistics
    PageRankScoreType avg_tht = 0;      // base theta
    PageRankScoreType avg_res = 0;      // reuse residue
    VertexIdType re_feat_num = 0;       // number of reused features

protected:
    std::vector<VertexIdType> base_nodes;
    MyMatrix base_matrix;
    MyMatrix base_result;

public:

    Base_reuse(Graph &_graph, Param &_param) :
            Base(_graph, _param),
            base_size(std::max(VertexIdType (3u), VertexIdType (feat_size * param.base_ratio))),
            base_matrix(base_size, Vt_num),
            base_result(base_size, V_num) {
        base_nodes = select_base(feature_matrix, base_matrix, base_size);
        printf("Base size: %ld \n", base_result.size());
    }

    void push_one_base(const VertexIdType idx) {
        // printf("ID: %4" IDFMT "  as base\n", idx);
        double time_start = getCurrentTime();
        SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> graph_structure(V_num);
        std::vector<PageRankScoreType> seed;
        propagate_vector(base_matrix[idx], seed, Vt_nodes, V_num, false);
        total_time2 += getCurrentTime() - time_start;

        time_start = getCurrentTime();
        ppr.compute_approximate_page_rank_3(graph_structure, seed, epsilon, alpha,
                                            lower_threshold, walkCache, param.gamma);
        total_time += getCurrentTime() - time_start;

        time_start = getCurrentTime();
        base_result.set_row(idx, graph_structure.means);
        total_time3 += getCurrentTime() - time_start;
    }

    void push_one_rest(const VertexIdType i) {
        VertexIdType idxf = i % spt_size;   // index of feature in split
        SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> graph_structure(V_num);
        bool is_base = false;
        for (VertexIdType idx = 0; idx < base_size; idx++) {
            if (base_nodes[idx] == i) {
                // printf("ID: %4" IDFMT "  is base\n", i);
                is_base = true;
                double time_start = getCurrentTime();
                std::copy(base_result[idx].begin(), base_result[idx].end(), out_matrix.begin() + idxf*V_num);
                total_time3 += getCurrentTime() - time_start;
                break;
            }
        }
        if (is_base) return;

        std::vector<PageRankScoreType> raw_seed;
        raw_seed.swap(feature_matrix[i]);
        std::vector<PageRankScoreType> base_weight = reuse_weight(raw_seed, base_matrix);
        PageRankScoreType theta_sum = vector_L1(base_weight);
        avg_tht += theta_sum;
        avg_res += vector_L1(raw_seed);
        // printf("ID: %4" IDFMT ", theta_sum: %.6f, residue_sum: %.6f\n", i, theta_sum, vector_L1(raw_seed));
        // Ignore less relevant features
        // if (theta_sum < 1.6) continue;
        re_feat_num++;

        double time_start = getCurrentTime();
        std::vector<PageRankScoreType> seed;
        propagate_vector(raw_seed, seed, Vt_nodes, V_num, true);
        total_time2 += getCurrentTime() - time_start;

        time_start = getCurrentTime();
        ppr.compute_approximate_page_rank_3(graph_structure, seed, epsilon, alpha,
                                            lower_threshold, walkCache, 2 - theta_sum * param.gamma);
        total_time += getCurrentTime() - time_start;

        time_start = getCurrentTime();
        for (VertexIdType idx = 0; idx < base_size; idx++){
            if (base_weight[idx] != 0) {
                for (VertexIdType j = 0; j < V_num; j++)
                    graph_structure.means[j] += base_result[idx][j] * base_weight[idx];
            }
        }
        // Save embedding vector of feature i on all nodes to out_matrix
        std::swap_ranges(graph_structure.means.begin(), graph_structure.means.end(),
                            out_matrix.begin() + idxf*V_num);
        total_time3 += getCurrentTime() - time_start;
    }

    void show_statistics() {
        avg_tht /= re_feat_num;
        avg_res /= re_feat_num;
        MSG(avg_tht);
        MSG(avg_res);
        MSG(re_feat_num);
        Base::show_statistics();
    }

    void push() {
        // Calculate base PPR
        for(VertexIdType i = 0; i < base_size; i++){
            push_one_base(i);
        }
        printf("Time Used on Base %.6f\n", total_time);

        // Calculate rest PPR
        for (VertexIdType spt_left = 0; spt_left < feat_size; spt_left += spt_size) {
            VertexIdType spt_right = std::min(feat_size, spt_left + spt_size);
            for (VertexIdType i = spt_left; i < spt_right; i++) {
                push_one_rest(i);
            }
            save_output(spt_left, spt_right);
        }
    }

};
