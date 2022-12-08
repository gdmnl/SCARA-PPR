#include <thread>
#include "Graph.h"
#include "SpeedPPR.h"
#include "HelperFunctions.h"
#include "FeatureOp.h"
#ifdef ENABLE_RW
#include "BatchRandomWalk.h"
#endif

// Wrapper class
class Base {

protected:
    Param &param;
    Graph &graph;
    class SpeedPPR ppr;
    std::vector<VertexIdType> Vt_nodes; // list of queried nodes
    MyMatrix feature_matrix;
#ifdef ENABLE_RW
    WalkCache walkCache;
#endif

    VertexIdType thread_num;            // number of threads
    PageRankScoreType epsilon;
    PageRankScoreType alpha;
    PageRankScoreType lower_threshold;

public:
    VertexIdType V_num;                 // number of vertices
    VertexIdType Vt_num;                // number of queried nodes
    VertexIdType feat_size;             // size of feature
    VertexIdType spt_size;              // size of feature per split (ceiling)
    VertexIdType thd_size;              // size of feature per thread per split
    // statistics
    double total_time = 0;
    std::vector<double> time_read;
    std::vector<double> time_write;
    std::vector<double> time_push;

    My2DVector out_matrix;

public:

    Base(Graph &_graph, Param &_param) :
            V_num(_graph.getNumOfVertices()),
            ppr(graph),
#ifdef ENABLE_RW
            walkCache(graph),
#endif
            epsilon(_param.epsilon),
            alpha(_param.alpha),
            lower_threshold(1.0 / _graph.getNumOfVertices()),
            graph(_graph),
            param(_param) {
        Vt_num = load_query(Vt_nodes, param.query_file);
        feat_size = load_feature(Vt_nodes, feature_matrix, param.feature_file, param.split_num);
        spt_size = (feat_size + param.split_num - 1) / param.split_num;
        out_matrix.allocate(spt_size, V_num);   // spt_size rows, V_num columns
        printf("Result size: %ld \n", out_matrix.size());
        // Perform cached random walk
#ifdef ENABLE_RW
         if (param.index) {
            graph.set_dummy_neighbor(graph.get_dummy_id());
            walkCache.generate();
            graph.reset_set_dummy_neighbor();
        }
#endif
         graph.fill_dead_end_neighbor_with_id();

        thread_num = std::min(spt_size, (VertexIdType) param.thread_num);
        thd_size = (spt_size + thread_num - 1) / thread_num;
        time_read.resize(thread_num, 0);
        time_write.resize(thread_num, 0);
        time_push.resize(thread_num, 0);
    }

    void push_one(const VertexIdType i, const VertexIdType tid,
            SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> &_graph_structure,
            std::vector<PageRankScoreType> &_seed) {
        // printf("ID: %4" IDFMT "\n", i);
        double time_start = getCurrentTime();
        propagate_vector(feature_matrix[i], _seed, Vt_nodes, V_num, true);
        time_read[tid] += getCurrentTime() - time_start;

        time_start = getCurrentTime();
#ifdef ENABLE_RW
        if (param.index)
            ppr.calc_ppr_cache(_graph_structure, _seed, epsilon, alpha, lower_threshold, walkCache);
        else
            ppr.calc_ppr_walk(_graph_structure, _seed, epsilon, alpha, lower_threshold);
#else
        ppr.calc_ppr_walk(_graph_structure, _seed, epsilon, alpha, lower_threshold);
#endif
        time_push[tid] += getCurrentTime() - time_start;

        // Save embedding vector of feature i on all nodes to out_matrix
        time_start = getCurrentTime();
        std::swap_ranges(_graph_structure.means.begin(), _graph_structure.means.end()-2,
                         out_matrix[i%spt_size].begin());
        time_write[tid] += getCurrentTime() - time_start;
    }

    void push_thread(const VertexIdType feat_left, const VertexIdType feat_right, const VertexIdType tid) {
        // std::cout<<"  Pushing: "<<feat_left<<" "<<feat_right<<std::endl;
        SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> graph_structure(V_num);
        std::vector<PageRankScoreType> seed;
        for (VertexIdType i = feat_left; i < feat_right; i++) {
            push_one(i, tid, graph_structure, seed);
        }
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
            output_feature(out_matrix.get_data(), res_file.str(), feat_right - feat_left, V_num);
        }
    }

    void show_statistics() {
        printf("Mem: %ld MB\n", get_proc_memory()/1000);
        printf("Total Time    : %.6f, Average: %.12f / node-thread\n", total_time, total_time * thread_num / feat_size);
        printf("Push  Time Sum: %.6f, Average: %.12f / thread\n", vector_L1(time_push), vector_L1(time_push) / thread_num);
        printf("Read  Time Sum: %.6f, Average: %.12f / thread\n", vector_L1(time_read), vector_L1(time_read) / thread_num);
        printf("Write Time Sum: %.6f, Average: %.12f / thread\n", vector_L1(time_write), vector_L1(time_write) / thread_num);
    }

    void push() {
        // !: output matrix corrupts when split_num > 1
        for (VertexIdType spt_left = 0; spt_left < feat_size; spt_left += spt_size) {
            VertexIdType spt_right = std::min(feat_size, spt_left + spt_size);
            std::vector<std::thread> threads;

            double time_start = getCurrentTime();
            for (VertexIdType thd_left = spt_left; thd_left < spt_right; thd_left += thd_size) {
                VertexIdType thd_right = std::min(spt_right, thd_left + thd_size);
                VertexIdType tid = thd_left / thd_size;
                threads.emplace_back(std::thread(&Base::push_thread, this, thd_left, thd_right, tid));
            }
            for (auto &t : threads) {
                t.join();
            }
            total_time += getCurrentTime() - time_start;

            save_output(spt_left, spt_right);
        }
    }

    // No threading, debug use
    void push_single() {
        for (VertexIdType spt_left = 0; spt_left < feat_size; spt_left += spt_size) {
            VertexIdType spt_right = std::min(feat_size, spt_left + spt_size);
            double time_start = getCurrentTime();
            SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> graph_structure(V_num);
            std::vector<PageRankScoreType> seed;
            for (VertexIdType i = spt_left; i < spt_right; i++) {
                push_one(i, 0, graph_structure, seed);
            }
            total_time += getCurrentTime() - time_start;
            save_output(spt_left, spt_right);
        }
    }

};

class Base_reuse : public Base {

public:
    VertexIdType base_size;
    // statistics
    std::vector<double> time_reuse;
    PageRankScoreType avg_tht = 0;      // base theta
    PageRankScoreType avg_res = 0;      // reuse residue
    VertexIdType re_feat_num = 0;       // number of reused features

protected:
    std::vector<VertexIdType> base_idx; // index of base features
    MyMatrix base_matrix;               // matrix of base features
    MyMatrix base_result;               // output result (on all features and nodes)

public:

    Base_reuse(Graph &_graph, Param &_param) :
            Base(_graph, _param),
            base_size(std::max(VertexIdType (3u), VertexIdType (feat_size * param.base_ratio))),
            base_matrix(base_size, Vt_num),
            base_result(base_size, V_num) {
        base_idx = select_base(feature_matrix, base_matrix);
        printf("Base size: %ld \n", base_result.size());
        time_reuse.resize(thread_num, 0);
    }

    void push_one_base(const VertexIdType idx, const VertexIdType tid,
            SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> &_graph_structure,
            std::vector<PageRankScoreType> &_seed) {
        // printf("ID: %4" IDFMT "  as base\n", idx);
        double time_start = getCurrentTime();
        propagate_vector(base_matrix[idx], _seed, Vt_nodes, V_num, false);
        time_read[tid] += getCurrentTime() - time_start;

        time_start = getCurrentTime();
#ifdef ENABLE_RW
        if (param.index)
            ppr.calc_ppr_cache(_graph_structure, _seed, epsilon, alpha, lower_threshold, walkCache, param.gamma);
        else
            ppr.calc_ppr_walk(_graph_structure, _seed, epsilon, alpha, lower_threshold, param.gamma);
#else
        ppr.calc_ppr_walk(_graph_structure, _seed, epsilon, alpha, lower_threshold, param.gamma);
#endif
        time_push[tid] += getCurrentTime() - time_start;

        time_start = getCurrentTime();
        base_result.set_row(idx, _graph_structure.means);
        time_write[tid] += getCurrentTime() - time_start;
    }

    void push_thread_base(const VertexIdType feat_left, const VertexIdType feat_right, const VertexIdType tid) {
        // std::cout<<"  Pushing: "<<feat_left<<" "<<feat_right<<std::endl;
        SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> graph_structure(V_num);
        std::vector<PageRankScoreType> seed;
        for (VertexIdType i = feat_left; i < feat_right; i++) {
            push_one_base(i, tid, graph_structure, seed);
        }
    }

    void push_one_rest(const VertexIdType i, const VertexIdType tid,
            SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> &_graph_structure,
            std::vector<PageRankScoreType> &_seed, std::vector<PageRankScoreType> &_raw_seed) {
        for (VertexIdType idx = 0; idx < base_size; idx++) {
            if (base_idx[idx] == i) {
                // printf("ID: %4" IDFMT "  is base\n", i);
                double time_start = getCurrentTime();
                std::copy(base_result[idx].begin(), base_result[idx].end(), out_matrix[i%spt_size].begin());
                time_write[tid] += getCurrentTime() - time_start;
                return;
            }
        }

        double time_start = getCurrentTime();
        _raw_seed.swap(feature_matrix[i]);
        std::vector<PageRankScoreType> base_weight = reuse_weight(_raw_seed, base_matrix);
        PageRankScoreType theta_sum = vector_L1(base_weight);
        avg_tht += theta_sum;
        avg_res += vector_L1(_raw_seed);
        // printf("ID: %4" IDFMT ", theta_sum: %.6f, residue_sum: %.6f\n", i, theta_sum, vector_L1(_raw_seed));
        // Ignore less relevant features
        // if (theta_sum < 1.6) return;
        re_feat_num++;
        time_reuse[tid] += getCurrentTime() - time_start;

        time_start = getCurrentTime();
        propagate_vector(_raw_seed, _seed, Vt_nodes, V_num, true);
        time_read[tid] += getCurrentTime() - time_start;

        time_start = getCurrentTime();
#ifdef ENABLE_RW
        if (param.index)
            ppr.calc_ppr_cache(_graph_structure, _seed, epsilon, alpha, lower_threshold, walkCache, 2 - theta_sum * param.gamma);
        else
            ppr.calc_ppr_walk(_graph_structure, _seed, epsilon, alpha, lower_threshold, 2 - theta_sum * param.gamma);
#else
        ppr.calc_ppr_walk(_graph_structure, _seed, epsilon, alpha, lower_threshold, 2 - theta_sum * param.gamma);
#endif
        time_push[tid] += getCurrentTime() - time_start;

        time_start = getCurrentTime();
        for (VertexIdType idx = 0; idx < base_size; idx++){
            if (base_weight[idx] != 0) {
                for (VertexIdType j = 0; j < V_num; j++)
                    _graph_structure.means[j] += base_result[idx][j] * base_weight[idx];
            }
        }
        // Save embedding vector of feature i on all nodes to out_matrix
        std::swap_ranges(_graph_structure.means.begin(), _graph_structure.means.end()-2,
                         out_matrix[i%spt_size].begin());
        time_write[tid] += getCurrentTime() - time_start;
    }

    void push_thread_rest(const VertexIdType feat_left, const VertexIdType feat_right, const VertexIdType tid) {
        // std::cout<<"  Pushing: "<<feat_left<<" "<<feat_right<<std::endl;
        SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> graph_structure(V_num);
        std::vector<PageRankScoreType> seed;        // seed (length V_num) for push
        std::vector<PageRankScoreType> raw_seed;    // seed (length Vt_num) for reuse
        for (VertexIdType i = feat_left; i < feat_right; i++) {
            push_one_rest(i, tid, graph_structure, seed, raw_seed);
        }
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
        VertexIdType thread_num_base = std::min(base_size, thread_num);
        VertexIdType thd_size_base = (base_size + thread_num_base - 1) / thread_num_base;
        std::vector<std::thread> threads_base;

        double time_start = getCurrentTime();
        for (VertexIdType thd_left = 0; thd_left < base_size; thd_left += thd_size_base) {
            VertexIdType thd_right = std::min(base_size, thd_left + thd_size_base);
            VertexIdType tid = thd_left / thd_size_base;
            threads_base.emplace_back(std::thread(&Base_reuse::push_thread_base, this, thd_left, thd_right, tid));
        }
        for (auto &t : threads_base) {
                t.join();
            }
        total_time += getCurrentTime() - time_start;
        printf("Time Used on Base %.6f\n", total_time);

        // Calculate rest PPR
        for (VertexIdType spt_left = 0; spt_left < feat_size; spt_left += spt_size) {
            VertexIdType spt_right = std::min(feat_size, spt_left + spt_size);
            std::vector<std::thread> threads;

            double time_start = getCurrentTime();
            for (VertexIdType thd_left = spt_left; thd_left < spt_right; thd_left += thd_size) {
                VertexIdType thd_right = std::min(spt_right, thd_left + thd_size);
                VertexIdType tid = thd_left / thd_size;
                threads.emplace_back(std::thread(&Base_reuse::push_thread_rest, this, thd_left, thd_right, tid));
            }
            for (auto &t : threads) {
                t.join();
            }
            total_time += getCurrentTime() - time_start;

            save_output(spt_left, spt_right);
        }
    }

    // No multithreading, debug use
    void push_single() {
        SpeedPPR::WHOLE_GRAPH_STRUCTURE<PageRankScoreType> graph_structure(V_num);
        std::vector<PageRankScoreType> seed;        // seed (length V_num) for push
        std::vector<PageRankScoreType> raw_seed;    // seed (length Vt_num) for reuse
        // Calculate base PPR
        double time_start = getCurrentTime();
        for(VertexIdType i = 0; i < base_size; i++){
            push_one_base(i, 0, graph_structure, seed);
        }
        total_time += getCurrentTime() - time_start;
        printf("Time Used on Base %.6f\n", total_time);
        // Calculate rest PPR
        for (VertexIdType spt_left = 0; spt_left < feat_size; spt_left += spt_size) {
            VertexIdType spt_right = std::min(feat_size, spt_left + spt_size);
            double time_start = getCurrentTime();
            for (VertexIdType i = spt_left; i < spt_right; i++) {
                push_one_rest(i, 0, graph_structure, seed, raw_seed);
            }
            total_time += getCurrentTime() - time_start;
            save_output(spt_left, spt_right);
        }
    }

};


class Base_pca : public Base {
public:
    VertexIdType base_size;
    // statistics
    std::vector<double> time_reuse;
    PageRankScoreType avg_tht = 0;      // base theta
    PageRankScoreType avg_res = 0;      // reuse residue
    VertexIdType re_feat_num = 0;       // number of reused features

protected:
    MyMatrix theta_matrix;              // matrix of principal directions
    MyMatrix base_matrix;               // matrix of principal components
    MyMatrix base_result;               // output result (on all features and nodes)

public:

    Base_pca (Graph &_graph, Param &_param) :
            Base(_graph, _param),
            base_size(std::max(VertexIdType (3u), VertexIdType (feat_size * param.base_ratio))),
            base_matrix(base_size, Vt_num),
            base_result(base_size, V_num) {
        theta_matrix = select_pc(feature_matrix, base_matrix);
        printf("Base size: %ld \n", base_result.size());
        time_reuse.resize(thread_num, 0);
    }

};
