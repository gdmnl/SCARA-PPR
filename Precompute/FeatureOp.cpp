/*
  Implementation of embedding computation from feature
  Author: nyLiao
*/
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
    IntVector   Vt_nodes;       // list of queried nodes
    My2DVector  feat_v2d;       // feature matrix data
    MyMatrix    feat_matrix;    // feature matrix mapped as vec of vec
#ifdef ENABLE_RW
    WalkCache walkCache;
#endif

    NInt thread_num;            // number of threads
    ScoreFlt epsilon;
    ScoreFlt alpha;
    ScoreFlt lower_threshold;

public:
    NInt V_num;                 // number of vertices
    NInt Vt_num;                // number of queried nodes
    NInt feat_size;             // size of feature
    NInt thd_size;              // size of feature per thread
    // statistics
    double total_time = 0;
    std::vector<double> time_read;
    std::vector<double> time_write;
    std::vector<double> time_push;
    std::vector<double> time_init;
    std::vector<double> time_fp;
    std::vector<double> time_it;
    std::vector<double> time_rw;

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
        Vt_num = load_query(Vt_nodes, param.query_file, V_num);
        feat_v2d.load_npy(param.feature_file);
        feat_size = feat_v2d.nrows();
        feat_matrix.set_size(feat_size, V_num);
        feat_matrix.from_V2D(feat_v2d, Vt_nodes);
        feat_v2d.clear();
#ifdef ENABLE_RW
        // Perform cached random walk
        if (param.index) {
            graph.set_dummy_neighbor(graph.get_dummy_id());
            walkCache.generate();
            graph.reset_set_dummy_neighbor();
        }
#endif
        graph.fill_dead_end_neighbor_with_id();

        thread_num = (NInt) param.thread_num;
        thd_size = (feat_size + thread_num - 1) / thread_num;
        time_read.resize(thread_num, 0);
        time_write.resize(thread_num, 0);
        time_push.resize(thread_num, 0);
        time_init.resize(thread_num, 0);
        time_fp.resize(thread_num, 0);
        time_it.resize(thread_num, 0);
        time_rw.resize(thread_num, 0);
    }

    void push_one(const NInt i, const NInt tid, SpeedPPR::GStruct<ScoreFlt> &_gstruct) {
        // printf("ID: %4" IDFMT "\n", i);
        double time_start = getCurrentTime();
#ifdef ENABLE_RW
        if (param.index)
            ppr.calc_ppr_cache(_gstruct, feat_matrix[i], epsilon, alpha, lower_threshold, walkCache);
        else
            ppr.calc_ppr_walk(_gstruct, feat_matrix[i], epsilon, alpha, lower_threshold);
#else
        ppr.calc_ppr_walk(_gstruct, feat_matrix[i], epsilon, alpha, lower_threshold);
#endif
        time_push[tid] += getCurrentTime() - time_start;

        // Save embedding vector of feature i on all nodes to feat_matrix in place
        time_start = getCurrentTime();
        feat_matrix[i].swap(_gstruct.means);
        time_write[tid] += getCurrentTime() - time_start;
    }

    void push_thread(const NInt feat_left, const NInt feat_right, const NInt tid) {
        // cout<<"  Pushing: "<<feat_left<<" "<<feat_right<<endl;
        SpeedPPR::GStruct<ScoreFlt> gstruct(V_num);
        for (NInt i = feat_left; i < feat_right; i++) {
            push_one(i, tid, gstruct);
        }
        time_init[tid] = gstruct.time_init;
        time_fp[tid]   = gstruct.time_fp;
        time_it[tid]   = gstruct.time_it;
        time_rw[tid]   = gstruct.time_rw;
    }

    void save_output(const NInt feat_left, const NInt feat_right) {
        if (param.output_estimations) {
            std::stringstream res_file;
            res_file << param.estimation_folder << "/score_" << param.alpha << '_' << param.epsilon << ".npy";
            feat_matrix.to_V2D(feat_v2d);
            feat_v2d.save_npy(res_file.str());
        }
    }

    void show_statistics() {
        printf("Mem: %ld MB\n", get_proc_memory()/1000);
        printf("Total Time    : %.6f, Average: %.12f / node-thread\n", total_time, total_time * thread_num / feat_size);
        printf("Push  Time Sum: %.6f, Average: %.12f / thread\n", vector_L1(time_push), vector_L1(time_push) / thread_num);
        printf("  Init     Sum: %.6f, Average: %.12f / thread\n", vector_L1(time_init), vector_L1(time_init) / thread_num);
        printf("  FwdPush  Sum: %.6f, Average: %.12f / thread\n", vector_L1(time_fp), vector_L1(time_fp) / thread_num);
        printf("  PIter    Sum: %.6f, Average: %.12f / thread\n", vector_L1(time_it), vector_L1(time_it) / thread_num);
        printf("  RW       Sum: %.6f, Average: %.12f / thread\n", vector_L1(time_rw), vector_L1(time_rw) / thread_num);
        // printf("Read  Time Sum: %.6f, Average: %.12f / thread\n", vector_L1(time_read), vector_L1(time_read) / thread_num);
        printf("Write Time Sum: %.6f, Average: %.12f / thread\n", vector_L1(time_write), vector_L1(time_write) / thread_num);
    }

    void push() {
        std::vector<std::thread> threads;

        double time_start = getCurrentTime();
        for (NInt thd_left = 0; thd_left < feat_size; thd_left += thd_size) {
            NInt thd_right = std::min(feat_size, thd_left + thd_size);
            NInt tid = thd_left / thd_size;
            threads.emplace_back(std::thread(&Base::push_thread, this, thd_left, thd_right, tid));
        }
        for (auto &t : threads) {
            t.join();
        }
        total_time += getCurrentTime() - time_start;

        save_output(0, feat_size);
    }

    // No threading, debug use
    void push_single() {
        double time_start = getCurrentTime();
        SpeedPPR::GStruct<ScoreFlt> gstruct(V_num);
        FltVector seed;
        for (NInt i = 0; i < feat_size; i++) {
            push_one(i, 0, gstruct);
        }
        total_time += getCurrentTime() - time_start;
        save_output(0, feat_size);
    }

};

/*
class Base_reuse : public Base {

public:
    NInt base_size;
    // statistics
    std::vector<double> time_reuse;
    ScoreFlt avg_tht = 0;      // base theta
    ScoreFlt avg_res = 0;      // reuse residue
    NInt re_feat_num = 0;       // number of reused features

protected:
    IntVector base_idx; // index of base features
    MyMatrix base_matrix;               // matrix of base features
    MyMatrix base_result;               // output result (on all features and nodes)

public:

    Base_reuse(Graph &_graph, Param &_param) :
            Base(_graph, _param),
            base_size(std::max(NInt (3u), NInt (feat_size * param.base_ratio))),
            base_matrix(base_size, Vt_num),
            base_result(base_size, V_num) {
        base_idx = select_base(feature_matrix, base_matrix);
        printf("Base size: %ld \n", base_result.size());
        time_reuse.resize(thread_num, 0);
    }

    void push_one_base(const NInt idx, const NInt tid,
            SpeedPPR::GStruct<ScoreFlt> &_gstruct, FltVector &_seed) {
        // printf("ID: %4" IDFMT "  as base\n", idx);
        double time_start = getCurrentTime();
        propagate_vector(base_matrix[idx], _seed, Vt_nodes, V_num, false);
        time_read[tid] += getCurrentTime() - time_start;

        time_start = getCurrentTime();
#ifdef ENABLE_RW
        if (param.index)
            ppr.calc_ppr_cache(_gstruct, _seed, epsilon, alpha, lower_threshold, walkCache, param.gamma);
        else
            ppr.calc_ppr_walk(_gstruct, _seed, epsilon, alpha, lower_threshold, param.gamma);
#else
        ppr.calc_ppr_walk(_gstruct, _seed, epsilon, alpha, lower_threshold, param.gamma);
#endif
        time_push[tid] += getCurrentTime() - time_start;

        time_start = getCurrentTime();
        base_result.set_row(idx, _gstruct.means);
        time_write[tid] += getCurrentTime() - time_start;
    }

    void push_thread_base(const NInt feat_left, const NInt feat_right, const NInt tid) {
        // cout<<"  Pushing: "<<feat_left<<" "<<feat_right<<endl;
        SpeedPPR::GStruct<ScoreFlt> gstruct(V_num);
        FltVector seed;
        for (NInt i = feat_left; i < feat_right; i++) {
            push_one_base(i, tid, gstruct, seed);
        }
    }

    void push_one_rest(const NInt i, const NInt tid,
            SpeedPPR::GStruct<ScoreFlt> &_gstruct, FltVector &_seed, FltVector &_raw_seed) {
        for (NInt idx = 0; idx < base_size; idx++) {
            if (base_idx[idx] == i) {
                // printf("ID: %4" IDFMT "  is base\n", i);
                double time_start = getCurrentTime();
                std::copy(base_result[idx].begin(), base_result[idx].end(), out_matrix[i].begin());
                time_write[tid] += getCurrentTime() - time_start;
                return;
            }
        }

        double time_start = getCurrentTime();
        _raw_seed.swap(feature_matrix[i]);
        FltVector base_weight = reuse_weight(_raw_seed, base_matrix);
        ScoreFlt theta_sum = vector_L1(base_weight);
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
            ppr.calc_ppr_cache(_gstruct, _seed, epsilon, alpha, lower_threshold, walkCache, 2 - theta_sum * param.gamma);
        else
            ppr.calc_ppr_walk(_gstruct, _seed, epsilon, alpha, lower_threshold, 2 - theta_sum * param.gamma);
#else
        ppr.calc_ppr_walk(_gstruct, _seed, epsilon, alpha, lower_threshold, 2 - theta_sum * param.gamma);
#endif
        time_push[tid] += getCurrentTime() - time_start;

        time_start = getCurrentTime();
        for (NInt idx = 0; idx < base_size; idx++){
            if (base_weight[idx] != 0) {
                for (NInt j = 0; j < V_num; j++)
                    _gstruct.means[j] += base_result[idx][j] * base_weight[idx];
            }
        }
        // Save embedding vector of feature i on all nodes to out_matrix
        std::swap_ranges(_gstruct.means.begin(), _gstruct.means.end()-2,
                         out_matrix[i].begin());
        time_write[tid] += getCurrentTime() - time_start;
    }

    void push_thread_rest(const NInt feat_left, const NInt feat_right, const NInt tid) {
        // cout<<"  Pushing: "<<feat_left<<" "<<feat_right<<endl;
        SpeedPPR::GStruct<ScoreFlt> gstruct(V_num);
        FltVector seed;        // seed (length V_num) for push
        FltVector raw_seed;    // seed (length Vt_num) for reuse
        for (NInt i = feat_left; i < feat_right; i++) {
            push_one_rest(i, tid, gstruct, seed, raw_seed);
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
        NInt thread_num_base = std::min(base_size, thread_num);
        NInt thd_size_base = (base_size + thread_num_base - 1) / thread_num_base;
        std::vector<std::thread> threads_base;

        double time_start = getCurrentTime();
        for (NInt thd_left = 0; thd_left < base_size; thd_left += thd_size_base) {
            NInt thd_right = std::min(base_size, thd_left + thd_size_base);
            NInt tid = thd_left / thd_size_base;
            threads_base.emplace_back(std::thread(&Base_reuse::push_thread_base, this, thd_left, thd_right, tid));
        }
        for (auto &t : threads_base) {
                t.join();
            }
        total_time += getCurrentTime() - time_start;
        printf("Time Used on Base %.6f\n", total_time);

        // Calculate rest PPR
        std::vector<std::thread> threads;

        time_start = getCurrentTime();
        for (NInt thd_left = 0; thd_left < feat_size; thd_left += thd_size) {
            NInt thd_right = std::min(feat_size, thd_left + thd_size);
            NInt tid = thd_left / thd_size;
            threads.emplace_back(std::thread(&Base_reuse::push_thread_rest, this, thd_left, thd_right, tid));
        }
        for (auto &t : threads) {
            t.join();
        }
        total_time += getCurrentTime() - time_start;

        save_output(0, feat_size);
    }

    // No multithreading, debug use
    void push_single() {
        SpeedPPR::GStruct<ScoreFlt> gstruct(V_num);
        FltVector seed;        // seed (length V_num) for push
        FltVector raw_seed;    // seed (length Vt_num) for reuse
        // Calculate base PPR
        double time_start = getCurrentTime();
        for(NInt i = 0; i < base_size; i++){
            push_one_base(i, 0, gstruct, seed);
        }
        total_time += getCurrentTime() - time_start;
        printf("Time Used on Base %.6f\n", total_time);
        // Calculate rest PPR
        time_start = getCurrentTime();
        for (NInt i = 0; i < feat_size; i++) {
            push_one_rest(i, 0, gstruct, seed, raw_seed);
        }
        total_time += getCurrentTime() - time_start;
        save_output(0, feat_size);
    }

};


class Base_pca : public Base {
public:
    NInt base_size;
    // statistics
    std::vector<double> time_reuse;
    ScoreFlt avg_tht = 0;      // base theta
    ScoreFlt avg_res = 0;      // reuse residue
    NInt re_feat_num = 0;       // number of reused features

protected:
    MyMatrix theta_matrix;              // matrix of principal directions
    MyMatrix base_matrix;               // matrix of principal components
    MyMatrix base_result;               // output result (on all features and nodes)

public:

    Base_pca (Graph &_graph, Param &_param) :
            Base(_graph, _param),
            base_size(std::max(NInt (3u), NInt (feat_size * param.base_ratio))),
            base_matrix(base_size, Vt_num),
            base_result(base_size, V_num) {
        theta_matrix = select_pc(feature_matrix, base_matrix);
        printf("Base size: %ld \n", base_result.size());
        time_reuse.resize(thread_num, 0);
    }

};
*/