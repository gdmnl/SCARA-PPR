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


// Wrapper processor class: feat-push
class FeatProc {
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
    double time_total = 0;
    std::vector<double> time_read, time_write, time_push;
    std::vector<double> time_init, time_fp, time_it, time_rw;

protected:

    void push_one(const NInt i, const NInt tid, SpeedPPR::GStruct<ScoreFlt> &_gstruct) {
        // printf("ID: %4" IDFMT "\n", i);
        double time_start = getCurrentTime();
#ifdef ENABLE_RW
        if (param.index)
            ppr.calc_ppr_cache(_gstruct, feat_matrix[i], Vt_nodes, epsilon, alpha, lower_threshold, walkCache);
        else
            ppr.calc_ppr_walk(_gstruct, feat_matrix[i], Vt_nodes, epsilon, alpha, lower_threshold);
#else
        ppr.calc_ppr_walk(_gstruct, feat_matrix[i], Vt_nodes, epsilon, alpha, lower_threshold);
#endif
        time_push[tid] += getCurrentTime() - time_start;

        // Save embedding vector of feature i on all nodes to feat_matrix in place
        time_start = getCurrentTime();
        feat_matrix[i].swap(_gstruct.means);        // feat_matrix[i] = _gstruct.means;
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

public:

    FeatProc(Graph &_graph, Param &_param) :
            V_num(_graph.getNumOfVertices()),
            ppr(_graph),
#ifdef ENABLE_RW
            walkCache(_graph),
#endif
            feat_matrix(2),
            epsilon(_param.epsilon),
            alpha(_param.alpha),
            lower_threshold(1.0 / _graph.getNumOfVertices()),
            graph(_graph),
            param(_param) {
        printf("Adj    RSS RAM: %.3f GB\n", get_stat_memory());
        Vt_num = load_query(Vt_nodes, param.query_file, V_num);
        feat_v2d.load_npy(param.feature_file);
        feat_size = feat_v2d.nrows();
        feat_matrix.set_size(feat_size, V_num);
        feat_matrix.from_V2D(feat_v2d, Vt_nodes);
        printf("Max   RSS PRAM: %.3f GB\n", get_proc_memory());
#ifdef ENABLE_RW
        // Perform cached random walk
        if (param.index) {
            graph.set_dummy_neighbor(graph.get_dummy_id());
            // walkCache.generate();
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

    void save_output(const NInt feat_left, const NInt feat_right) {
        if (param.output_estimations) {
            std::stringstream res_file;
            res_file << param.estimation_folder << "/score_" << param.alpha << '_' << param.epsilon << ".npy";
            feat_matrix.to_V2D(feat_v2d);
            feat_v2d.save_npy(res_file.str());
        }
    }

    void show_statistics() {
        printf("%s\n", std::string(80, '-').c_str());
        printf("Max RURSS PRAM: %.3f GB\n", get_proc_memory());
        printf("End    RSS RAM: %.3f GB\n", get_stat_memory());
        printf("Total Time    : %.6f, Average: %.8f / node-thread\n", time_total, time_total * thread_num / feat_size);
        printf("Push  Time Sum: %.6f, Average: %.8f / thread\n", vector_L1(time_push), vector_L1(time_push) / thread_num);
        printf("  Init     Sum: %.6f, Average: %.8f / thread\n", vector_L1(time_init), vector_L1(time_init) / thread_num);
        printf("  FwdPush  Sum: %.6f, Average: %.8f / thread\n", vector_L1(time_fp), vector_L1(time_fp) / thread_num);
        printf("  PwrIter  Sum: %.6f, Average: %.8f / thread\n", vector_L1(time_it), vector_L1(time_it) / thread_num);
        printf("  RW       Sum: %.6f, Average: %.8f / thread\n", vector_L1(time_rw), vector_L1(time_rw) / thread_num);
        // printf("Read  Time Sum: %.6f, Average: %.8f / thread\n", vector_L1(time_read), vector_L1(time_read) / thread_num);
        printf("Write Time Sum: %.6f, Average: %.8f / thread\n", vector_L1(time_write), vector_L1(time_write) / thread_num);
    }

    void push() {
        std::vector<std::thread> threads;

        double time_start = getCurrentTime();
        for (NInt thd_left = 0; thd_left < feat_size; thd_left += thd_size) {
            NInt thd_right = std::min(feat_size, thd_left + thd_size);
            NInt tid = thd_left / thd_size;
            threads.emplace_back(std::thread(&FeatProc::push_thread, this, thd_left, thd_right, tid));
        }
        for (auto &t : threads) {
            t.join();
        }
        time_total += getCurrentTime() - time_start;

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
        time_total += getCurrentTime() - time_start;
        save_output(0, feat_size);
    }

};


// Feat-reuse template
class FeatProc_reuse : public FeatProc {
private:
    virtual ScoreFlt reduce_feat(const NInt i, FltVector &base_weight) = 0;

public:
    NInt base_size;
    // statistics
    std::vector<double> time_reuse;
    ScoreFlt avg_tht = 0;       // average base coefficient
    ScoreFlt avg_res = 0;       // average reuse residue
    NInt re_feat_num = 0;       // number of reused features

protected:
    ScoreFlt gamma;
    IntVector base_idx;         // index of base features
    MyMatrix base_matrix;       // matrix of base features
    MyMatrix base_result;       // output result (on all features and nodes)
    ScoreFlt TOL;               // tolerance for reuse coefficient

protected:

    void push_one_base(const NInt idx, const NInt tid, SpeedPPR::GStruct<ScoreFlt> &_gstruct) {
        // printf("ID: %4" IDFMT "  as base\n", idx);
        double time_start = getCurrentTime();
#ifdef ENABLE_RW
        if (param.index)
            ppr.calc_ppr_cache(_gstruct, feat_matrix[base_idx[idx]], Vt_nodes, epsilon, alpha, lower_threshold, walkCache, gamma);
        else
            ppr.calc_ppr_walk(_gstruct, feat_matrix[base_idx[idx]], Vt_nodes, epsilon, alpha, lower_threshold, gamma);
#else
        ppr.calc_ppr_walk(_gstruct, feat_matrix[base_idx[idx]], Vt_nodes, epsilon, alpha, lower_threshold, gamma);
#endif
        time_push[tid] += getCurrentTime() - time_start;

        time_start = getCurrentTime();
        base_result[idx].swap(_gstruct.means);      // base_result[idx] = _gstruct.means;
        time_write[tid] += getCurrentTime() - time_start;
    }

    void push_thread_base(const NInt feat_left, const NInt feat_right, const NInt tid) {
        // cout<<"  Pushing: "<<feat_left<<" "<<feat_right<<endl;
        SpeedPPR::GStruct<ScoreFlt> gstruct(V_num);
        for (NInt i = feat_left; i < feat_right; i++) {
            push_one_base(i, tid, gstruct);
        }
        time_init[tid] += gstruct.time_init;
        time_fp[tid]   += gstruct.time_fp;
        time_it[tid]   += gstruct.time_it;
        time_rw[tid]   += gstruct.time_rw;
    }

    void push_one_rest(const NInt i, const NInt tid, SpeedPPR::GStruct<ScoreFlt> &_gstruct) {
        for (NInt idx = 0; idx < base_size; idx++) {
            if (base_idx[idx] == i) {
                // printf("ID: %4" IDFMT "  is base\n", i);
                double time_start = getCurrentTime();
                feat_matrix.copy_row(i, base_result[idx]);
                time_write[tid] += getCurrentTime() - time_start;
                return;
            }
        }

        // ===== Base reduction
        double time_start = getCurrentTime();
        FltVector base_weight;
        ScoreFlt theta_sum = reduce_feat(i, base_weight);
        time_reuse[tid] += getCurrentTime() - time_start;
        // printf("ID: %4" IDFMT ", theta_sum: %.6f, residue_sum: %.6f\n", i, theta_sum, vector_L1(_raw_seed));
        // Ignore less relevant features
        // if (theta_sum < 1.6) return;
        avg_tht += theta_sum;
        re_feat_num++;
#ifdef DEBUG
        ScoreFlt res_i = vector_L1(feat_matrix[i]);
        avg_res += res_i;
        // if (i < 10)
        //     cout<<"Re "<<i<<": theta "<<theta_sum<<" L1 "<<res_i<<" L2 "<<vector_L2(feat_matrix[i])<<endl;
#endif

        time_start = getCurrentTime();
#ifdef ENABLE_RW
        if (param.index)
            ppr.calc_ppr_cache(_gstruct, feat_matrix[i], Vt_nodes, epsilon, alpha, lower_threshold, walkCache, 2 - theta_sum * gamma);
        else
            ppr.calc_ppr_walk(_gstruct, feat_matrix[i], Vt_nodes, epsilon, alpha, lower_threshold, 2 - theta_sum * gamma);
#else
        ppr.calc_ppr_walk(_gstruct, feat_matrix[i], Vt_nodes, epsilon, alpha, lower_threshold, 2 - theta_sum * gamma);
#endif
        time_push[tid] += getCurrentTime() - time_start;

        // ===== Residue reuse
        time_start = getCurrentTime();
        for (NInt idx = 0; idx < base_size; idx++) {
            if (fabs(base_weight[idx]) > TOL) {
                for (NInt j = 0; j < V_num; j++)
                    _gstruct.means[j] += base_result[idx][j] * base_weight[idx];
            }
        }
        // Save embedding vector of feature i on all nodes to out_matrix
        feat_matrix[i].swap(_gstruct.means);        // feat_matrix[i] = _gstruct.means;
        time_write[tid] += getCurrentTime() - time_start;
    }

    void push_thread_rest(const NInt feat_left, const NInt feat_right, const NInt tid) {
        // cout<<"  Pushing: "<<feat_left<<" "<<feat_right<<endl;
        SpeedPPR::GStruct<ScoreFlt> gstruct(V_num);
        for (NInt i = feat_left; i < feat_right; i++) {
            push_one_rest(i, tid, gstruct);
        }
        time_init[tid] += gstruct.time_init;
        time_fp[tid]   += gstruct.time_fp;
        time_it[tid]   += gstruct.time_it;
        time_rw[tid]   += gstruct.time_rw;
    }

public:

    FeatProc_reuse(Graph &_graph, Param &_param) :
            FeatProc(_graph, _param),
            gamma(_param.gamma),
            base_size(std::max(NInt (3u), NInt (feat_size * param.base_ratio))),
            base_matrix(base_size, V_num, 2),
            base_result(base_size, V_num, 2) {
        time_reuse.resize(thread_num, 0);
    }

    void show_statistics() {
        avg_tht /= re_feat_num;
        avg_res /= re_feat_num;
        MSG(avg_tht);
        MSG(avg_res);
        MSG(re_feat_num);
        FeatProc::show_statistics();
        printf("Reuse Time Sum: %.6f, Average: %.8f / thread\n", vector_L1(time_reuse), vector_L1(time_reuse) / thread_num);
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
            threads_base.emplace_back(std::thread(&FeatProc_reuse::push_thread_base, this, thd_left, thd_right, tid));
        }
        for (auto &t : threads_base) {
                t.join();
            }
        time_total += getCurrentTime() - time_start;
        printf("Time Used on Base %.6f\n", time_total);

        // Calculate rest PPR
        std::vector<std::thread> threads;

        time_start = getCurrentTime();
        for (NInt thd_left = 0; thd_left < feat_size; thd_left += thd_size) {
            NInt thd_right = std::min(feat_size, thd_left + thd_size);
            NInt tid = thd_left / thd_size;
            threads.emplace_back(std::thread(&FeatProc_reuse::push_thread_rest, this, thd_left, thd_right, tid));
        }
        for (auto &t : threads) {
            t.join();
        }
        time_total += getCurrentTime() - time_start;

        save_output(0, feat_size);
    }

    // No multithreading, debug use
    void push_single() {
        SpeedPPR::GStruct<ScoreFlt> gstruct(V_num);
        // Calculate base PPR
        double time_start = getCurrentTime();
        for(NInt i = 0; i < base_size; i++){
            push_one_base(i, 0, gstruct);
        }
        time_total += getCurrentTime() - time_start;
        printf("Time Used on Base %.6f\n", time_total);
        // Calculate rest PPR
        time_start = getCurrentTime();
        for (NInt i = 0; i < feat_size; i++) {
            push_one_rest(i, 0, gstruct);
        }
        time_total += getCurrentTime() - time_start;
        save_output(0, feat_size);
    }

};

// Feat-reuse: greedy
class FeatProc_greedy : public FeatProc_reuse {
private:
    MyMatrix base_inv;

    /* Compute coefficients (theta) of bases, feat_matrix[i] is updated to residue */
    ScoreFlt reduce_feat(const NInt i, FltVector &base_weight) {
        base_weight.resize(base_size);
        base_weight = reuse_weight(feat_matrix[i], base_matrix);

        /* Least square regression on x = B * w */
        // std::fill(base_weight.begin(), base_weight.end(), 0.0);
        // for (NInt idx = 0; idx < base_size; idx++) {
        //     for (NInt j = 0; j < V_num; j++) {
        //         base_weight[idx] += feat_matrix[i][j] * base_inv[idx][j];
        //     }
        //     if (fabs(base_weight[idx]) > TOL) {
        //         for (NInt j = 0; j < V_num; j++)
        //             feat_matrix[i][j] -= base_matrix[idx][j] * base_weight[idx];
        //     }
        // }
        return vector_L1(base_weight);
    }

public:

    FeatProc_greedy (Graph &_graph, Param &_param) :
            FeatProc_reuse(_graph, _param),
            base_inv(base_size, V_num) {
        TOL = 1e-2;
    }

    void fit() {
        base_idx = select_base(feat_matrix, base_size);
        base_matrix.copy_rows(base_idx, feat_matrix);
        // ScoreMatrix base_Matrix_ = base_matrix.to_Eigen();
        // RandomizedSvd rsvd(base_Matrix_, base_size);
        // ScoreMatrix base_Inv_ = rsvd.pinv();
        // base_Inv_.transposeInPlace();
        // base_inv.from_Eigen(base_Inv_);
        cout<<"Base  size: "<<base_result.nrows()<<" "<<base_result.ncols()<<" "<<base_result.size()<<endl;
    }

};

// Feat-reuse: PCA
class FeatProc_pca : public FeatProc_reuse {
protected:
    MyMatrix theta_matrix;      // matrix of principal directions

private:

    /* Compute coefficients (theta) of bases, feat_matrix[i] is updated to residue */
    ScoreFlt reduce_feat(const NInt i, FltVector &base_weight) {
        base_weight.swap(theta_matrix[i]);
        for (NInt idx = 0; idx < base_size; idx++) {
            if (fabs(base_weight[idx]) > TOL) {
                for (NInt j = 0; j < V_num; j++)
                    feat_matrix[i][j] -= base_matrix[idx][j] * base_weight[idx];
            }
        }

        return vector_L1(base_weight);
    }

public:

    FeatProc_pca (Graph &_graph, Param &_param) :
            FeatProc_reuse(_graph, _param),
            theta_matrix(feat_size, base_size) {
        TOL = 1e-2;
    }

    void fit() {
        ScoreFlt avg_degree = graph.getNumOfEdges() / (ScoreFlt) graph.getNumOfVertices() / 2;
        // NInt        Vs_num = std::max(3*base_size, NInt (V_num*param.base_ratio));
        NInt        Vs_num = ceil(0.1 * V_num);
        IntVector   Vs_nodes = sample_nodes(Vt_nodes, Vs_num);
        ScoreMatrix feat_sample_Matrix = feat_matrix.to_Eigen(Vs_nodes);
        base_idx = select_pc(feat_sample_Matrix, theta_matrix, base_size, sqrt(avg_degree));
        base_matrix.copy_rows(base_idx, feat_matrix);
        cout<<"Theta size: "<<theta_matrix.nrows()<<" "<<theta_matrix.ncols()<<" "<<theta_matrix.size()<<endl;
        cout<<"PBase size: "<<base_result.nrows()<<" "<<base_result.ncols()<<" "<<base_result.size()<<endl;
    }

};
