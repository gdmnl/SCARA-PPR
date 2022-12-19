/*
  Feature calculation
  Author: nyLiao
*/
#ifndef SCARA_FEATUREOP_H
#define SCARA_FEATUREOP_H

#include <vector>
#include <iostream>
#include <cmath>
#include "BasicDefinition.h"
#include "MyType.h"
#include "FeatureDecomp.h"


// ==================== Basic
template<class FLT>
inline FLT vector_L1(const std::vector<FLT> Vec){
    FLT sum = 0;
    for(FLT a : Vec)
        sum += fabsf(a);
    return sum;
}

template<class FLT>
inline IntVector arg_kmax(const std::vector<FLT> &Vec, const NInt k) {
    std::priority_queue<std::pair<FLT, NInt>,
                        std::vector< std::pair<FLT, NInt> >,
                        std::less<std::pair<FLT, NInt> >> q;
    for(NInt i = 0; i < Vec.size(); i++){
        if (q.size() < k)
            q.push({Vec[i], i});
        else if(Vec[i] < q.top().first){
            q.pop();
            q.push({Vec[i], i});
        }
    }
    IntVector res(k);
    for(NInt i = 0; i < k; i++){
        res[i] = q.top().second;
        q.pop();
    }
    return res;
}

// ==================== Vector measurement
template <class FLT>
inline FLT calc_L1_residue(std::vector<FLT> &V1, const std::vector<FLT> &V2, const float pace = 1.0) {
    NInt index;
    FLT used_sum = 0;
    FLT theta;
    // std::vector<std::pair<NInt, FLT>> theta_counter;
    std::vector<IdScorePair<FLT>> theta_counter;
    for (NInt i = 0; i < V1.size(); i++) {
        if (V2[i] != 0) {
            theta = V1[i] / V2[i];
            theta_counter.push_back({i, theta});
        }
    }
    std::sort(theta_counter.begin(), theta_counter.end(),
              [](const IdScorePair<FLT> &a, const IdScorePair<FLT> &b) {
                  return fabsf(a.score) < fabsf(b.score);
              });
    for (NInt i = 0; i < theta_counter.size(); i += 1) {
        index = theta_counter[i].id;
        theta = theta_counter[i].score;
        used_sum += fabsf(V2[index]);
        if (used_sum > 0.5 || index == -1)
            break;
    }

    FLT orig_sum = 0;
    FLT diff_sum = 0;
    for (NInt i = 0; i < V1.size(); i++) {
        orig_sum += fabsf(V1[i]);
        diff_sum += fabsf(V1[i] - theta * V2[i] * pace);
    }
    if (diff_sum > orig_sum)
        return 0;

    for (NInt i = 0; i < V1.size(); i++) {
        V1[i] = V1[i] - theta * V2[i] * pace;
    }
    // printf("theta: %.6f, residue: %.6f\n", theta, diff_sum);
    return theta * pace;
}

template <class FLT>
inline FLT calc_L2_residue(std::vector<FLT> &V1, const std::vector<FLT> &V2, const float pace = 1.0) {
    FLT prd = 0;
    FLT sum2 = 0;
    for (NInt i = 0; i < V1.size(); i++) {
        prd += V1[i] * V2[i];
        sum2 += V2[i] * V2[i];
    }
    FLT theta = prd / sum2;

    FLT orig_sum = 0;
    FLT diff_sum = 0;
    for (NInt i = 0; i < V1.size(); i++) {
        orig_sum += fabsf(V1[i]);
        diff_sum += fabsf(V1[i] - theta * V2[i] * pace);
    }
    if (diff_sum > orig_sum)
        return 0;

    for (NInt i = 0; i < V1.size(); i++) {
        V1[i] = V1[i] - theta * V2[i] * pace;
    }
    // printf("theta: %.6f, residue: %.6f\n", theta, diff_sum);
    return theta * pace;
}

template <class FLT>
inline FLT calc_L1_distance(const std::vector<FLT> &V1, const std::vector<FLT> &V2) {
    /* Ranges:
        V~[0, 1] -> distance~[0, 1]
        V~[-1, 1] -> distance~[0, 2]
    */
    FLT distance = 0;
    for (NInt i = 0; i < V1.size(); i++) {
        distance += fabsf(V1[i] - V2[i]);
    }
    return distance;
}

template <class FLT>
inline FLT calc_L2_distance(const std::vector<FLT> &V1, const std::vector<FLT> &V2) {
    /* Ranges: (cosine angle = prd / (sqrt(sum1) * sqrt(sum2)))
        V~[0, 1] -> distance~[0, 1]
        V~[-1, 1] -> distance~[0, 2]
    */
    // TODO: cache for feature-feature product and difference
    FLT distance = 0;
    FLT prd = 0;
    FLT sum1 = 0;
    FLT sum2 = 0;
    for (NInt i = 0; i < V1.size(); i++) {
        prd += V1[i] * V2[i];
        sum1 += V1[i] * V1[i];
        sum2 += V2[i] * V2[i];
    }
    // distance = 1 - prd / (sqrt(sum1) * sqrt(sum2));
    distance = 1 - fabsf(prd / (sqrt(sum1) * sqrt(sum2)));
    return distance;
}

// ==================== Reuse functions
inline IntVector select_base(MyMatrix &feat_matrix, const NInt base_size) {
    NInt feat_size = feat_matrix.nrows();
    IntVector base_idx(base_size, 0);                                       // index of base features
    std::vector<IdScorePair<ScoreFlt>> min_counter(feat_size, {0, 0.0});    // (min base id, min norm) for each feature
    // Find minimum distance feature for each feature
    for (NInt i = 0; i < feat_size; i++) {
        ScoreFlt dis_min = 4.0 * feat_size;
        NInt idx_min = -1;
        for (NInt j = 0; j < feat_size; j++) {
            if (i != j) {
                ScoreFlt dis = calc_L2_distance(feat_matrix[i], feat_matrix[j]);
                if (dis_min > dis) {
                    dis_min = dis;
                    idx_min = j;
                }
            }
        }
        // printf("id: %4d, dis: %.8f, tar: %4d\n", i, dis_min, idx_min);
        if (idx_min < 0 || idx_min > feat_size) continue;
        min_counter[idx_min].id = idx_min;
        // Add weight for counter, distance closer to 1 is smaller weight
        min_counter[idx_min].score += fabsf(1 - dis_min);
    }

    // Decide base features with most closest features
    std::sort(min_counter.begin(), min_counter.end(), IdScorePairComparatorGreater<ScoreFlt>());
    for (NInt i = 0; i < base_size; i++) {
        // printf("Base %4d: dis: %.8f, tar: %4d\n", i, min_counter[i].score, min_counter[i].id);
        base_idx[i] = min_counter[i].id;
    }
    return base_idx;
}

inline FltVector reuse_weight(FltVector &feat_vector, const MyMatrix &base_matrix) {
    FltVector base_weight(base_matrix.nrows(), 0.0);
    for (ScoreFlt delta = 1; delta <= 16; delta *= 2) {
        ScoreFlt dis_min = base_matrix.nrows();
        NInt idx_min = 0;
        for (NInt j = 0; j < base_matrix.nrows(); j++) {
            ScoreFlt dis = calc_L2_distance(feat_vector, base_matrix[j]);
            if (dis_min > dis) {
                dis_min = dis;
                idx_min = j;
            }
        }
        ScoreFlt theta = calc_L2_residue(feat_vector, base_matrix[idx_min], 1.0);
        if (fabs(theta) / delta < 1 / 16) break;
        base_weight[idx_min] += theta;
    }
    return base_weight;
}

// ==================== Decomposition functions
inline IntVector sample_nodes(const IntVector Vt_nodes, NInt Vs_num) {
    // IntVector Vs_nodes(Vt_nodes);
    // NInt Vs_num_real(std::min(Vs_num, (NInt)Vt_nodes.size()));
    // std::shuffle(Vs_nodes.begin(), Vs_nodes.end(), std::mt19937(param.seed));
    // Vs_nodes.resize(Vs_num_real);
    IntVector Vs_nodes(9320);
    std::iota(Vs_nodes.begin(), Vs_nodes.end(), 0);
    return Vs_nodes;
}

inline IntVector select_pc(ScoreMatrix &feat_Matrix, MyMatrix &theta_matrix,
                           const NInt base_size, const ScoreFlt eps) {
    NInt feat_size = feat_Matrix.rows();
    NInt V_num     = feat_Matrix.cols();
    assert(feat_size < V_num && "ERROR: Feature size should be smaller than sampled vertex number");
    feat_Matrix.transposeInPlace();                                 // feat_Matrix: Vs_num * feat_size

    // Select base features (columns) by minimum residue
    RobustPca Rpca(feat_Matrix, eps);
    Rpca.fit(base_size);
    ScoreVector feat_Res_ = Rpca.SparseComponent().colwise().norm();
    FltVector feat_res(feat_Res_.data(), feat_Res_.data() + feat_Res_.size());
    IntVector base_idx = arg_kmax(feat_res, base_size);
    ScoreMatrix base_Matrix = feat_Matrix(Eigen::all, base_idx);    // base_Matrix: Vs_num * base_size

    // Fit theta matrix
    ScoreMatrix theta_Matrix_ = Rpca.fit_fixed(base_Matrix);
    theta_matrix.from_Eigen(theta_Matrix_);                         // theta_Matrix_: base_size * feat_size
    cout<<"res: \n"<<Rpca.SparseComponent().topLeftCorner(3, 5)<<endl;
    ScoreMatrix diff = feat_Matrix - base_Matrix * theta_Matrix_;
    cout<< "diff Fro norm: "<<diff.norm() <<" Abs norm: "<<diff.lpNorm<1>() << endl;
    cout<<"theta: "<<theta_Matrix_.rows()<<" "<<theta_Matrix_.cols()<<"\n"<<theta_Matrix_.topLeftCorner(3, 5)<<endl;
    return base_idx;
}

#endif  // SCARA_FEATUREOP_H
