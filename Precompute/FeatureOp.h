
#ifndef SPEEDPPR_FEATURE_H
#define SPEEDPPR_FEATURE_H

#include <vector>
#include <iostream>
#include <cmath>
#include "BasicDefinition.h"
#include "MyQueue.h"

// ==================== Vector measurement
template <class FLOAT_TYPE>
inline FLOAT_TYPE
calc_L1_residue(std::vector<FLOAT_TYPE> &V1, std::vector<FLOAT_TYPE> &V2, float pace = 1.0) {
    VertexIdType index;
    FLOAT_TYPE used_sum = 0;
    FLOAT_TYPE theta;
    std::vector<std::pair<VertexIdType, FLOAT_TYPE>> theta_counter;
    for (VertexIdType i = 0; i < V1.size(); i++) {
        if (V2[i] != 0) {
            theta = V1[i] / V2[i];
            theta_counter.push_back(std::make_pair(i, theta));
        }
    }
    std::sort(theta_counter.begin(), theta_counter.end(),
              [](std::pair<FLOAT_TYPE, VertexIdType> a, std::pair<FLOAT_TYPE, VertexIdType> b) {
                  return fabs(a.second) < fabs(b.second);
              });
    for (VertexIdType i = 0; i < theta_counter.size(); i += 1) {
        index = theta_counter[i].first;
        theta = theta_counter[i].second;
        used_sum += fabs(V2[index]);
        if (used_sum > 0.5 || index == -1)
            break;
    }

    FLOAT_TYPE orig_sum = 0;
    FLOAT_TYPE diff_sum = 0;
    for (VertexIdType i = 0; i < V1.size(); i++) {
        orig_sum += fabs(V1[i]);
        diff_sum += fabs(V1[i] - theta * V2[i] * pace);
    }
    if (diff_sum > orig_sum)
        return 0;

    for (VertexIdType i = 0; i < V1.size(); i++) {
        V1[i] = V1[i] - theta * V2[i] * pace;
    }
    // printf("theta: %.6f, residue: %.6f\n", theta, diff_sum);
    return theta * pace;
}

template <class FLOAT_TYPE>
inline FLOAT_TYPE
calc_L2_residue(std::vector<FLOAT_TYPE> &V1, std::vector<FLOAT_TYPE> &V2, float pace = 1.0) {
    FLOAT_TYPE prd = 0;
    FLOAT_TYPE sum2 = 0;
    for (VertexIdType i = 0; i < V1.size(); i++) {
        prd += V1[i] * V2[i];
        sum2 += V2[i] * V2[i];
    }
    FLOAT_TYPE theta = prd / sum2;

    FLOAT_TYPE orig_sum = 0;
    FLOAT_TYPE diff_sum = 0;
    for (VertexIdType i = 0; i < V1.size(); i++) {
        orig_sum += fabs(V1[i]);
        diff_sum += fabs(V1[i] - theta * V2[i] * pace);
    }
    if (diff_sum > orig_sum)
        return 0;

    for (VertexIdType i = 0; i < V1.size(); i++) {
        V1[i] = V1[i] - theta * V2[i] * pace;
    }
    // printf("theta: %.6f, residue: %.6f\n", theta, diff_sum);
    return theta * pace;
}

template <class FLOAT_TYPE>
inline FLOAT_TYPE
calc_L1_distance(std::vector<FLOAT_TYPE> &V1, std::vector<FLOAT_TYPE> &V2) {
    /* Ranges:
        V~[0, 1] -> distance~[0, 1]
        V~[-1, 1] -> distance~[0, 2]
    */
    FLOAT_TYPE distance = 0;
    for (VertexIdType i = 0; i < V1.size(); i++) {
        distance += fabs(V1[i] - V2[i]);
    }
    return distance;
}

template <class FLOAT_TYPE>
inline FLOAT_TYPE
calc_L2_distance(std::vector<FLOAT_TYPE> &V1, std::vector<FLOAT_TYPE> &V2) {
    /* Ranges: (cosine angle = prd / (sqrt(sum1) * sqrt(sum2)))
        V~[0, 1] -> distance~[0, 1]
        V~[-1, 1] -> distance~[0, 2]
    */
    // TODO: cache for feature-feature product and difference
    FLOAT_TYPE distance = 0;
    FLOAT_TYPE prd = 0;
    FLOAT_TYPE sum1 = 0;
    FLOAT_TYPE sum2 = 0;
    for (VertexIdType i = 0; i < V1.size(); i++) {
        prd += V1[i] * V2[i];
        sum1 += V1[i] * V1[i];
        sum2 += V2[i] * V2[i];
    }
    // distance = 1 - prd / (sqrt(sum1) * sqrt(sum2));
    distance = 1 - fabs(prd / (sqrt(sum1) * sqrt(sum2)));
    return distance;
}

// ==================== Reuse functions
inline std::vector<VertexIdType>
select_base(MyMatrix &feature_matrix, MyMatrix &base_matrix, size_t base_size) {
    std::vector<VertexIdType> base_nodes(base_size, 0);                                                                // list of base nodes
    std::vector<std::pair<VertexIdType, PageRankScoreType>> min_counter(feature_matrix.size(), std::make_pair(0, 0));  // (min norm base, min norm) for each feature
    // Find minimum distance feature for each feature
    for (VertexIdType i = 0; i < feature_matrix.size(); i++) {
        PageRankScoreType dis_min = 4.0 * feature_matrix.size();
        VertexIdType idx_min = -1;
        for (VertexIdType j = 0; j < feature_matrix.size(); j++) {
            if (i != j) {
                PageRankScoreType dis = calc_L2_distance(feature_matrix[i], feature_matrix[j]);
                if (dis_min > dis) {
                    dis_min = dis;
                    idx_min = j;
                }
            }
        }
        // printf("id: %4d, dis: %.8f, tar: %4d\n", i, dis_min, idx_min);
        if (idx_min < 0 || idx_min > feature_matrix.size()) continue;
        min_counter[idx_min].first = idx_min;
        // Add weight for counter, distance closer to 1 is smaller weight
        min_counter[idx_min].second += fabs(1 - dis_min);
    }

    // Decide base features with most closest features
    std::sort(min_counter.begin(), min_counter.end(),
              [](std::pair<VertexIdType, PageRankScoreType> a1, std::pair<VertexIdType, PageRankScoreType> a2) {
                  return a1.second > a2.second;
              });
    for (VertexIdType i = 0; i < base_size; i++) {
        // printf("Base %4d: dis: %.8f, tar: %4d\n", i, min_counter[i].second, min_counter[i].first);
        base_matrix.set_row(i, feature_matrix[min_counter[i].first]);
        // base_matrix[i].swap(feature_matrix[min_counter[i].first]);
        base_nodes[i] = min_counter[i].first;
    }
    return base_nodes;
}

template <class FLOAT_TYPE>
inline std::vector<FLOAT_TYPE>
reuse_weight(std::vector<FLOAT_TYPE> &raw_seed, MyMatrix &base_matrix) {
    std::vector<FLOAT_TYPE> base_weight(base_matrix.size(), 0.0);
    for (FLOAT_TYPE delta = 1; delta <= 16; delta *= 2) {
        FLOAT_TYPE dis_min = base_matrix.size();
        VertexIdType idx_min = 0;
        for (VertexIdType j = 0; j < base_matrix.size(); j++) {
            FLOAT_TYPE dis = calc_L2_distance(raw_seed, base_matrix[j]);
            if (dis_min > dis) {
                dis_min = dis;
                idx_min = j;
            }
        }
        FLOAT_TYPE theta = calc_L2_residue(raw_seed, base_matrix[idx_min], 1.0);
        if (fabs(theta) / delta < 1 / 16) break;
        base_weight[idx_min] += theta;
    }
    return base_weight;
}

#endif  // SPEEDPPR_FEATURE_H
