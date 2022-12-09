// Ref: https://github.com/wuhao-wu-jiang/Personalized-PageRank
#ifndef SCARA_SPEEDPPR_H
#define SCARA_SPEEDPPR_H

#include <cmath>
#include <vector>
#include <cassert>
#include <cmath>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <sstream>
#include "BasicDefinition.h"
#include "Graph.h"
#include "MyType.h"
#ifdef ENABLE_RW
#include "BatchRandomWalk.h"
#endif

class SpeedPPR {

public:

    template<class FLT>
    struct GStruct {
        std::vector<FLT> means;

        GStruct(const NInt &_numOfVertices) :
                means(_numOfVertices + 2, 0),
                active_vertices(_numOfVertices + 2),
                is_active(_numOfVertices + 2, false),
                pi(_numOfVertices + 2, 0),
                residuals(_numOfVertices + 2, 0) {
        }

    protected:
        MyQueue active_vertices;
        std::vector<bool> is_active;
        std::vector<FLT> pi;
        std::vector<FLT> residuals;

        IntVector active_ids;
        std::vector<FLT> active_residuals;
        IntVector current_vertices;

        friend class SpeedPPR;
    };


private:

    uint32_t num_of_residual_updates_per_second;
    uint32_t num_of_walks_per_second;
    const NInt numOfVertices;
    const double d_log_numOfVertices;
    Graph &graph;

public:

#ifdef ENABLE_RW
    void get_random_walk_speed() {
        // we need to call graph.reset_set_dummy_neighbor(); before return
        graph.set_dummy_neighbor(graph.get_dummy_id());
        IntVector active_ids;
        FltVector active_residuals;
        for (NInt sid = 0; sid < numOfVertices; ++sid) {
            const NInt &sidx_start = graph.get_neighbor_list_start_pos(sid);
            if (graph.original_out_degree(sid) > 0) {
                active_ids.emplace_back(sid);
                active_residuals.emplace_back(graph.original_out_degree(sid));
            }
        }
        const uint32_t num_of_walks = 1'000'000;
        IntVector current_vertices;
        FltVector means(numOfVertices + 1, 0);
        double time_start = getCurrentTime();
        Alias<float> alias(active_ids, active_residuals);
        for (uint32_t i = 0; i < num_of_walks; ++i) {
            current_vertices.emplace_back(alias.generate_random_id());
        }
        for (auto &id : current_vertices) {
            const NInt &idx_start = graph.get_neighbor_list_start_pos(id);
            const NInt &idx_end = graph.get_neighbor_list_start_pos(id + 1);
            const NInt degree = idx_end - idx_start;
            // Generate a uniform shift from 0 to degree - 1
            const NInt shift = fRNG.uniform_int(degree);
            id = graph.getOutNeighbor(idx_start + shift);
        }
        for (uint32_t j = 0; j < current_vertices.size(); ++j) {
            NInt current_id = current_vertices[j];
            if (fRNG.bias_coin_is_head(param.alpha)) {
                means[current_id] += 1;
            } else {
                const NInt &current_idx_start = graph.get_neighbor_list_start_pos(current_id);
                const NInt &current_idx_end = graph.get_neighbor_list_start_pos(current_id + 1);
                const NInt current_degree = current_idx_end - current_idx_start;
                const NInt current_shift = fRNG.uniform_int(current_degree);
                current_id = graph.getOutNeighbor(current_idx_start + current_shift);
                current_vertices.push_back(current_id);
            }
        }
        double time_end = getCurrentTime();
        num_of_walks_per_second = num_of_walks / (time_end - time_start);
        MSG(num_of_walks_per_second)
        graph.reset_set_dummy_neighbor();
    }
#endif

    explicit SpeedPPR(Graph &_graph) :
            numOfVertices(_graph.getNumOfVertices()),
            d_log_numOfVertices(log(_graph.getNumOfVertices())),
            graph(_graph) {
#ifdef ENABLE_RW
        get_random_walk_speed();
#endif
    }

public:

#ifdef ENABLE_RW
    template<class FLT>
    void calc_ppr_cache(
                GStruct<FLT> &_gstruct,
                const std::vector<FLT> &_seeds, const FLT _epsilon,
                const FLT _alpha, const FLT _lower_threshold,
                const WalkCache &_walk_cache, const FLT gamma = 1.0) {
        long long number_of_pushes = 0;
        const auto avg_deg = static_cast<FLT>(graph.getNumOfEdges() / (FLT) graph.getNumOfVertices());
        FLT num_walks = ceil( (2 + (2.0 / 3.0) * _epsilon) * d_log_numOfVertices /
                                    (_epsilon * _epsilon * _lower_threshold) / gamma );
        auto &active_vertices = _gstruct.active_vertices;
        auto &is_active = _gstruct.is_active;
        auto &pi = _gstruct.pi;
        auto &residuals = _gstruct.residuals;
        auto &means = _gstruct.means;

        std::fill(pi.begin(), pi.end(), 0);
        std::fill(residuals.begin(), residuals.end(), 0.0);

        for(int i = 0; i < graph.getNumOfVertices(); i++){
            if(_seeds[i] != 0.0){
                active_vertices.push(i);
                is_active[i] = true;
                residuals[i] = _seeds[i] * num_walks;
            }
        }

        uint32_t num_active = 0;
        const FLT one_minus_alpha = 1.0 - _alpha;
        const NInt queue_threshold = (numOfVertices / avg_deg * 4);
        const uint32_t initial_size = std::max(num_walks / (1000 * d_log_numOfVertices), 1.0);
        const uint32_t step_size = std::max(powf(initial_size, 1.0 / 3.0), 2.0f);

        for (uint32_t scale_factor = initial_size;
             scale_factor >= 1 && active_vertices.size() < queue_threshold;) {
            const FLT scale_factor_over_one_minus_alpha = scale_factor / one_minus_alpha;
            while (!active_vertices.empty() && active_vertices.size() < queue_threshold) {
                const NInt id = active_vertices.front();
                active_vertices.pop();
                is_active[id] = false;
                const FLT residual = residuals[id];
                const NInt &idx_start = graph.get_neighbor_list_start_pos(id);
                const NInt &idx_end = graph.get_neighbor_list_start_pos(id + 1);
                const FLT degree_f = idx_end - idx_start;
                const FLT one_minus_alpha_residual = one_minus_alpha * residual;

                if (fabs(one_minus_alpha_residual) >= degree_f * scale_factor) {
                    const FLT alpha_residual = residual - one_minus_alpha_residual;
                    pi[id] += alpha_residual;
                    residuals[id] = 0;
                    const FLT increment = one_minus_alpha_residual / degree_f;

                    for (uint32_t j = idx_start; j < idx_end; ++j) {
                        const NInt &nid = graph.getOutNeighbor(j);
                        residuals[nid] += increment;
                        if (!is_active[nid]) {
                            active_vertices.push(nid);
                            is_active[nid] = true;
                        }
                    }
                }
            }
            scale_factor /= step_size;

            if (active_vertices.empty()) {
                for (NInt id = 0; id < numOfVertices; ++id) {
                    if (abs(one_minus_alpha * residuals[id]) >= scale_factor) {
                        active_vertices.push(id);
                        is_active[id] = true;
                    }
                }
            }
        }

        num_active = active_vertices.size();
        const FLT one_over_one_minus_alpha = 1.0 / one_minus_alpha;

        num_active = 0;
        active_vertices.clear();
        std::fill(is_active.begin(), is_active.end(), false);
        for (NInt id = 0; id < numOfVertices; ++id) {
            if (residuals[id] >= one_over_one_minus_alpha) {
                active_vertices.push(id);
                is_active[id] = true;
            }
        }
        while (!active_vertices.empty()) {
            const NInt id = active_vertices.front();
            active_vertices.pop();
            is_active[id] = false;
            const FLT &residual = residuals[id];
            const NInt &idx_start = graph.get_neighbor_list_start_pos(id);
            const NInt &idx_end = graph.get_neighbor_list_start_pos(id + 1);
            const auto degree_f = static_cast<FLT>(idx_end - idx_start);
            const FLT one_minus_alpha_residual = one_minus_alpha * residual;
            if (fabs(one_minus_alpha_residual) >= degree_f && degree_f) {
                const FLT alpha_residual = residual - one_minus_alpha_residual;
                pi[id] += alpha_residual;
                residuals[id] = 0;
                const FLT increment = one_minus_alpha_residual / degree_f;
                for (uint32_t j = idx_start; j < idx_end; ++j) {
                    const NInt &nid = graph.getOutNeighbor(j);
                    residuals[nid] += increment;
                    if (!is_active[nid]) {
                        active_vertices.push(nid);
                        is_active[nid] = true;
                    }
                }
            }
        }

        // random walks
        means.swap(pi);
        for (NInt id = 0; id < numOfVertices; ++id) {
            FLT &residual = residuals[id];
            if (residual != 0) {
                const FLT alpha_residual = _alpha * residuals[id];
                means[id] += alpha_residual;
                residuals[id] -= alpha_residual;

                NInt idx_one_hop = _walk_cache.get_one_hop_start_index(id);
                const FLT num_one_hop_walks = std::ceil(abs(residual));
                const FLT correction_factor = residual / num_one_hop_walks;
                const uint32_t end_one_hop = idx_one_hop + num_one_hop_walks;
                for (; idx_one_hop < end_one_hop; ++idx_one_hop) {
                    means[_walk_cache.get_walk(idx_one_hop)] += correction_factor;
                }
            }
        }

        // compute bounds
        const FLT one_over_num_walks = (1.0f / num_walks);
        const auto scale_factor = static_cast<FLT>(1.0 / (1.0 - residuals[numOfVertices] * one_over_num_walks
                                                                 - means[numOfVertices] * one_over_num_walks));
        const auto one_over_num_walks_x_scale_factor = one_over_num_walks * scale_factor;
        for (auto &mean :means) {
            mean *= one_over_num_walks_x_scale_factor;
        }
        means[numOfVertices] = 0;
    }
#endif

    template<class FLT>
    void calc_ppr_walk(
            GStruct<FLT> &_gstruct,
            const std::vector<FLT> &_seeds, const FLT _epsilon,
            const FLT _alpha, const FLT _lower_threshold,
            const FLT gamma = 1.0) {
        long long number_of_pushes = 0;
        const auto avg_deg = static_cast<FLT>(graph.getNumOfEdges() / (FLT) graph.getNumOfVertices());
        FLT time_scaling_factor = 1.0;
        FLT one_over_time_scaling_factor = 1.0 / time_scaling_factor;
        FLT num_walks = ceil( (2 + (2.0 / 3.0) * _epsilon) * d_log_numOfVertices /
                                    (_epsilon * _epsilon * _lower_threshold) / gamma );
        auto &active_vertices = _gstruct.active_vertices;
        auto &is_active = _gstruct.is_active;
        auto &pi = _gstruct.pi;
        auto &residuals = _gstruct.residuals;
        auto &means = _gstruct.means;

        std::fill(pi.begin(), pi.end(), 0);
        std::fill(residuals.begin(), residuals.end(), 0.0);

        for(int i = 0; i < graph.getNumOfVertices(); i++){
            if(_seeds[i] != 0.0){
                active_vertices.push(i);
                is_active[i] = true;
                residuals[i] = _seeds[i] * num_walks;
            }
        }

        uint32_t num_active = 0;
        const FLT one_minus_alpha = 1.0 - _alpha;
        const NInt queue_threshold = (numOfVertices / avg_deg * 4);
        const uint32_t initial_size = std::max(num_walks / (1000 * d_log_numOfVertices), 1.0);
        const uint32_t step_size = std::max(powf(initial_size, 1.0 / 3.0), 2.0f);

        for (uint32_t scale_factor = initial_size;
             scale_factor >= 1 && active_vertices.size() < queue_threshold;) {
            const FLT scale_factor_over_one_minus_alpha = scale_factor / one_minus_alpha;
            while (!active_vertices.empty() && active_vertices.size() < queue_threshold) {
                const NInt id = active_vertices.front();
                active_vertices.pop();
                is_active[id] = false;
                const FLT residual = residuals[id];
                const NInt &idx_start = graph.get_neighbor_list_start_pos(id);
                const NInt &idx_end = graph.get_neighbor_list_start_pos(id + 1);
                const FLT degree_f = idx_end - idx_start;
                const FLT one_minus_alpha_residual = one_minus_alpha * residual;

                if (fabs(one_minus_alpha_residual) >= degree_f * scale_factor) {
                    const FLT alpha_residual = residual - one_minus_alpha_residual;
                    pi[id] += alpha_residual;
                    residuals[id] = 0;
                    const FLT increment = one_minus_alpha_residual / degree_f;

                    for (uint32_t j = idx_start; j < idx_end; ++j) {
                        const NInt &nid = graph.getOutNeighbor(j);
                        residuals[nid] += increment;
                        if (!is_active[nid]) {
                            active_vertices.push(nid);
                            is_active[nid] = true;
                        }
                    }
                }
            }
            scale_factor /= step_size;

            if (active_vertices.empty()) {
                for (NInt id = 0; id < numOfVertices; ++id) {
                    if (abs(one_minus_alpha * residuals[id]) >= scale_factor) {
                        active_vertices.push(id);
                        is_active[id] = true;
                    }
                }
            }
        }

        num_active = active_vertices.size();
        const FLT one_over_one_minus_alpha = 1.0 / one_minus_alpha;

        num_active = 0;
        active_vertices.clear();
        std::fill(is_active.begin(), is_active.end(), false);
        for (NInt id = 0; id < numOfVertices; ++id) {
            if (residuals[id] >= one_over_one_minus_alpha) {
                active_vertices.push(id);
                is_active[id] = true;
            }
        }
        while (!active_vertices.empty()) {
            const NInt id = active_vertices.front();
            active_vertices.pop();
            is_active[id] = false;
            const FLT &residual = residuals[id];
            const NInt &idx_start = graph.get_neighbor_list_start_pos(id);
            const NInt &idx_end = graph.get_neighbor_list_start_pos(id + 1);
            const auto degree_f = static_cast<FLT>(idx_end - idx_start);
            const FLT one_minus_alpha_residual = one_minus_alpha * residual;
            if (fabs(one_minus_alpha_residual) >= degree_f && degree_f) {
                const FLT alpha_residual = residual - one_minus_alpha_residual;
                pi[id] += alpha_residual;
                residuals[id] = 0;
                const FLT increment = one_minus_alpha_residual / degree_f;
                for (uint32_t j = idx_start; j < idx_end; ++j) {
                    const NInt &nid = graph.getOutNeighbor(j);
                    residuals[nid] += increment;
                    if (!is_active[nid]) {
                        active_vertices.push(nid);
                        is_active[nid] = true;
                    }
                }
            }
        }

        // random walks
        means.swap(pi);

#ifdef ENABLE_RW
        uint32_t num_of_walks_performed = 0;
        FLT r_sum = 0;
        auto &active_ids = _gstruct.active_ids;
        auto &active_residuals = _gstruct.active_residuals;
        auto &current_vertices = _gstruct.current_vertices;
        active_ids.clear();
        active_residuals.clear();
        current_vertices.clear();
        one_over_time_scaling_factor = 1.0 / time_scaling_factor;
        for (NInt id = 0; id < numOfVertices; ++id) {
            FLT &residual = residuals[id];
            if (residual != 0) {
                // do not change the order of the following operations
                const FLT alpha_residual = _alpha * residual;
                means[id] += alpha_residual;
                residuals[id] -= alpha_residual;
                residual *= time_scaling_factor;
                active_ids.push_back(id);
                active_residuals.push_back(residual);
                r_sum += fabs(residual);
            }
        }

        num_of_walks_performed += r_sum;
        Alias<FLT> alias(active_ids, active_residuals);
        current_vertices.clear();
        for (uint32_t index = 0, size = r_sum; index < size; ++index) {
            current_vertices.push_back(alias.generate_random_id());
        }
        // replace the id with its neighbor
        for (auto &id : current_vertices) {
            const NInt &idx_start = graph.get_neighbor_list_start_pos(id);
            const NInt &idx_end = graph.get_neighbor_list_start_pos(id + 1);
            const NInt degree = idx_end - idx_start;
            // Generate a uniform shift from 0 to degree - 1
            const NInt shift = fRNG.uniform_int(degree);
            id = graph.getOutNeighbor(idx_start + shift);
        }

        for (uint32_t j = 0; j < current_vertices.size(); ++j) {
            NInt current_id = current_vertices[j];
            // TODO: stop at L-hop
            if (fRNG.bias_coin_is_head(_alpha)) {
                means[current_id] += one_over_time_scaling_factor;
            } else {
                const NInt &current_idx_start = graph.get_neighbor_list_start_pos(current_id);
                const NInt &current_idx_end = graph.get_neighbor_list_start_pos(current_id + 1);
                const NInt current_degree = current_idx_end - current_idx_start;
                const NInt current_shift = fRNG.uniform_int(current_degree);
                current_id = graph.getOutNeighbor(current_idx_start + current_shift);
                current_vertices.push_back(current_id);
            }
        }

        // compute bounds
        // cout << "Walks performed: " << num_of_walks_performed << " / " << num_walks << endl;
        const FLT one_over_num_walks = (1.0f / num_walks);
        const auto one_over_num_walks_x_scale_factor = one_over_num_walks;
        for (auto &mean :means) {
            mean *= one_over_num_walks_x_scale_factor;
        }
#else
        for (NInt id = 0; id < numOfVertices; ++id) {
            FLT &residual = residuals[id];
            if (residual != 0) {
                const FLT alpha_residual = _alpha * residuals[id];
                means[id] += alpha_residual;
                residuals[id] -= alpha_residual;
            }
        }

        // compute bounds
        const FLT one_over_num_walks = (1.0f / num_walks);
        const auto scale_factor = static_cast<FLT>(1.0 / (1.0 - residuals[numOfVertices] * one_over_num_walks
                                                                 - means[numOfVertices] * one_over_num_walks));
        const auto one_over_num_walks_x_scale_factor = one_over_num_walks * scale_factor;
        for (auto &mean :means) {
            mean *= one_over_num_walks_x_scale_factor;
        }
#endif
        means[numOfVertices] = 0;
    }
};

#endif //SCARA_SPEEDPPR_H
