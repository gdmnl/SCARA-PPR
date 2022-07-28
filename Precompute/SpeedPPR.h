

#ifndef SPEEDPPR_SPEEDPPR_H
#define SPEEDPPR_SPEEDPPR_H

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
#include "MyQueue.h"
#include "BatchRandomWalk.h"
#include "fast_double_parser.h"


class SpeedPPR {

public:

    template<class FLOAT_TYPE>
    struct WHOLE_GRAPH_STRUCTURE {
        std::vector<FLOAT_TYPE> means;

        WHOLE_GRAPH_STRUCTURE(const VertexIdType &_numOfVertices) :
                means(_numOfVertices + 2, 0),
                active_vertices(_numOfVertices + 2),
                is_active(_numOfVertices + 2, false),
                pi(_numOfVertices + 2, 0),
                residuals(_numOfVertices + 2, 0) {
        }

    protected:
        MyQueue active_vertices;
        std::vector<bool> is_active;
        std::vector<FLOAT_TYPE> pi;
        std::vector<FLOAT_TYPE> residuals;

        std::vector<VertexIdType> active_ids;
        std::vector<FLOAT_TYPE> active_residuals;
        std::vector<VertexIdType> current_vertices;

        friend class SpeedPPR;
    };


private:

    uint32_t num_of_residual_updates_per_second;
    uint32_t num_of_walks_per_second;
    const VertexIdType numOfVertices;
    const double d_log_numOfVertices;
    Graph &graph;

public:

    void get_random_walk_speed() {
        // we need to call graph.reset_set_dummy_neighbor(); before return
        graph.set_dummy_neighbor(graph.get_dummy_id());
        std::vector<VertexIdType> active_ids;
        std::vector<float> active_residuals;
        for (VertexIdType sid = 0; sid < numOfVertices; ++sid) {
            const VertexIdType &sidx_start = graph.get_neighbor_list_start_pos(sid);
            if (graph.original_out_degree(sid) > 0) {
                active_ids.emplace_back(sid);
                active_residuals.emplace_back(graph.original_out_degree(sid));
            }
        }
        const uint32_t num_of_walks = 10'000'000;
        std::vector<VertexIdType> current_vertices;
        std::vector<float> means(numOfVertices + 1, 0);
        double time_start = getCurrentTime();
        Alias<float> alias(active_ids, active_residuals);
        for (uint32_t i = 0; i < num_of_walks; ++i) {
            current_vertices.emplace_back(alias.generate_random_id());
        }
        for (auto &id : current_vertices) {
            const VertexIdType &idx_start = graph.get_neighbor_list_start_pos(id);
            const VertexIdType &idx_end = graph.get_neighbor_list_start_pos(id + 1);
            const VertexIdType degree = idx_end - idx_start;
            // Generate a uniform shift from 0 to degree - 1
            const VertexIdType shift = MinimalStandardGenerator::uniform_int(degree);
            id = graph.getOutNeighbor(idx_start + shift);
        }
        for (uint32_t j = 0; j < current_vertices.size(); ++j) {
            VertexIdType current_id = current_vertices[j];
            if (MinimalStandardGenerator::bias_coin_is_head(param.alpha)) {
                means[current_id] += 1;
            } else {
                const VertexIdType &current_idx_start = graph.get_neighbor_list_start_pos(current_id);
                const VertexIdType &current_idx_end = graph.get_neighbor_list_start_pos(current_id + 1);
                const VertexIdType current_degree = current_idx_end - current_idx_start;
                const VertexIdType current_shift = MinimalStandardGenerator::uniform_int(current_degree);
                current_id = graph.getOutNeighbor(current_idx_start + current_shift);
                current_vertices.push_back(current_id);
            }
        }
        double time_end = getCurrentTime();
        num_of_walks_per_second = num_of_walks / (time_end - time_start);
        MSG(num_of_walks_per_second)
        graph.reset_set_dummy_neighbor();
    }

    explicit SpeedPPR(Graph &_graph) :
            numOfVertices(_graph.getNumOfVertices()),
            d_log_numOfVertices(log(_graph.getNumOfVertices())),
            graph(_graph) {
        get_random_walk_speed();
    }

public:


    template<class FLOAT_TYPE>
    void compute_approximate_page_rank_3(
            WHOLE_GRAPH_STRUCTURE<FLOAT_TYPE> &_whole_graph_structure,
            const VertexIdType &_sid, const FLOAT_TYPE _epsilon,
            const FLOAT_TYPE _alpha, const FLOAT_TYPE _lower_threshold,
            const WalkCache &_walk_cache) {
        long long number_of_pushes = 0;
        const auto avg_deg = static_cast<FLOAT_TYPE>(graph.getNumOfEdges() / (double) graph.getNumOfVertices());
//        printf("Average Degree:%.12f\n", avg_deg);
        FLOAT_TYPE num_walks = ceil((2 + (2.0 / 3.0) * _epsilon) * d_log_numOfVertices /
                                    (_epsilon * _epsilon * _lower_threshold));
        auto &active_vertices = _whole_graph_structure.active_vertices;
        auto &is_active = _whole_graph_structure.is_active;
        auto &pi = _whole_graph_structure.pi;
        auto &residuals = _whole_graph_structure.residuals;
        auto &means = _whole_graph_structure.means;
        std::fill(pi.begin(), pi.end(), 0);
        std::fill(residuals.begin(), residuals.end(), 0);
        active_vertices.push(_sid);
        is_active[_sid] = true;
        residuals[_sid] = num_walks;
        uint32_t num_active = 0;
        const FLOAT_TYPE one_minus_alpha = 1.0 - _alpha;
        const VertexIdType queue_threshold = (numOfVertices / avg_deg * 4);
        const uint32_t initial_size = std::max(num_walks / (1000 * d_log_numOfVertices), 1.0);
        const uint32_t step_size = std::max(powf(initial_size, 1.0 / 3.0), 2.0f);
        for (uint32_t scale_factor = initial_size;
             scale_factor >= 1 && active_vertices.size() < queue_threshold;) {
            const FLOAT_TYPE scale_factor_over_one_minus_alpha = scale_factor / one_minus_alpha;
            while (!active_vertices.empty() && active_vertices.size() < queue_threshold) {
                const VertexIdType id = active_vertices.front();
                active_vertices.pop();
                is_active[id] = false;
                const FLOAT_TYPE residual = residuals[id];
                const VertexIdType &idx_start = graph.get_neighbor_list_start_pos(id);
                const VertexIdType &idx_end = graph.get_neighbor_list_start_pos(id + 1);
                const FLOAT_TYPE degree_f = idx_end - idx_start;
                const FLOAT_TYPE one_minus_alpha_residual = one_minus_alpha * residual;
                if (one_minus_alpha_residual >= degree_f * scale_factor) {
                    const FLOAT_TYPE alpha_residual = residual - one_minus_alpha_residual;
                    pi[id] += alpha_residual;
                    residuals[id] = 0;
                    const FLOAT_TYPE increment = one_minus_alpha_residual / degree_f;
                    for (uint32_t j = idx_start; j < idx_end; ++j) {
                        const VertexIdType &nid = graph.getOutNeighbor(j);
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
                for (VertexIdType id = 0; id < numOfVertices; ++id) {
                    if (one_minus_alpha * residuals[id] >= scale_factor) {
                        active_vertices.push(id);
                        is_active[id] = true;
                    }
                }
            }
        }

        num_active = active_vertices.size();
        const FLOAT_TYPE one_over_one_minus_alpha = 1.0 / one_minus_alpha;
        for (; num_active > queue_threshold;) {
            num_active = 0;
            for (VertexIdType id = 0, next_id = 1, degree, idx_start, idx_end = graph.get_neighbor_list_start_pos(id);
                 id < numOfVertices; ++id, ++next_id) {
                idx_start = idx_end;
                idx_end = graph.get_neighbor_list_start_pos(next_id);
                degree = idx_end - idx_start;
                const FLOAT_TYPE &residual = residuals[id];
                const FLOAT_TYPE one_minus_alpha_residual = one_minus_alpha * residual;
                if (one_minus_alpha_residual >= degree) {
                    const FLOAT_TYPE alpha_residual = residual - one_minus_alpha_residual;
                    pi[id] += alpha_residual;
                    residuals[id] = 0;
                    const FLOAT_TYPE increment = one_minus_alpha_residual / degree;
                    num_active += degree;
                    for (uint32_t j = idx_start; j < idx_end; ++j) {
                        const VertexIdType &nid = graph.getOutNeighbor(j);
                        residuals[nid] += increment;
                    }
                }
            }
        }

        num_active = 0;
        active_vertices.clear();
        std::fill(is_active.begin(), is_active.end(), false);
        for (VertexIdType id = 0; id < numOfVertices; ++id) {
            if (residuals[id] >= one_over_one_minus_alpha) {
                active_vertices.push(id);
                is_active[id] = true;
            }
        }
        while (!active_vertices.empty()) {
            const VertexIdType id = active_vertices.front();
            active_vertices.pop();
            is_active[id] = false;
            const FLOAT_TYPE &residual = residuals[id];
            const VertexIdType &idx_start = graph.get_neighbor_list_start_pos(id);
            const VertexIdType &idx_end = graph.get_neighbor_list_start_pos(id + 1);
            const auto degree_f = static_cast<FLOAT_TYPE>(idx_end - idx_start);
            const FLOAT_TYPE one_minus_alpha_residual = one_minus_alpha * residual;
            if (one_minus_alpha_residual >= degree_f && degree_f) {
                const FLOAT_TYPE alpha_residual = residual - one_minus_alpha_residual;
                pi[id] += alpha_residual;
                residuals[id] = 0;
                const FLOAT_TYPE increment = one_minus_alpha_residual / degree_f;
                for (uint32_t j = idx_start; j < idx_end; ++j) {
                    const VertexIdType &nid = graph.getOutNeighbor(j);
                    residuals[nid] += increment;
                    if (!is_active[nid]) {
                        active_vertices.push(nid);
                        is_active[nid] = true;
                    }
                }
            }
        }

        means.swap(pi);
        for (VertexIdType id = 0; id < numOfVertices; ++id) {
            FLOAT_TYPE &residual = residuals[id];
            if (residual > 0) {
                const FLOAT_TYPE alpha_residual = _alpha * residuals[id];
                means[id] += alpha_residual;
                residuals[id] -= alpha_residual;
                VertexIdType idx_one_hop = _walk_cache.get_one_hop_start_index(id);
                const FLOAT_TYPE num_one_hop_walks = std::ceil(residual);
                const FLOAT_TYPE correction_factor = residual / num_one_hop_walks;
                const uint32_t end_one_hop = idx_one_hop + num_one_hop_walks;

                for (; idx_one_hop < end_one_hop; ++idx_one_hop) {
                    means[_walk_cache.get_walk(idx_one_hop)] += correction_factor;
                }
            }
        }

        // compute bounds
        const FLOAT_TYPE one_over_num_walks = (1.0f / num_walks);
        const auto scale_factor = static_cast<FLOAT_TYPE>(1.0 / (1.0 - residuals[numOfVertices] * one_over_num_walks
                                                                 - means[numOfVertices] * one_over_num_walks));
        const auto one_over_num_walks_x_scale_factor = one_over_num_walks * scale_factor;
        for (auto &mean :means) {
            mean *= one_over_num_walks_x_scale_factor;
        }
        means[numOfVertices] = 0;
    }

    template<class FLOAT_TYPE>
    void compute_approximate_page_rank_3(
                WHOLE_GRAPH_STRUCTURE<FLOAT_TYPE> &_whole_graph_structure,
                const std::vector<double> &_seeds, const FLOAT_TYPE _epsilon,
                const FLOAT_TYPE _alpha, const FLOAT_TYPE _lower_threshold,
                const WalkCache &_walk_cache, const double gamma = 1.0) {
        long long number_of_pushes = 0;
        const auto avg_deg = static_cast<FLOAT_TYPE>(graph.getNumOfEdges() / (double) graph.getNumOfVertices());
        FLOAT_TYPE num_walks = ceil( (2 + (2.0 / 3.0) * _epsilon) * d_log_numOfVertices /
                                    (_epsilon * _epsilon * _lower_threshold) / gamma );
        auto &active_vertices = _whole_graph_structure.active_vertices;
        auto &is_active = _whole_graph_structure.is_active;
        auto &pi = _whole_graph_structure.pi;
        auto &residuals = _whole_graph_structure.residuals;
        auto &means = _whole_graph_structure.means;

        std::fill(pi.begin(), pi.end(), 0);
        std::fill(residuals.begin(), residuals.end(), 0.0);

        for(int i = 0; i < graph.getNumOfVertices(); i++){
            if(_seeds[i] != 0.0){
                active_vertices.push(i);
                is_active[i] = true;
                residuals[i] = _seeds[i] * num_walks;
            }
        }

        /*
        double rsum = 0.0;
        for(int i = 0; i < graph.getNumOfVertices(); i++){
            rsum += residuals[i];
        }
        std::cout<< "rsum: " << rsum << std::endl;
        */
        uint32_t num_active = 0;
        const FLOAT_TYPE one_minus_alpha = 1.0 - _alpha;
        const VertexIdType queue_threshold = (numOfVertices / avg_deg * 4);
        const uint32_t initial_size = std::max(num_walks / (1000 * d_log_numOfVertices), 1.0);
        const uint32_t step_size = std::max(powf(initial_size, 1.0 / 3.0), 2.0f);

        for (uint32_t scale_factor = initial_size;
             scale_factor >= 1 && active_vertices.size() < queue_threshold;) {
            const FLOAT_TYPE scale_factor_over_one_minus_alpha = scale_factor / one_minus_alpha;
            while (!active_vertices.empty() && active_vertices.size() < queue_threshold) {
                const VertexIdType id = active_vertices.front();
                active_vertices.pop();
                is_active[id] = false;
                const FLOAT_TYPE residual = residuals[id];
                const VertexIdType &idx_start = graph.get_neighbor_list_start_pos(id);
                const VertexIdType &idx_end = graph.get_neighbor_list_start_pos(id + 1);
                const FLOAT_TYPE degree_f = idx_end - idx_start;
                const FLOAT_TYPE one_minus_alpha_residual = one_minus_alpha * residual;

                if (abs(one_minus_alpha_residual) >= degree_f * scale_factor) {
                    const FLOAT_TYPE alpha_residual = residual - one_minus_alpha_residual;
                    pi[id] += alpha_residual;
                    residuals[id] = 0;
                    const FLOAT_TYPE increment = one_minus_alpha_residual / degree_f;
                    for (uint32_t j = idx_start; j < idx_end; ++j) {
                        const VertexIdType &nid = graph.getOutNeighbor(j);
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
                for (VertexIdType id = 0; id < numOfVertices; ++id) {
                    if (abs(one_minus_alpha * residuals[id]) >= scale_factor) {
                        active_vertices.push(id);
                        is_active[id] = true;
                    }
                }
            }
        }

        num_active = active_vertices.size();
        const FLOAT_TYPE one_over_one_minus_alpha = 1.0 / one_minus_alpha;

        num_active = 0;
        active_vertices.clear();
        std::fill(is_active.begin(), is_active.end(), false);
        for (VertexIdType id = 0; id < numOfVertices; ++id) {
            if (residuals[id] >= one_over_one_minus_alpha) {
                active_vertices.push(id);
                is_active[id] = true;
            }
        }
        while (!active_vertices.empty()) {
            const VertexIdType id = active_vertices.front();
            active_vertices.pop();
            is_active[id] = false;
            const FLOAT_TYPE &residual = residuals[id];
            const VertexIdType &idx_start = graph.get_neighbor_list_start_pos(id);
            const VertexIdType &idx_end = graph.get_neighbor_list_start_pos(id + 1);
            const auto degree_f = static_cast<FLOAT_TYPE>(idx_end - idx_start);
            const FLOAT_TYPE one_minus_alpha_residual = one_minus_alpha * residual;
            if (abs(one_minus_alpha_residual) >= degree_f && degree_f) {
                const FLOAT_TYPE alpha_residual = residual - one_minus_alpha_residual;
                pi[id] += alpha_residual;
                residuals[id] = 0;
                const FLOAT_TYPE increment = one_minus_alpha_residual / degree_f;
                for (uint32_t j = idx_start; j < idx_end; ++j) {
                    const VertexIdType &nid = graph.getOutNeighbor(j);
                    residuals[nid] += increment;
                    if (!is_active[nid]) {
                        active_vertices.push(nid);
                        is_active[nid] = true;
                    }
                }
            }
        }

        means.swap(pi);
        for (VertexIdType id = 0; id < numOfVertices; ++id) {
            FLOAT_TYPE &residual = residuals[id];
            if (residual != 0) {
                const FLOAT_TYPE alpha_residual = _alpha * residuals[id];
                means[id] += alpha_residual;
                residuals[id] -= alpha_residual;
                VertexIdType idx_one_hop = _walk_cache.get_one_hop_start_index(id);
                const FLOAT_TYPE num_one_hop_walks = std::ceil(abs(residual));
                const FLOAT_TYPE correction_factor = residual / num_one_hop_walks;
                const uint32_t end_one_hop = idx_one_hop + num_one_hop_walks;

                for (; idx_one_hop < end_one_hop; ++idx_one_hop) {
                    means[_walk_cache.get_walk(idx_one_hop)] += correction_factor;
                }
            }
        }

        // compute bounds
        const FLOAT_TYPE one_over_num_walks = (1.0f / num_walks);
        const auto scale_factor = static_cast<FLOAT_TYPE>(1.0 / (1.0 - residuals[numOfVertices] * one_over_num_walks
                                                                 - means[numOfVertices] * one_over_num_walks));
        const auto one_over_num_walks_x_scale_factor = one_over_num_walks * scale_factor;
        for (auto &mean :means) {
            mean *= one_over_num_walks_x_scale_factor;
        }
        means[numOfVertices] = 0;
    }

    template<class FLOAT_TYPE>
    void compute_approximate_page_rank_walk_on_the_fly(
            WHOLE_GRAPH_STRUCTURE<FLOAT_TYPE> &_whole_graph_structure,
            const VertexIdType &_sid, const FLOAT_TYPE _epsilon,
            const FLOAT_TYPE _alpha, const FLOAT_TYPE _lower_threshold) {
        long long number_of_pushes = 0;
        const auto avg_deg = static_cast<FLOAT_TYPE>(graph.getNumOfEdges() / (double) graph.getNumOfVertices());
        FLOAT_TYPE time_scaling_factor = 1.0;
        FLOAT_TYPE one_over_time_scaling_factor = 1.0 / time_scaling_factor;
        FLOAT_TYPE num_walks = one_over_time_scaling_factor * ceil((2 + (2.0 / 3.0) * _epsilon) * d_log_numOfVertices /
                                                                   (_epsilon * _epsilon * _lower_threshold));
        auto &active_vertices = _whole_graph_structure.active_vertices;
        auto &is_active = _whole_graph_structure.is_active;
        auto &pi = _whole_graph_structure.pi;
        auto &residuals = _whole_graph_structure.residuals;
        auto &means = _whole_graph_structure.means;

        std::fill(pi.begin(), pi.end(), 0);
        std::fill(residuals.begin(), residuals.end(), 0);
        active_vertices.push(_sid);
        is_active[_sid] = true;
        residuals[_sid] = num_walks;
        uint32_t num_active = 0;
        const FLOAT_TYPE one_minus_alpha = 1.0 - _alpha;
        const VertexIdType queue_threshold = (numOfVertices / avg_deg * 4);
        const uint32_t initial_size = std::max(num_walks / (1000 * d_log_numOfVertices), 1.0);
        const uint32_t step_size = std::max(powf(initial_size, 1.0 / 3.0), 2.0f);
        const double time_forward_start = getCurrentTime();
//        double time_end_prev_round = time_forward_start;
//        double time_end_this_round;
        for (bool more_pushes = true; more_pushes;) {
            const double time_round_start = getCurrentTime();
            for (uint32_t scale_factor = initial_size;
                 scale_factor >= 1 && active_vertices.size() < queue_threshold;) {
                const FLOAT_TYPE scale_factor_over_one_minus_alpha = scale_factor / one_minus_alpha;
                while (!active_vertices.empty() && active_vertices.size() < queue_threshold) {
                    const VertexIdType id = active_vertices.front();
                    active_vertices.pop();
                    is_active[id] = false;
                    const FLOAT_TYPE residual = residuals[id];
                    const VertexIdType &idx_start = graph.get_neighbor_list_start_pos(id);
                    const VertexIdType &idx_end = graph.get_neighbor_list_start_pos(id + 1);
                    const FLOAT_TYPE degree_f = idx_end - idx_start;
                    const FLOAT_TYPE one_minus_alpha_residual = one_minus_alpha * residual;

                    if (one_minus_alpha_residual >= degree_f * scale_factor) {
                        const FLOAT_TYPE alpha_residual = residual - one_minus_alpha_residual;
                        pi[id] += alpha_residual;
                        residuals[id] = 0;
                        const FLOAT_TYPE increment = one_minus_alpha_residual / degree_f;

                        for (uint32_t j = idx_start; j < idx_end; ++j) {
                            const VertexIdType &nid = graph.getOutNeighbor(j);
                            residuals[nid] += increment;
                            if (!is_active[nid]) {
                                active_vertices.push(nid);
                                is_active[nid] = true;
                            }
                        }
                    }
                }
                scale_factor /= step_size;
//            scale_factor >>= 1u;
                if (active_vertices.empty()) {
                    for (VertexIdType id = 0; id < numOfVertices; ++id) {
                        if (one_minus_alpha * residuals[id] >= scale_factor) {
                            active_vertices.push(id);
                            is_active[id] = true;
                        }
                    }
                }
            }

            num_active = active_vertices.size();
            const FLOAT_TYPE one_over_one_minus_alpha = 1.0 / one_minus_alpha;
            for (; num_active > queue_threshold;) {
                num_active = 0;
                for (VertexIdType id = 0, next_id = 1, degree, idx_start, idx_end = graph.get_neighbor_list_start_pos(
                        id);
                     id < numOfVertices; ++id, ++next_id) {
                    idx_start = idx_end;
                    idx_end = graph.get_neighbor_list_start_pos(next_id);
                    degree = idx_end - idx_start;
                    const FLOAT_TYPE &residual = residuals[id];
                    const FLOAT_TYPE one_minus_alpha_residual = one_minus_alpha * residual;
                    if (one_minus_alpha_residual >= degree) {
                        const FLOAT_TYPE alpha_residual = residual - one_minus_alpha_residual;
                        pi[id] += alpha_residual;
                        residuals[id] = 0;
                        const FLOAT_TYPE increment = one_minus_alpha_residual / degree;

                        num_active += degree;
                        for (uint32_t j = idx_start; j < idx_end; ++j) {
                            const VertexIdType &nid = graph.getOutNeighbor(j);
                            residuals[nid] += increment;
                        }
                    }
                }
            }

            num_active = 0;
            active_vertices.clear();
            std::fill(is_active.begin(), is_active.end(), false);
            for (VertexIdType id = 0; id < numOfVertices; ++id) {
                if (residuals[id] >= one_over_one_minus_alpha) {
                    active_vertices.push(id);
                    is_active[id] = true;
                }
            }
            while (!active_vertices.empty()) {
                const VertexIdType id = active_vertices.front();
                active_vertices.pop();
                is_active[id] = false;
                const FLOAT_TYPE &residual = residuals[id];
                const VertexIdType &idx_start = graph.get_neighbor_list_start_pos(id);
                const VertexIdType &idx_end = graph.get_neighbor_list_start_pos(id + 1);
                const auto degree_f = static_cast<FLOAT_TYPE>(idx_end - idx_start);
                const FLOAT_TYPE one_minus_alpha_residual = one_minus_alpha * residual;
                if (one_minus_alpha_residual >= degree_f && degree_f) {
                    const FLOAT_TYPE alpha_residual = residual - one_minus_alpha_residual;
                    pi[id] += alpha_residual;
                    residuals[id] = 0;
                    const FLOAT_TYPE increment = one_minus_alpha_residual / degree_f;
                    for (uint32_t j = idx_start; j < idx_end; ++j) {
                        const VertexIdType &nid = graph.getOutNeighbor(j);
                        residuals[nid] += increment;
                        if (!is_active[nid]) {
                            active_vertices.push(nid);
                            is_active[nid] = true;
                        }
                    }
                }
            }

            const double time_current = getCurrentTime();
            const double time_this_round = time_current - time_round_start;
            const double time_for_pushes = time_current - time_forward_start;
            const FLOAT_TYPE r_sum = std::accumulate(residuals.begin(), residuals.end(), 0.0) * one_minus_alpha;
//            const FLOAT_TYPE walk_speed_per_second = 1.0 / 1'500'000;
            const FLOAT_TYPE walk_speed_per_second = 1.0 / num_of_walks_per_second;
            const FLOAT_TYPE increment_factor = std::exp(1.0);

            const double estimate_time = r_sum * time_scaling_factor * walk_speed_per_second;
            if (estimate_time >= time_for_pushes ||
                (1.0 - 1.0 / increment_factor) * estimate_time >= 1.1 * time_this_round) {
                more_pushes = true;
                time_scaling_factor *= (1.0 / increment_factor);
                num_walks *= increment_factor;
                for (VertexIdType id = 0; id < numOfVertices; ++id) {
                    if (pi[id] || residuals[id]) {
                        pi[id] *= increment_factor;
                        residuals[id] *= increment_factor;
                        if (residuals[id] >= one_over_one_minus_alpha) {
                            active_vertices.push(id);
                            is_active[id] = true;
                        }
                    }
                }
            } else {
                more_pushes = false;
            }
        }

        uint32_t num_of_walks_performed = 0;
        // random walks
        do {
            means.swap(pi);
            double r_sum = 0;
            auto &active_ids = _whole_graph_structure.active_ids;
            auto &active_residuals = _whole_graph_structure.active_residuals;
            auto &current_vertices = _whole_graph_structure.current_vertices;
            active_ids.clear();
            active_residuals.clear();
            current_vertices.clear();
            one_over_time_scaling_factor = 1.0 / time_scaling_factor;
            for (VertexIdType id = 0; id < numOfVertices; ++id) {
                FLOAT_TYPE &residual = residuals[id];
                if (residual > 0) {
                    // do not change the order of the following operations
                    const FLOAT_TYPE alpha_residual = _alpha * residual;
                    means[id] += alpha_residual;
                    residuals[id] -= alpha_residual;
                    residual *= time_scaling_factor;
                    active_ids.push_back(id);
                    active_residuals.push_back(residual);
                    r_sum += residual;
                }
            }

            num_of_walks_performed += r_sum;
            Alias<FLOAT_TYPE> alias(active_ids, active_residuals);
            current_vertices.clear();
            for (uint32_t index = 0, size = r_sum; index < size; ++index) {
                current_vertices.push_back(alias.generate_random_id());
            }
            // replace the id with its neighbor
            for (auto &id : current_vertices) {
                const VertexIdType &idx_start = graph.get_neighbor_list_start_pos(id);
                const VertexIdType &idx_end = graph.get_neighbor_list_start_pos(id + 1);
                const VertexIdType degree = idx_end - idx_start;
                // Generate a uniform shift from 0 to degree - 1
                const VertexIdType shift = MinimalStandardGenerator::uniform_int(degree);
                id = graph.getOutNeighbor(idx_start + shift);
            }

            for (uint32_t j = 0; j < current_vertices.size(); ++j) {
                VertexIdType current_id = current_vertices[j];
                if (MinimalStandardGenerator::bias_coin_is_head(_alpha)) {
                    means[current_id] += one_over_time_scaling_factor;
                } else {
                    const VertexIdType &current_idx_start = graph.get_neighbor_list_start_pos(current_id);
                    const VertexIdType &current_idx_end = graph.get_neighbor_list_start_pos(current_id + 1);
                    const VertexIdType current_degree = current_idx_end - current_idx_start;
                    const VertexIdType current_shift = MinimalStandardGenerator::uniform_int(current_degree);
                    current_id = graph.getOutNeighbor(current_idx_start + current_shift);
                    current_vertices.push_back(current_id);
                }
            }
        } while (false);

        // compute bounds
        const FLOAT_TYPE one_over_num_walks = (1.0f / num_walks);
        const auto one_over_num_walks_x_scale_factor = one_over_num_walks;
        for (auto &mean :means) {
            mean *= one_over_num_walks_x_scale_factor;
        }
        means[numOfVertices] = 0;
    }

};


#endif //SPEEDPPR_SPEEDPPR_H
