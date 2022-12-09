#ifndef SCARA_BATCHRANDOMWALK_H
#define SCARA_BATCHRANDOMWALK_H

#ifdef ENABLE_RW

#include <vector>
#include <numeric>
#include <stack>
#include <queue>
#include "BasicDefinition.h"
#include "Graph.h"
#include "MyType.h"
#include "fastPRNG.h"


class XoshiroGenerator {
private:
    fastPRNG::fastXS64 generator;
public:
    void initialize(uint64_t seed) {
       generator.seed(seed);
    }

    inline double uniform_real() {
        return generator.xorShift_UNI<double>();
    }

    inline uint32_t uniform_int(const uint32_t &_end) {
        return generator.xorShift() % _end;
    }

    inline bool bias_coin_is_head(const double &_prob) {
        return generator.xorShift_UNI<double>() <= _prob;
    }

    inline bool bias_coin_is_tail(const double &_prob) {
        return generator.xorShift_UNI<double>() > _prob;
    }
};

extern XoshiroGenerator fRNG;

extern XoshiroGenerator init_rng(uint64_t seed);


template<class FLT>
class Alias {
private:
    const unsigned int size;
    const IntVector &first;
    IntVector second;
    std::vector<FLT> probability;
public:
    Alias(const IntVector &_active_ids, const std::vector<FLT> &_active_residuals) :
            size(_active_ids.size()),
            first(_active_ids),
            second(_active_ids.size(), 0),
            probability(_active_residuals) {
        const FLT sum = std::accumulate(_active_residuals.begin(), _active_residuals.end(), 0.0);
        std::stack<NInt, std::vector<NInt >> small;
        std::stack<NInt, std::vector<NInt >> big;
        const FLT size_over_sum = size / sum;
        for (NInt id = 0; id < size; ++id) {
            probability[id] *= size_over_sum;
            if (probability[id] > 1) {
                big.push(id);
            } else {
                small.push(id);
            }
        }
        while (!small.empty() && !big.empty()) {
            const NInt small_id = small.top();
            small.pop();
            const NInt big_id = big.top();
            second[small_id] = first[big_id];
            probability[big_id] -= (1 - probability[small_id]);
            if (probability[big_id] < 1) {
                small.push(big_id);
                big.pop();
            }
        }
    }

    inline NInt generate_random_id() const {
        const unsigned int bucket_id = fRNG.uniform_int(size);
        return fRNG.bias_coin_is_head(probability[bucket_id]) ? first[bucket_id] : second[bucket_id];
    }

};


class WalkCache {
private:
    const Graph &graph;
    IntVector walks;
    IntVector start_indices;

public:

    explicit WalkCache(const Graph &_graph) :
            graph(_graph),
            walks(_graph.getNumOfEdges() + _graph.get_num_dead_end(), 0),
            start_indices(_graph.getNumOfVertices(), 0) {
    }

    void generate() {
        double time_start = getCurrentTime();
        const NInt num_vertices = graph.getNumOfVertices();
        for (NInt sid = 0, index = 0; sid < num_vertices; ++sid) {
            // if (sid % 500000 == 0) { cout << sid << " vertices processed.\n"; }
            start_indices[sid] = index;
            const NInt &sid_idx_start = graph.get_neighbor_list_start_pos(sid);
            const NInt &sid_idx_end = graph.get_neighbor_list_start_pos(sid + 1);
            const NInt sid_degree = sid_idx_end - sid_idx_start;
            for (uint32_t j = 0; j < sid_degree; ++j) {
                const NInt sid_shift = fRNG.uniform_int(sid_degree);
                NInt current_id = graph.getOutNeighbor(sid_idx_start + sid_shift);
                while (fRNG.bias_coin_is_tail(graph.get_alpha())) {
                    // TODO: stop at L-hop
                    const NInt &idx_start = graph.get_neighbor_list_start_pos(current_id);
                    const NInt &idx_end = graph.get_neighbor_list_start_pos(current_id + 1);
                    const NInt degree = idx_end - idx_start;
                    const NInt shift = fRNG.uniform_int(degree);
                    const NInt nid = graph.getOutNeighbor(idx_start + shift);
                    current_id = nid;
                }
                walks[index++] = current_id;
            }
        }
        printf("Walk Cache Time: %.6f\n", getCurrentTime() - time_start);
    }

    void save(const std::string &_filename) const {
        const auto start = getCurrentTime();
        if (std::FILE *f = std::fopen(_filename.c_str(), "wb")) {
            std::fwrite(walks.data(), sizeof walks[0], walks.size(), f);
            std::fclose(f);
        } else {
            printf("WalkCache::save; File Not Exists.\n");
        }
        const auto end = getCurrentTime();
        // printf("Time Used For Saving : %.2f\n", end - start);
    }

    void load(const std::string &_filename) {
        const auto start = getCurrentTime();
        walks.clear();
        walks.resize(graph.getNumOfEdges() + graph.get_num_dead_end(), 0);
        assert(walks.size() == graph.get_neighbor_list_start_pos(graph.getNumOfVertices()));
        if (std::FILE *f = std::fopen(_filename.c_str(), "rb")) {
            MSG(walks.size())
            size_t rtn = std::fread(walks.data(), sizeof walks[0], walks.size(), f);
            // printf("Returned Value of fread: %zu\n", rtn);
            std::fclose(f);
            start_indices.clear();
            start_indices.resize(graph.getNumOfVertices(), 0);
            for (NInt prev_id = 0, id = 1; id < graph.getNumOfVertices(); ++prev_id, ++id) {
                start_indices[id] =
                        start_indices[prev_id]
                        + std::max((NInt) 1u, graph.original_out_degree(prev_id));
            }
        } else {
            printf("WalkCache::load; File Not Exists.\n");
            exit(1);
        }
        const auto end = getCurrentTime();
        // printf("Time Used For Loading Cache : %.2f\n", end - start);
    }

    void show() {
        show_vector("The cache walk vector", walks);
    }

    inline const NInt &get_zero_hop_start_index(const NInt &_vid) const {
        // assert(_vid < graph.getNumOfVertices());
        return start_indices[_vid];
    }

    inline NInt get_one_hop_start_index(const NInt &_vid) const {
        // assert(_vid < graph.getNumOfVertices());
        return start_indices[_vid];
    }

    inline const NInt &get_walk(const NInt &_index) const {
        assert(_index < walks.size());
        return walks[_index];
    }
};

#endif
#endif //SCARA_BATCHRANDOMWALK_H
