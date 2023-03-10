/*
  Type and class definitions
  Author: nyLiao
*/
#ifndef SCARA_BASICDEFINITION_H
#define SCARA_BASICDEFINITION_H

// #ifndef ENABLE_RW
// #define ENABLE_RW
// #endif
// #ifndef ENABLE_PI
// #define ENABLE_PI
// #endif
// #ifndef ENABLE_INITTH
// #define ENABLE_INITTH
// #endif
// #ifndef DEBUG
// #define DEBUG
// #endif

#include <queue>
#include <vector>
#include <cmath>

using std::cout;
using std::endl;

#define IDFMT "lu"                  // NInt print format
typedef unsigned long NInt;         // Type of Node / Edge size
typedef float ScoreFlt;             // Type of PPR Score
typedef std::vector<NInt> IntVector;
typedef std::vector<ScoreFlt> FltVector;

template<class FLT>
struct IdScorePair {
    NInt id = 0;
    FLT score = 0;

    IdScorePair(const NInt &_id = 0, const FLT &_score = 0) :
            id(_id), score(_score) {}
};

template<class FLT>
struct IdScorePairComparatorGreater {
    // Compare 2 IdScorePair objects using name
    bool operator()(const IdScorePair<FLT> &pair1, const IdScorePair<FLT> &pair2) {
        return pair1.score > pair2.score || pair1.score == pair2.score && pair1.id < pair2.id;
    }
};

template<class FLT>
struct IdScorePairComparatorLess {
    // Compare 2 IdScorePair objects using name
    bool operator()(const IdScorePair<FLT> &pair1, const IdScorePair<FLT> &pair2) {
        return pair1.score < pair2.score || pair1.score == pair2.score && pair1.id < pair2.id;
    }
};

struct Edge {
    NInt from_id;
    NInt to_id;

    Edge() : from_id(0), to_id(0) {}

    Edge(const NInt &_from, const NInt &_to) :
            from_id(_from), to_id(_to) {}

    bool operator<(const Edge &_edge) const {
        return from_id < _edge.from_id || (from_id == _edge.from_id && to_id < _edge.to_id);
    }
};

class VertexQueue {
private:
    const NInt mask;
    IntVector queue;
    NInt num = 0;
    NInt idx_front = 0;
    NInt idx_last_plus_one = 0;
private:
    static inline NInt compute_queue_size(const NInt &_numOfVertices) {
        return (1u) << (uint32_t) ceil(log2(_numOfVertices + 2u));
    }

public:
    explicit VertexQueue(const NInt &_numOfVertices) :
            mask(compute_queue_size(_numOfVertices) - 1),
            queue(mask + 2u, 0) {}

    inline void clear() {
        idx_front = 0;
        idx_last_plus_one = 0;
        num = 0;
    }

    inline const NInt &size() const { return num; }

    inline const NInt &front() const { return queue[idx_front]; }

    inline void pop() {
        --num;
        ++idx_front;
        idx_front &= mask;
    }

    inline void push(const NInt &_elem) {
        ++num;
        queue[idx_last_plus_one] = _elem;
        ++idx_last_plus_one;
        idx_last_plus_one &= mask;
    }

    inline bool empty() const {
        return idx_last_plus_one == idx_front;
    }
};

#endif //SCARA_BASICDEFINITION_H
