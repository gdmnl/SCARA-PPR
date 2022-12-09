#ifndef SCARA_BASICDEFINITION_H
#define SCARA_BASICDEFINITION_H

// #ifndef ENABLE_RW
// #define ENABLE_RW
// #endif

#include <queue>

using std::cout;
using std::endl;

#define IDFMT "lu"
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

#endif //SCARA_BASICDEFINITION_H
