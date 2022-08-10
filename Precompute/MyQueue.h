

#ifndef SPEEDPPR_MYQUEUE_H
#define SPEEDPPR_MYQUEUE_H

#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>
#include "BasicDefinition.h"

class MyQueue {
private:
    const VertexIdType mask;
    std::vector<VertexIdType> queue;
    VertexIdType num = 0;
    VertexIdType idx_front = 0;
    VertexIdType idx_last_plus_one = 0;
private:
    static inline VertexIdType compute_queue_size(const VertexIdType &_numOfVertices) {
        return (1u) << (uint32_t) ceil(log2(_numOfVertices + 2u));
    }

public:
    explicit MyQueue(const VertexIdType &_numOfVertices) :
            mask(compute_queue_size(_numOfVertices) - 1),
            queue(mask + 2u, 0) {}

    inline void clear() {
        idx_front = 0;
        idx_last_plus_one = 0;
        num = 0;
    }


    inline const VertexIdType &size() const {
        return num;
    }

    inline const VertexIdType &front() const {
        return queue[idx_front];
    }

    inline void pop() {
        --num;
        ++idx_front;
        idx_front &= mask;
    }

    inline void push(const VertexIdType &_elem) {
        ++num;
        queue[idx_last_plus_one] = _elem;
        ++idx_last_plus_one;
        idx_last_plus_one &= mask;
    }

    inline bool empty() const {
        return idx_last_plus_one == idx_front;
    }
};

struct FwdPushStructure {
    // reserve one slot for the dummy vertex
    MyQueue active_vertices;
    // reserve one slot for the dummy vertex
    std::vector<bool> is_active;

    explicit FwdPushStructure(const VertexIdType &numOfVertices) :
            active_vertices(numOfVertices + 1),
            is_active(numOfVertices + 1, false) {}
};


class MyMatrix {
private:
    // Use vector of vector instead of single vector, so directly assign vector
    // value by std::vector.swap can be 2x faster
    std::vector<std::vector<PageRankScoreType>> data;
    VertexIdType nrow = 0;
    VertexIdType ncol = 0;

public:
    explicit MyMatrix() {};

    explicit MyMatrix(const VertexIdType &_nrows, const VertexIdType &_ncols) :
        data(_nrows, std::vector<PageRankScoreType>(_ncols)),
        nrow(_nrows),
        ncol(_ncols) {}

    void allocate(const VertexIdType &_nrows, const VertexIdType &_ncols) {
        // data.resize(_nrows, std::vector<PageRankScoreType>(_ncols, 0));
        data = std::vector<std::vector<PageRankScoreType>>(_nrows, std::vector<PageRankScoreType>(_ncols));
        nrow = _nrows;
        ncol = _ncols;
    }

    std::vector<PageRankScoreType> &operator[] (const VertexIdType &row) {
        return data[row];
    }

    const std::vector<PageRankScoreType> &operator[] (const VertexIdType &row) const {
        return data[row];
    }

    inline const VertexIdType &size() const { return nrow; }

    inline const VertexIdType nrows() const { return nrow; }

    inline const VertexIdType ncols() const { return ncol; }

    inline bool empty() const {
        return nrow == 0;
    }

    inline bool is_regular() const {
        for (VertexIdType i = 0; i < nrow; ++i) {
            if (data[i].size() != ncol) {
                return false;
            }
        }
        return true;
    }

    inline bool is_regular(const VertexIdType &row) const {
        return data[row].size() == ncol;
    }

    inline void set_col(const VertexIdType &col, const std::vector<PageRankScoreType> &_data) {
        assert(col < ncol);
        assert(_data.size() == nrow);
        for (VertexIdType i = 0; i < nrow; ++i) {
            data[i][col] = _data[i];
        }
    }

    inline void set_row(const VertexIdType &row, const std::vector<PageRankScoreType> &_data) {
        data[row] = _data;
    }
};


#endif //SPEEDPPR_MYQUEUE_H
