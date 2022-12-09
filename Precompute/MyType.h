#ifndef SCARA_MYTYPE_H
#define SCARA_MYTYPE_H

#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>
#include "BasicDefinition.h"

class MyQueue {
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
    explicit MyQueue(const NInt &_numOfVertices) :
            mask(compute_queue_size(_numOfVertices) - 1),
            queue(mask + 2u, 0) {}

    inline void clear() {
        idx_front = 0;
        idx_last_plus_one = 0;
        num = 0;
    }


    inline const NInt &size() const {
        return num;
    }

    inline const NInt &front() const {
        return queue[idx_front];
    }

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

struct FwdPushStructure {
    // reserve one slot for the dummy vertex
    MyQueue active_vertices;
    // reserve one slot for the dummy vertex
    std::vector<bool> is_active;

    explicit FwdPushStructure(const NInt &numOfVertices) :
            active_vertices(numOfVertices + 1),
            is_active(numOfVertices + 1, false) {}
};


/*
 * Matrix in vector of vectors allow for fast row assign. Directly assign by std::vector.swap is 2x faster
 */
class MyMatrix {
private:
    std::vector<FltVector> data;
    NInt nrow = 0;
    NInt ncol = 0;

public:
    explicit MyMatrix() {}

    explicit MyMatrix(const NInt &_nrows, const NInt &_ncols) :
        data(_nrows, FltVector(_ncols)),
        nrow(_nrows),
        ncol(_ncols) {
        // cout << "Init Matrix of: " << nrow << " " << ncol << endl;
    }

    void allocate(const NInt &_nrows, const NInt &_ncols) {
        // data.resize(_nrows, FltVector(_ncols, 0));
        data = std::vector<FltVector>(_nrows, FltVector(_ncols));
        nrow = _nrows;
        ncol = _ncols;
        // cout << "Allocate Matrix of: " << nrow << " " << ncol << endl;
    }

    FltVector &operator[] (const NInt &row) {
        return data[row];
    }

    const FltVector &operator[] (const NInt &row) const {
        return data[row];
    }

    inline const NInt size() const { return nrow; }

    inline const NInt nrows() const { return nrow; }

    inline const NInt ncols() const { return ncol; }

    inline bool empty() const {
        return nrow == 0;
    }

    inline bool is_regular() const {
        for (NInt i = 0; i < nrow; ++i) {
            if (data[i].size() != ncol) {
                return false;
            }
        }
        return true;
    }

    inline bool is_regular(const NInt &row) const {
        return data[row].size() == ncol;
    }

    inline void set_col(const NInt &col, const FltVector &_data) {
        assert(col < ncol);
        assert(_data.size() == nrow);
        for (NInt i = 0; i < nrow; ++i) {
            data[i][col] = _data[i];
        }
    }

    inline void set_row(const NInt &row, const FltVector &_data) {
        data[row] = _data;
        data[row].resize(ncol);
    }
};


/*
 * Matrix in 1D vector.
 */
class My2DVector {
private:
    FltVector data;
    NInt nrow = 0;
    NInt ncol = 0;

    friend class My2DVectorRow;

    class My2DVectorRow {
    private:
        My2DVector &parent;
        NInt row;
    public:
        My2DVectorRow(My2DVector &_parent, const NInt &_row) :
            parent(_parent), row(_row) {}

        ScoreFlt &operator[] (const NInt &col) {
            return parent.data[row * parent.ncol + col];
        }

        const ScoreFlt &operator[] (const NInt &col) const {
            return parent.data[row * parent.ncol + col];
        }

        FltVector::iterator begin() {
            return parent.data.begin() + row * parent.ncol;
        }
    };

public:
    explicit My2DVector() {}

    explicit My2DVector(const NInt &_nrows, const NInt &_ncols) :
        data(_nrows * _ncols),
        nrow(_nrows),
        ncol(_ncols) {
        // cout << "Init 2DVector of: " << nrow << " " << ncol << endl;
    }

    void allocate(const NInt &_nrows, const NInt &_ncols) {
        data.resize(_nrows * _ncols);
        nrow = _nrows;
        ncol = _ncols;
        // cout << "Allocate 2DVector of: " << nrow << " " << ncol << endl;
    }

    My2DVectorRow operator[] (const NInt &row) {
        return My2DVectorRow(*this, row);
    }

    inline const NInt size() const { return nrow * ncol; }

    inline const NInt nrows() const { return nrow; }

    inline const NInt ncols() const { return ncol; }

    inline FltVector& get_data() { return data; }

    inline const FltVector& get_data() const { return data; }

    inline bool empty() const {
        return data.empty();
    }
};

#endif //SCARA_MYTYPE_H
