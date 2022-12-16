/*
  Matrix algebra computation
  Author: nyLiao
*/
#ifndef SCARA_MYTYPE_H
#define SCARA_MYTYPE_H

#include <cmath>
#include <cassert>
#include <iostream>
#include <Eigen/Dense>
#include "npy.hpp"
#include "BasicDefinition.h"

typedef Eigen::Matrix<ScoreFlt, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ScoreMatrix;
typedef Eigen::Map<ScoreMatrix> ScoreMap;
typedef Eigen::Ref<ScoreMatrix> ScoreRef;


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

        FltVector::iterator end() {
            return parent.data.begin() + (row + 1) * parent.ncol;
        }

        void erase() {
            parent.data.erase(parent.data.begin() + row * parent.ncol,
                              parent.data.begin() + (row + 1) * parent.ncol);
            parent.nrow--;
            parent.data.resize(parent.nrow * parent.ncol);
        }
    };

public:
    explicit My2DVector() {}

    explicit My2DVector(const NInt &_nrows, const NInt &_ncols) :
        data(_nrows * _ncols),
        nrow(_nrows),
        ncol(_ncols) {
    }

    void allocate(const NInt &_nrows, const NInt &_ncols) {
        data.resize(_nrows * _ncols);
        nrow = _nrows;
        ncol = _ncols;
    }

    My2DVectorRow operator[] (const NInt &row) {
        return My2DVectorRow(*this, row);
    }

    inline const NInt size() const { return nrow * ncol; }

    inline const NInt nrows() const { return nrow; }

    inline const NInt ncols() const { return ncol; }

    inline FltVector& get_data() { return data; }

    inline const FltVector& get_data() const { return data; }

    inline bool is_empty() const {
        return data.empty();
    }

    inline void set_ncol(const NInt &_ncol) { ncol = _ncol; }

    inline void set_nrow(const NInt &_nrow) { nrow = _nrow; }

    inline void clear() {
        data.clear();
        nrow = 0;
        ncol = 0;
    }

    void emplace_row(FltVector::iterator begin, FltVector::iterator end) {
        data.insert(data.end(), begin, end);
        nrow++;
    }

    void load_npy(const std::string file_path) {
        std::vector<unsigned long> shape;
        bool fortran_order;
        data.clear();
        npy::LoadArrayFromNumpy(file_path, shape, fortran_order, data);
        nrow = shape[0];    // feature size F
        ncol = shape[1];    // node num Vt_num
        printf("V2D    RSS RAM: %.3f GB\n", get_stat_memory());
        cout<<"Input file: "<<nrow<<" "<<ncol<<" "<<file_path<<endl;
    }

    void save_npy(const std::string file_path) {
        std::array<long unsigned, 2> res_shape {{nrow, ncol}};
        npy::SaveArrayAsNumpy(file_path, false, res_shape.size(), res_shape.data(), data);
        cout<<"Saved file: "<<nrow<<" "<<ncol<<" "<<file_path<<endl;
    }
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
        data(_nrows, FltVector(_ncols+2)),
        nrow(_nrows),
        ncol(_ncols) {
    }

    void allocate(const NInt &_nrows, const NInt &_ncols) {
        // data.resize(_nrows, FltVector(_ncols, 0));
        data = std::vector<FltVector>(_nrows, FltVector(_ncols+2));
        nrow = _nrows;
        ncol = _ncols;
    }

    FltVector &operator[] (const NInt &row) {
        return data[row];
    }

    const FltVector &operator[] (const NInt &row) const {
        return data[row];
    }

    inline const NInt size() const { return nrow * ncol; }

    inline const NInt nrows() const { return nrow; }

    inline const NInt ncols() const { return ncol; }

    inline bool is_empty() const {
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

    inline void set_size(const NInt &_nrow, const NInt &_ncol) {
        nrow = _nrow;
        ncol = _ncol;
    }

    inline void set_ncol(const NInt &_ncol) { ncol = _ncol; }

    inline void set_nrow(const NInt &_nrow) { nrow = _nrow; }

    inline void set_col(const NInt &col, const FltVector &_data) {
        assert(col < ncol);
        assert(_data.size() == nrow);
        for (NInt i = 0; i < nrow; ++i) {
            data[i][col] = _data[i];
        }
    }

    inline void copy_row(const NInt &row, const FltVector &_data) {
        data[row] = _data;
        data[row].resize(ncol+2);
    }

    void copy_rows(const IntVector row_idx, const MyMatrix &_data) {
        /*
         * row_idx[i] = j means copy _data[j] to data[i]
         */
        assert(row_idx.size() == nrow);
        for (NInt i = 0; i < row_idx.size(); ++i) {
            copy_row(i, _data[row_idx[i]]);
        }
    }

    void swap_rows(const IntVector row_idx, MyMatrix &_data) {
        assert(row_idx.size() == nrow);
        for (NInt i = 0; i < row_idx.size(); ++i) {
            data[i].swap(_data[row_idx[i]]);
        }
    }

    void from_V2D(My2DVector matv2d, const IntVector Vt_nodes) {
        data.resize(nrow);
        if (matv2d.ncols() == ncol) {
            for (long i = nrow-1; i >= 0; --i) {
                data[i] = FltVector(std::make_move_iterator(matv2d[i].begin()),
                                    std::make_move_iterator(matv2d[i].end()) );
                // NOTE: Actual size of data[i] is V_num+2 to be in line with SpeedPPR::gstruct.means
                data[i].emplace_back(0);
                data[i].emplace_back(0);
                matv2d[i].erase();
            }
        } else {
            // populate Vt_nodes to all nodes
            assert(matv2d.ncols() == Vt_nodes.size());
            NInt idx = 0;
            for (long i = nrow-1; i >= 0; --i) {
                data[i] = FltVector(ncol+2, 0);
                for (NInt j = 0; j < ncol; ++j) {
                    if (Vt_nodes[idx] == j) {
                        data[i][j] = matv2d[i][idx];
                        idx++;
                    }
                }
                matv2d[i].erase();
            }
        }
        matv2d.clear();
        // ! Still require O(2n) RAM as My2DVectorRow::erase does not reallocate
        printf("Mat    RSS RAM: %.3f GB\n", get_stat_memory());
        cout<<"Feat  size: "<<data.size()<<" "<<data[0].size()-2<<endl;
    }

    void to_V2D(My2DVector &matv2d) {
        matv2d.clear();
        matv2d.set_ncol(ncol);
        for (NInt i = 0; i < nrow; ++i) {
            matv2d.emplace_row(data[i].begin(), data[i].end()-2);
            data[i].clear();
        }
        cout<<"Save  size: "<<matv2d.nrows()<<" "<<matv2d.ncols()<<" "<<matv2d.get_data().size()<<endl;
    }
};

#endif //SCARA_MYTYPE_H
