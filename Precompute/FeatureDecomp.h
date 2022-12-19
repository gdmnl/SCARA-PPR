/*
  Ref: https://github.com/kazuotani14/RandomizedSvd
  Based on: Candes, E. J., Li, X., Ma, Y., & Wright, J. (2009). Robust Principal Component Analysis.
*/

#ifndef SCARA_FEATUREDECOMP_H
#define SCARA_FEATUREDECOMP_H

#include <algorithm>
#include <iostream>
#include <cmath>
#include "BasicDefinition.h"
#include "MyType.h"

const ScoreFlt TOL = 1e-2;

/*
 * Randomized SVD for fast approximate matrix decomposition
 * Interface is same as Eigen's jacobiSVD
 */
class RandomizedSvd {
public:
    RandomizedSvd(const ScoreMatrix& m, const int rank, int oversamples = 0, int iter = 2)
        : U_(), V_(), S_(), rank_(rank) {
        ComputeRandomizedSvd(m, oversamples, iter);
    }

    ScoreVector singularValues() { return S_; }
    ScoreMatrix matrixU() { return U_; }
    ScoreMatrix matrixV() { return V_; }

private:
    ScoreMatrix U_, V_;
    ScoreVector S_;
    int rank_;

    /*
     * Main function for randomized svd
       * oversamples: additional samples/rank for accuracy, to account for random sampling
     */
    void ComputeRandomizedSvd(const ScoreMatrix& A, int oversamples, int iter) {
        // If matrix is too small for desired rank/oversamples
        if ((rank_ + oversamples) > std::min(A.rows(), A.cols())) {
            rank_ = std::min(A.rows(), A.cols());
            oversamples = 0;
        }

        ScoreMatrix Q = FindRandomizedRange(A, rank_ + oversamples, iter);
        ScoreMatrix B = Q.transpose() * A;

        // Compute the SVD on the thin matrix (much cheaper than SVD on original)
        Eigen::JacobiSVD<ScoreMatrix> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

        U_ = (Q * svd.matrixU()).block(0, 0, A.rows(), rank_);
        V_ = svd.matrixV().block(0, 0, A.cols(), rank_);
        S_ = svd.singularValues().head(rank_);
    }

    /*
      Finds a set of orthonormal vectors that approximates the range of A
      Basic idea is that finding orthonormal basis vectors for A*W, where W is set of some
      random vectors w_i, can approximate the range of A
      Most of the time/computation in the randomized SVD is spent here
    */
    ScoreMatrix FindRandomizedRange(const ScoreMatrix& A, int size, int iter) {
        int nr = A.rows(), nc = A.cols();
        ScoreMatrix L(nr, size);
        Eigen::FullPivLU<ScoreMatrix> lu1(nr, size);
        ScoreMatrix Q = ScoreMatrix::Random(nc, size);
        Eigen::FullPivLU<ScoreMatrix> lu2(nc, nr);

        // Normalized power iterations. Simply multiplying by A repeatedly makes alg unstable, so use LU to "normalize"
        for (int i = 0; i < iter; ++i) {
            lu1.compute(A * Q);
            L.setIdentity();
            L.block(0, 0, nr, size).triangularView<Eigen::StrictlyLower>() =
                lu1.matrixLU();

            lu2.compute(A.transpose() * L);
            Q.setIdentity();
            Q.block(0, 0, nc, size).triangularView<Eigen::StrictlyLower>() =
                lu2.matrixLU();
        }

        Eigen::ColPivHouseholderQR<ScoreMatrix> qr(A * Q);
        return qr.householderQ() * ScoreMatrix::Identity(nr, size);  // recover skinny Q matrix
    }
};

/*
 * Implementation of Robust PCA algorithm via Principal Component Pursuit
 * Separates a matrix into two-components: low-rank and sparse
 */
class RobustPca {
public:
    RobustPca(ScoreMatrix M, int rank = 1, int maxiter = 5, int k = 1) : L(), S() {
        ComputeRobustPca(M, rank, maxiter, k);
    }

    ScoreMatrix LowRankComponent() { return L; }
    ScoreMatrix SparseComponent() { return S; }

    // Encourages sparsity in M by slightly shrinking all values, thresholding small values to zero
    void shrink(const ScoreMatrix& M, ScoreFlt tau, ScoreMatrix& S) {
        ScoreMatrix S0 = M.cwiseAbs() - tau * ScoreMatrix::Ones(M.rows(), M.cols());
        S = (S0.array() > 0).select(M, ScoreMatrix::Zero(M.rows(), M.cols()));
    }

    // Encourages low-rank by taking (truncated) SVD, then setting small singular values to zero
    int svd_truncate(const ScoreMatrix& M, int rank, ScoreFlt min_sv, ScoreMatrix& L) {
        RandomizedSvd rsvd(M, rank, (int)ceil(0.2*rank)+1, 1);
        ScoreVector s  = rsvd.singularValues();
        ScoreVector s0 = s.cwiseAbs() - min_sv * ScoreVector::Ones(s.size());
        int nnz = (s0.array() > 0).count();
        // cout<<"S: "<<s.head(1).transpose()<<", nS: "<<nnz<<" ";
        L = rsvd.matrixU().leftCols(nnz) * s.head(nnz).asDiagonal() * rsvd.matrixV().transpose().topRows(nnz);
        return nnz;
    }

    // Returns largest singular value of M
    ScoreFlt norm_op(const ScoreMatrix& M, int rank = 5) {
        RandomizedSvd rsvd(M, rank, (int)ceil(0.2*rank)+1, 1);
        return rsvd.singularValues()[0];
    }

private:
    ScoreMatrix L, S;
    ScoreFlt nr, nc;

    /*
     * k: L2 mu stepsize, larger k = less sparse but faster convergence
     */
    void ComputeRobustPca(ScoreMatrix& M, int rank, int maxiter, int k) {
        bool trans = (M.rows() < M.cols());
        if (trans) {
            cout<<"! Transposing matrix"<<endl;
            M.transposeInPlace();
        }
        nr = M.rows();      // dimension of sample
        nc = M.cols();      // dimension of feature

        ScoreFlt lambda = sqrt(nr / (rank * nc));
        ScoreFlt op_norm = norm_op(M);
        ScoreFlt d_norm = M.norm();
        // ScoreFlt l1_norm = M.cwiseAbs().colwise().sum().maxCoeff();
        ScoreFlt l1_norm = M.lpNorm<1>();
        // cout<< "L1 norm: "<<M.cwiseAbs().colwise().sum().maxCoeff() <<" Abs norm: "<<M.lpNorm<1>() << endl;

        // ScoreFlt mu = k * 1.25 / op_norm;
        ScoreFlt mu = k * nc * nr / (4 * l1_norm);
        ScoreFlt mu_bar = mu / TOL;
        ScoreFlt rho = k * 1.5;
        ScoreFlt init_scale = std::max(op_norm, M.lpNorm<Eigen::Infinity>()) * lambda;
        ScoreMatrix Z = M / init_scale;
        ScoreMatrix M2;

        S = ScoreMatrix::Zero(nr, nc);
        const int sv_step = std::max(1, (int)ceil(rank / maxiter));
        int sv = sv_step;

        for (int i = 0; i < maxiter; ++i) {
            // cout<<"rank: "<<sv<<", r: "<<lambda/mu<<", s: "<<1/mu<<", ";
            M2 = M + Z / mu;
            /* update estimate of low-rank component */
            int svp = svd_truncate(M2 - S, sv, 1/mu, L);
            sv += ((svp < sv) ? 1 : sv_step);
            /* update estimate of sparse component */
            shrink(M2 - L, lambda/mu, S);

            /* compute residual */
            ScoreMatrix Zi = M - L - S;
            ScoreFlt err = Zi.norm();
            // cout << "err: "<<err/d_norm << " Abs: "<<S.lpNorm<1>() << endl;
            if (err < TOL * d_norm) {
                break;
            }

            Z += mu * Zi;
            mu = (mu * rho < mu_bar) ? mu * rho : mu_bar;
        }
        // mu = k * nc * nr / (4 * l1_norm);
        // M2 = M + Z / mu;
        // svd_truncate(M2 - S, rank, 1/mu, L);
        // shrink(M2 - L, lambda/mu, S);
        if (trans) {
            L.transposeInPlace();
            S.transposeInPlace();
        }
    }
};

#endif  // SCARA_FEATUREDECOMP_H
