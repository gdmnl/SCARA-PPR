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


inline ScoreMatrix shrink(const ScoreMatrix& X, const ScoreFlt tol) {
    const ScoreMatrix a_plus = X.array() + tol;
    const ScoreMatrix a_minus = X.array() - tol;
    return a_plus.cwiseMin(0) + a_minus.cwiseMax(0);
}

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

    ScoreVector singularValues() { return S_; }     // shape: rank
    ScoreMatrix matrixU() { return U_; }            // shape: m.rows * rank
    ScoreMatrix matrixV() { return V_; }            // shape: m.cols * rank

    ScoreMatrix pinv() {                            // shape: m.cols * m.rows
        ScoreVector Sinv(S_.size());
        for (int i = 0; i < S_.size(); ++i) {
            Sinv(i) = (S_(i) > 1e-6) ? (1.0/S_(i)) : 0.0;
        }
        // if ((Sinv.array() == 0).count() > 0)
        //     std::cout << "Warning: SVD matrix is singular" << std::endl;
        return V_ * Sinv.asDiagonal() * U_.transpose();
    }

private:
    ScoreMatrix U_, V_;
    ScoreVector S_;
    ScoreFlt TOL = 1e-6;
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
private:
    ScoreMatrix M, L, S;
    ScoreFlt nr, nc, spe_norm, fro_norm, l1_norm, errmin, mul;
    ScoreFlt TOL = 1e-3;
    bool trans;

public:
    RobustPca(const ScoreMatrix _M, const ScoreFlt _mul = 4.0) :
            M(_M), mul((_mul > 4) ? _mul : 4.0) {
        trans = (M.rows() < M.cols());
        if (trans) {
            cout<<"! Transposing matrix"<<endl;
            M.transposeInPlace();
        }
        nr = M.rows();                              // dimension of sample
        nc = M.cols();                              // dimension of feature
        L = ScoreMatrix::Zero(nr, nc);
        S = ScoreMatrix::Zero(nr, nc);

        spe_norm = speNorm(M);                      // matrix spectral norm
        fro_norm = M.norm();                        // matrix frobenius norm
        // l1_norm = M.cwiseAbs().colwise().sum().maxCoeff();   // matrix l1 norm
        l1_norm = M.lpNorm<1>();                    // coefficient-wise l1 norm
        cout<< "  RPCA L1 norm: "<<M.cwiseAbs().colwise().sum().maxCoeff() <<" Abs norm: "<<M.lpNorm<1>() << endl;
        errmin = TOL * fro_norm;
    }

    ScoreMatrix LowRankComponent() { return (trans? L.transpose() : L ); }

    ScoreMatrix SparseComponent()  { return (trans? S.transpose() : S ); }

    // Encourages low-rank by taking (truncated) SVD, then setting small singular values to zero
    int svd_truncate(const ScoreMatrix& X, int rank, ScoreFlt min_sv, ScoreMatrix& L) {
        RandomizedSvd rsvd(X, rank, (int)ceil(0.2*rank)+1, 1);
        ScoreVector s  = rsvd.singularValues();
        ScoreVector s0 = s.cwiseAbs() - min_sv * ScoreVector::Ones(s.size());
        int nnz = (s0.array() > 0).count();
        // cout<<"S: "<<s.head(1).transpose()<<", nS: "<<nnz<<" ";
        L = rsvd.matrixU().leftCols(nnz) * s.head(nnz).asDiagonal() * rsvd.matrixV().transpose().topRows(nnz);
        return nnz;
    }

    // Returns largest singular value of M
    ScoreFlt speNorm(const ScoreMatrix& M, int rank = 5) {
        RandomizedSvd rsvd(M, rank, (int)ceil(0.2*rank)+1, 1);
        return rsvd.singularValues()[0];
    }

    /*
     * mul: threshold multiplier, larger mul = more sparse but larger error
     * k: L2 mu stepsize, larger k = less sparse but faster convergence
     */
    void fit(const int rank = 1, const int maxiter = 5, const ScoreFlt k = 1.0) {
        ScoreFlt lambda = mul * rank / (4 * sqrt(nr + nc));
        ScoreFlt mu = k * nc * nr / (4 * l1_norm);
        ScoreFlt mu_bar = mu / TOL;
        ScoreFlt rho = k * 1.5;
        ScoreFlt init_scale = std::max(spe_norm, M.lpNorm<Eigen::Infinity>()) * lambda;
        ScoreMatrix Z = M / init_scale;
        ScoreMatrix M2;

        const int sv_step = std::max(1, (int)ceil(rank / maxiter));
        int sv = sv_step;

        for (int i = 0; i < maxiter; ++i) {
            // cout<<"  rank: "<<sv<<", r: "<<lambda/mu<<". s: "<<1/mu<<", ";
            M2 = M + Z / mu;
            /* update estimate of low-rank component */
            int svp = svd_truncate(M2 - S, sv, 1/mu, L);
            sv += ((svp < sv) ? 1 : sv_step);
            /* update estimate of sparse component */
            // Encourages sparsity in M by slightly shrinking all values, thresholding small values to zero
            S = shrink(M2 - L, lambda/mu);

            /* compute residual */
            ScoreMatrix Zi = M - L - S;
            ScoreFlt err = Zi.norm();
            // cout << "err: "<<err/fro_norm << " Abs: "<<S.lpNorm<1>() << endl;
            if (err < errmin) {
                break;
            }

            Z += mu * Zi;
            mu = (mu * rho < mu_bar) ? mu * rho : mu_bar;
        }
        // mu = k * nc * nr / (4 * l1_norm);
        // M2 = M + Z / mu;
        // svd_truncate(M2 - S, rank, 1/mu, L);
        // shrink(M2 - L, lambda/mu, S);
    }

    ScoreMatrix fit_fixed(const ScoreMatrix B, const int maxiter = 5, const ScoreFlt k = 1.0) {
        int rank = B.cols();
        ScoreFlt lambda = mul * rank / (4 * sqrt(nr + nc));
        ScoreFlt mu = k * nc * nr / (4 * l1_norm);
        ScoreFlt mu_bar = mu / TOL;
        ScoreFlt rho = k * 1.5;
        ScoreFlt init_scale = std::max(spe_norm, M.lpNorm<Eigen::Infinity>()) * lambda;
        // ScoreMatrix Z = M / init_scale;
        ScoreMatrix Z = ScoreMatrix::Zero(nr, nc);
        ScoreMatrix M2, Theta;

        RandomizedSvd Bsvd(B, rank);
        ScoreMatrix Binv = Bsvd.pinv();

        for (int i = 0; i < maxiter; ++i) {
            // cout<<"  r: "<<lambda/mu<<", ";
            M2 = M + Z / mu;
            /* update estimate of low-rank component */
            Theta = Binv * (M2 - S);
            L = B * Theta;
            /* update estimate of sparse component */
            S = shrink(M2 - L, lambda/mu);

            /* compute residual */
            ScoreMatrix Zi = M - L - S;
            ScoreFlt err = Zi.norm();
            // cout << "err: "<<err/fro_norm << " Abs: "<<S.lpNorm<1>() << endl;
            if (err < TOL * fro_norm) {
                break;
            }

            Z += mu * Zi;
            mu = (mu * rho < mu_bar) ? mu * rho : mu_bar;
        }
        return Binv * (M - S);
    }
};

#endif  // SCARA_FEATUREDECOMP_H
