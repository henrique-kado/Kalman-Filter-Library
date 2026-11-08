#pragma once

#include <kalman/filter_base.hpp>

#include <functional>
#include <stdexcept>
#include <cmath>

namespace kalman {
/**
 * @brief An unscented Kalman filter implementation.
 * @tparam Scalar The scalar type (e.g., float, double).
 * @tparam N The dimension of the state vector.
 * @tparam M The dimension of the measurement vector.
 * @tparam C The dimension of the control input vector (default is 0).
 */
template <typename Scalar, int N, int M, int C = 0>
class UnscentedKalmanFilter : public FilterBase<Scalar, N, M, C> {
    using Base = FilterBase<Scalar, N, M, C>;
public:
    using typename Base::State;
    using typename Base::StateCovariance;
    using typename Base::Measurement;
    using typename Base::MeasurementCovariance;
    using typename Base::Control;

    using SigmaPoints = Eigen::Matrix<Scalar, N, 2 * N + 1>;
    using WeightsMean = Eigen::Matrix<Scalar, 2 * N + 1, 1>;
    using WeightsCovariance = Eigen::Matrix<Scalar, 2 * N + 1, 1>;
    using GainMatrix = Eigen::Matrix<Scalar, N, M>;
    using StateTransitionFunction = std::function<State(const State&, const Control&)>;
    using MeasurementFunction = std::function<Measurement(const State&)>;

    UnscentedKalmanFilter() : alpha_(1e-3), beta_(2.0), kappa_(0.0) { computeWeights(); }

    void predict() override {
        predict(Control::Zero());
    }

    void predict(const Control& u) override {
        if (!f_) throw std::runtime_error("State transition function is not set.");

        const SigmaPoints chi = generateSigmaPoints();

        // 1st pass: propagate every sigma point and compute the new mean.
        SigmaPoints chi_star;
        State x_new = State::Zero();
        for (int i = 0; i < 2 * N + 1; ++i) {
            chi_star.col(i) = f_(chi.col(i), u);
            x_new += wm_(i) * chi_star.col(i);
        }

        // 2nd pass: covariance around the NEW mean.
        StateCovariance P_new = StateCovariance::Zero();
        for (int i = 0; i < 2 * N + 1; ++i) {
            const State d = chi_star.col(i) - x_new;
            P_new += wc_(i) * d * d.transpose();
        }

        this->x_ = x_new;
        this->P_ = P_new + this->Q_;
    }

    void update(const Measurement& z) override {
        if (!h_) throw std::runtime_error("Measurement function is not set.");

        const SigmaPoints chi = generateSigmaPoints();

        // 1st pass: pass every sigma point through h and compute z_hat.
        Eigen::Matrix<Scalar, M, 2 * N + 1> Z;
        Measurement z_hat = Measurement::Zero();
        for (int i = 0; i < 2 * N + 1; ++i) {
            Z.col(i) = h_(chi.col(i));
            z_hat += wm_(i) * Z.col(i);
        }

        // 2nd pass: innovation covariance Pzz and cross covariance Pxz.
        MeasurementCovariance Pzz = MeasurementCovariance::Zero();
        GainMatrix Pxz = GainMatrix::Zero();  // same size as K: n x m
        for (int i = 0; i < 2 * N + 1; ++i) {
            const Measurement dz = Z.col(i) - z_hat;
            const State dx = chi.col(i) - this->x_;
            Pzz += wc_(i) * dz * dz.transpose();
            Pxz += wc_(i) * dx * dz.transpose();
        }
        Pzz += this->R_;

        // Gain K = Pxz Pzz^-1  <=>  K^T = Pzz^-1 Pxz^T
        const Eigen::LDLT<MeasurementCovariance> ldlt(Pzz);
        if (ldlt.info() != Eigen::Success || !ldlt.isPositive() ||
            (ldlt.vectorD().array() <= Scalar(0)).any()) {
            throw std::runtime_error("Innovation covariance Pzz is not positive definite.");
        }
        const GainMatrix K = ldlt.solve(Pxz.transpose()).transpose();

        this->x_ += K * (z - z_hat);
        this->P_ -= K * Pzz * K.transpose();
    }

    // Setters for parameters and functions
    void setParameters(Scalar alpha, Scalar beta, Scalar kappa) {
        alpha_ = alpha;
        beta_ = beta;
        kappa_ = kappa;
        computeWeights();
    }

    void setStateTransitionFunction(StateTransitionFunction f) {
        f_ = f;
    }

    void setMeasurementFunction(MeasurementFunction h) {
        h_ = h;
    }

private:
    Scalar alpha_;
    Scalar beta_;
    Scalar kappa_;

    StateTransitionFunction f_;
    MeasurementFunction h_;

    WeightsMean wm_;
    WeightsCovariance wc_;
    
    SigmaPoints generateSigmaPoints() const {
        // Cholesky: P = L * L.transpose
        const Eigen::LLT<StateCovariance> llt(this->P_);
        if (llt.info() != Eigen::Success) throw std::runtime_error("State Covariance P is not positive definite");

        const Scalar lambda = alpha_ * alpha_ * (N + kappa_) - N;
        const StateCovariance L = llt.matrixL();
        const StateCovariance offsets = std::sqrt(N + lambda) * L;

        SigmaPoints chi;
        chi.col(0) = this->x_;
        for(int i = 0; i < N; ++i) {
            chi.col(1 + i) = this->x_ + offsets.col(i);
            chi.col(1 + N + i) = this->x_ - offsets.col(i);
        }

        return chi;
    }

    void computeWeights() {
        const Scalar lambda = alpha_ * alpha_ * (N + kappa_) - N;
        wm_.setConstant(Scalar(1) / (Scalar(2) * (N + lambda)));
        wm_(0) = lambda / (N + lambda);
        wc_ = wm_;
        wc_(0) += Scalar(1) - alpha_ * alpha_ + beta_;
    }

};
} // namespace kalman