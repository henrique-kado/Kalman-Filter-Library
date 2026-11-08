#pragma once

#include <kalman/filter_base.hpp>

#include <stdexcept>

namespace kalman {
/**
 * @brief A simple Kalman filter implementation.
 * @tparam Scalar The scalar type (e.g., float, double).
 * @tparam N The dimension of the state vector.
 * @tparam M The dimension of the measurement vector.
 * @tparam C The dimension of the control input vector (default is 0).
 */
template <typename Scalar, int N, int M, int C = 0>
class KalmanFilter : public FilterBase<Scalar, N, M, C> {
    using Base = FilterBase<Scalar, N, M, C>;
public:
    using typename Base::State;
    using typename Base::StateCovariance;
    using typename Base::Measurement;
    using typename Base::MeasurementCovariance;
    using typename Base::Control;

    using StateTransitionMatrix = Eigen::Matrix<Scalar, N, N>;
    using ControlMatrix = Eigen::Matrix<Scalar, N, C>;
    using MeasurementMatrix = Eigen::Matrix<Scalar, M, N>;
    using GainMatrix = Eigen::Matrix<Scalar, N, M>;

    KalmanFilter() : 
        F_(StateTransitionMatrix::Identity()), 
        B_(ControlMatrix::Zero()), 
        H_(MeasurementMatrix::Zero()) {}

    void predict() override {
        // Predict the next state
        this->x_ = F_ * this->x_;
        this->P_ = F_ * this->P_ * F_.transpose() + this->Q_;
    }

    void predict(const Control& u) override {
        // Predict the next state with control input
        this->x_ = F_ * this->x_ + B_ * u;
        this->P_ = F_ * this->P_ * F_.transpose() + this->Q_;
    }

    void update(const Measurement& z) override {
        // Update the state with measurement
        const Measurement y = z - H_ * this->x_; // Measurement residual
        const MeasurementCovariance S = H_ * this->P_ * H_.transpose() + this->R_; // Residual covariance
        Eigen::LDLT<MeasurementCovariance> S_ldlt(S);
        if (S_ldlt.info() != Eigen::Success || !S_ldlt.isPositive() || (S_ldlt.vectorD().array() <= Scalar(0)).any()) {
            throw std::runtime_error("Residual covariance S is not positive definite.");
        }
        const GainMatrix K = S_ldlt.solve(H_ * this->P_).transpose(); // Kalman gain

        this->x_ += K * y; // Update state estimate
        const StateCovariance I_KH = StateCovariance::Identity() - K * H_;
        this->P_ = I_KH * this->P_ * I_KH.transpose() + K * this->R_ * K.transpose(); // Update covariance estimate
    }

    // Setters for matrices
    void setStateTransitionMatrix(const StateTransitionMatrix& F) {
        F_ = F;
    }

    void setControlMatrix(const ControlMatrix& B) {
        B_ = B;
    }

    void setMeasurementMatrix(const MeasurementMatrix& H) {
        H_ = H;
    }

    // Getters for matrices
    const StateTransitionMatrix& getStateTransitionMatrix() const {
        return F_;
    }

    const ControlMatrix& getControlMatrix() const {
        return B_;
    }

    const MeasurementMatrix& getMeasurementMatrix() const {
        return H_;
    }

private:
    StateTransitionMatrix F_; // State transition matrix
    ControlMatrix B_; // Control input matrix
    MeasurementMatrix H_; // Measurement matrix

};
} // namespace kalman