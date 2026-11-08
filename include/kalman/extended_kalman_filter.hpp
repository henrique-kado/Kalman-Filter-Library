#pragma once

#include <kalman/filter_base.hpp>
#include <functional>
#include <stdexcept>

namespace kalman {
/**
 * @brief An extended Kalman filter implementation.
 * @tparam Scalar The scalar type (e.g., float, double).
 * @tparam N The dimension of the state vector.
 * @tparam M The dimension of the measurement vector.
 * @tparam C The dimension of the control input vector (default is 0).
*/
template <typename Scalar, int N, int M, int C = 0>
class ExtendedKalmanFilter : public FilterBase<Scalar, N, M, C> {

public:
    using Base = FilterBase<Scalar, N, M, C>;
    using typename Base::State;
    using typename Base::StateCovariance;
    using typename Base::Measurement;
    using typename Base::MeasurementCovariance;
    using typename Base::Control;

    using StateTransitionMatrix = Eigen::Matrix<Scalar, N, N>;
    using MeasurementMatrix = Eigen::Matrix<Scalar, M, N>;
    using GainMatrix = Eigen::Matrix<Scalar, N, M>;

    using StateTransitionFunction = std::function<State(const State&, const Control&)>;
    using MeasurementFunction = std::function<Measurement(const State&)>;
    using StateTransitionJacobian = std::function<StateTransitionMatrix(const State&, const Control&)>;
    using MeasurementJacobian = std::function<MeasurementMatrix(const State&)>;

    void predict() override {
        if (f_ == nullptr) throw std::runtime_error("State transition function is not set.");
        if (F_ == nullptr) throw std::runtime_error("State transition Jacobian is not set.");

        const StateTransitionMatrix F_Jacobian = F_(this->x_, Control::Zero());
        this->x_ = f_(this->x_, Control::Zero());
        this->P_ = F_Jacobian * this->P_ * F_Jacobian.transpose() + this->Q_;
    }

    void predict(const Control& u) override {
        if (f_ == nullptr) throw std::runtime_error("State transition function is not set.");
        if (F_ == nullptr) throw std::runtime_error("State transition Jacobian is not set.");

        const StateTransitionMatrix F_Jacobian = F_(this->x_, u);
        this->x_ = f_(this->x_, u);
        this->P_ = F_Jacobian * this->P_ * F_Jacobian.transpose() + this->Q_;
    }

    void update(const Measurement& z) override {
        if (h_ == nullptr) throw std::runtime_error("Measurement function is not set.");
        if (H_ == nullptr) throw std::runtime_error("Measurement Jacobian is not set.");

        const MeasurementMatrix H_Jacobian = H_(this->x_);
        const Measurement y = z - h_(this->x_); // Measurement residual
        const MeasurementCovariance S = H_Jacobian * this->P_ * H_Jacobian.transpose() + this->R_; // Residual covariance
        
        const Eigen::LDLT<MeasurementCovariance> S_ldlt(S);
        if (S_ldlt.info() != Eigen::Success || !S_ldlt.isPositive() || (S_ldlt.vectorD().array() <= Scalar(0)).any()) {
            throw std::runtime_error("Residual covariance S is not positive definite.");
        }
        const GainMatrix K = S_ldlt.solve(H_Jacobian * this->P_).transpose(); // Kalman gain
        this->x_ += K * y; // Update state estimate
        const StateCovariance I_KH = StateCovariance::Identity() - K * H_Jacobian;
        this->P_ = I_KH * this->P_ * I_KH.transpose() + K * this->R_ * K.transpose(); // Update covariance estimate
    }

    // Setters for functions and Jacobians
    void setStateTransitionFunction(const StateTransitionFunction& f) {
        f_ = f;
    }

    void setMeasurementFunction(const MeasurementFunction& h) {
        h_ = h;
    }

    void setStateTransitionJacobian(const StateTransitionJacobian& F) {
        F_ = F;
    }

    void setMeasurementJacobian(const MeasurementJacobian& H) {
        H_ = H;
    }

private:
    StateTransitionFunction f_; // State transition function
    MeasurementFunction h_; // Measurement function
    StateTransitionJacobian F_; // State transition Jacobian
    MeasurementJacobian H_; // Measurement Jacobian

};
} // namespace kalman