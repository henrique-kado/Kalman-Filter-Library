#pragma once

#include <Eigen/Dense>
#include <type_traits>

namespace kalman {

/**
 * @brief Base class for Kalman filters.
 * @tparam Scalar The scalar type (e.g., float, double).
 * @tparam N The dimension of the state vector.
 * @tparam M The dimension of the measurement vector.
 * @tparam C The dimension of the control input vector (default is 0).
 */
template <typename Scalar, int N, int M, int C = 0>
class FilterBase {
public:
    // Type definitions
    using State = Eigen::Matrix<Scalar, N, 1>;
    using StateCovariance = Eigen::Matrix<Scalar, N, N>;
    using Measurement = Eigen::Matrix<Scalar, M, 1>;
    using MeasurementCovariance = Eigen::Matrix<Scalar, M, M>;
    using Control = Eigen::Matrix<Scalar, C, 1>;

    static_assert(std::is_floating_point<Scalar>::value, "Scalar must be a floating-point type.");

    virtual ~FilterBase() = default;

    // Pure virtual functions to be implemented by derived classes
    virtual void predict() = 0;
    virtual void predict(const Control& u) = 0;
    virtual void update(const Measurement& z) = 0;

    // Initialization function to set the initial state and covariance
    void init(const State& x0, const StateCovariance& P0) {
        x_ = x0;
        P_ = P0;
    }

    // Setters for process and measurement noise covariance
    void setProcessNoiseCovariance(const StateCovariance& Q) {
        Q_ = Q;
    }
    void setMeasurementNoiseCovariance(const MeasurementCovariance& R) {
        R_ = R;
    }

    // Getters for state and covariance
    const State& getState() const {
        return x_;
    }
    const StateCovariance& getStateCovariance() const {
        return P_;
    }
    const StateCovariance& getProcessNoiseCovariance() const {
        return Q_;
    }
    const MeasurementCovariance& getMeasurementNoiseCovariance() const {
        return R_;
    }

protected:
    State x_; // State vector
    StateCovariance P_; // State covariance matrix
    StateCovariance Q_; // Process noise covariance matrix
    MeasurementCovariance R_; // Measurement noise covariance matrix

    FilterBase() :  
        x_(State::Zero()),
        P_(StateCovariance::Identity()),
        Q_(StateCovariance::Zero()), 
        R_(MeasurementCovariance::Identity()) {}

};
} // namespace kalman