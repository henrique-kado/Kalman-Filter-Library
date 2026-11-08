#include <kalman/kalman_filter.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

using KF = kalman::KalmanFilter<double, 1, 1, 1>;

int main() {
    KF filter;

    // Test default initialization
    assert(filter.getStateTransitionMatrix().isIdentity() && "Default state transition matrix should be identity.");
    assert(filter.getControlMatrix().isZero() && "Default control matrix should be zero.");
    assert(filter.getMeasurementMatrix().isZero() && "Default measurement matrix should be zero.");

    // Test initialization with specific F, B, H matrices
    KF::StateTransitionMatrix F;
    F << 2.0;
    KF::ControlMatrix B;
    B << 1.0;
    KF::MeasurementMatrix H;
    H << 0.5;
    filter.setStateTransitionMatrix(F);
    filter.setControlMatrix(B);
    filter.setMeasurementMatrix(H);

    assert(filter.getStateTransitionMatrix().isApprox(F) && "State transition matrix should match.");
    assert(filter.getControlMatrix().isApprox(B) && "Control matrix should match.");
    assert(filter.getMeasurementMatrix().isApprox(H) && "Measurement matrix should match.");

    // Test predict and update methods
    KF::State x0;
    x0 << 1.0;
    KF::StateCovariance P0;
    P0 << 1.0;
    filter.init(x0, P0);

    filter.predict();
    assert(filter.getState().isApprox(F * x0) && "Predicted state should match.");
    assert(filter.getStateCovariance().isApprox(F * P0 * F.transpose() + filter.getProcessNoiseCovariance()) && "Predicted covariance should match.");

    // Expected values computed by hand (F = 2, x0 = 1, P0 = 1, Q = 0, H = 0.5, R = 1, z = 1.5):
    //   predict: x = 2,  P = 4
    //   update:  y = 1.5 - 0.5 * 2 = 0.5
    //            S = 0.5 * 4 * 0.5 + 1 = 2
    //            K = 4 * 0.5 / 2 = 1
    //            x = 2 + 1 * 0.5 = 2.5
    //            P = (1 - 1 * 0.5) * 4 * (1 - 1 * 0.5) + 1 * 1 * 1 = 2
    KF::Measurement z;
    z << 1.5;
    filter.update(z);
    assert(std::abs(filter.getState()(0) - 2.5) < 1e-12 && "Updated state should be 2.5.");
    assert(std::abs(filter.getStateCovariance()(0, 0) - 2.0) < 1e-12 && "Updated covariance should be 2.");

    std::cout << "All KalmanFilter tests passed" << std::endl;
}
