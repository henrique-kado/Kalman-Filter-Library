#include <kalman/filter_base.hpp>
#include <cassert>
#include <iostream>

// Dummy derived class for testing purposes
class DummyFilter : public kalman::FilterBase<double, 2, 1> {
public:
    void predict() override {}
    void predict(const Control&) override {}
    void update(const Measurement&) override {}
};

int main() {
    DummyFilter filter;

    // Test default initialization
    assert(filter.getState().isZero() && "Default state should be zero.");
    assert(filter.getStateCovariance().isIdentity() && "Default covariance should be identity.");
    assert(filter.getProcessNoiseCovariance().isZero() && "Default process noise covariance should be zero.");
    assert(filter.getMeasurementNoiseCovariance().isIdentity() && "Default measurement noise covariance should be identity.");

    using Base = kalman::FilterBase<double, 2, 1>;

    // Test initialization with specific state and covariance
    Base::State x0;
    x0 << 1.0, 2.0;
    
    Base::StateCovariance P0;
    P0 << 1.0, 0.5,
          0.5, 1.0;
    filter.init(x0, P0);

    assert(filter.getState().isApprox(x0) && "Initialized state should match.");
    assert(filter.getStateCovariance().isApprox(P0) && "Initialized covariance should match.");

    // Test setters
    Base::StateCovariance Q;
    Q << 0.1, 0.0,
         0.0, 0.1;

    Base::MeasurementCovariance R;
    R << 0.5;

    filter.setProcessNoiseCovariance(Q);
    filter.setMeasurementNoiseCovariance(R);

    assert(filter.getProcessNoiseCovariance().isApprox(Q) && "Process noise covariance should match.");
    assert(filter.getMeasurementNoiseCovariance().isApprox(R) && "Measurement noise covariance should match.");

    std::cout << "All tests passed" << std::endl;
}