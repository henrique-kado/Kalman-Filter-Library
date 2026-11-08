// Tests for kalman::ExtendedKalmanFilter.
//
// Every expected value below is computed by hand (shown in the comments),
// never read back from the filter, so a wrong implementation cannot pass.

// Keep assert() active even in Release builds: these are tests.
#undef NDEBUG
#include <kalman/extended_kalman_filter.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

constexpr double kTol = 1e-12;

bool near(double a, double b, double tol = kTol) { return std::abs(a - b) < tol; }

// Returns true if calling fn() throws std::runtime_error.
template <typename Fn>
bool throwsRuntimeError(Fn fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Missing functions or Jacobians must throw a clear error, not crash.
// ---------------------------------------------------------------------------
void testThrowsWhenModelIsMissing() {
    using EKF = kalman::ExtendedKalmanFilter<double, 1, 1>;
    EKF::Measurement z;
    z << 1.0;

    {
        EKF filter;  // nothing set
        assert(throwsRuntimeError([&] { filter.predict(); }) && "predict() without f should throw.");
        assert(throwsRuntimeError([&] { filter.update(z); }) && "update() without h should throw.");
    }
    {
        EKF filter;  // f set, Jacobian of f missing
        filter.setStateTransitionFunction([](const EKF::State& x, const EKF::Control&) { return x; });
        assert(throwsRuntimeError([&] { filter.predict(); }) && "predict() without the Jacobian of f should throw.");
    }
    {
        EKF filter;  // h set, Jacobian of h missing
        filter.setMeasurementFunction([](const EKF::State& x) { return x; });
        assert(throwsRuntimeError([&] { filter.update(z); }) && "update() without the Jacobian of h should throw.");
    }
}

// ---------------------------------------------------------------------------
// With a linear model the EKF must give exactly the linear KF result.
// Same scenario as testPositionVelocity in test_kalman_filter.cpp:
//
//   f(x) = F x, F = [1 1; 0 1]     h(x) = H x, H = [1 0]
//   x0 = [0, 1], P0 = I, Q = 0, R = 1, z = 2
//   predict: x = [1, 1],      P = [2 1; 1 1]
//   update:  x = [5/3, 4/3],  P = [2/3 1/3; 1/3 2/3]
// ---------------------------------------------------------------------------
void testLinearModelMatchesKalmanFilter() {
    using EKF = kalman::ExtendedKalmanFilter<double, 2, 1>;
    EKF filter;

    EKF::StateTransitionMatrix F;
    F << 1.0, 1.0,
         0.0, 1.0;
    EKF::MeasurementMatrix H;
    H << 1.0, 0.0;

    filter.setStateTransitionFunction([F](const EKF::State& x, const EKF::Control&) -> EKF::State { return F * x; });
    filter.setStateTransitionJacobian([F](const EKF::State&, const EKF::Control&) { return F; });
    filter.setMeasurementFunction([H](const EKF::State& x) -> EKF::Measurement { return H * x; });
    filter.setMeasurementJacobian([H](const EKF::State&) { return H; });

    EKF::State x0;
    x0 << 0.0, 1.0;
    filter.init(x0, EKF::StateCovariance::Identity());

    filter.predict();
    EKF::State x_pred;
    x_pred << 1.0, 1.0;
    EKF::StateCovariance P_pred;
    P_pred << 2.0, 1.0,
              1.0, 1.0;
    assert(filter.getState().isApprox(x_pred, kTol) && "Linear: predicted x should be [1, 1].");
    assert(filter.getStateCovariance().isApprox(P_pred, kTol) && "Linear: predicted P should be [2 1; 1 1].");

    EKF::Measurement z;
    z << 2.0;
    filter.update(z);
    EKF::State x_upd;
    x_upd << 5.0 / 3.0, 4.0 / 3.0;
    EKF::StateCovariance P_upd;
    P_upd << 2.0 / 3.0, 1.0 / 3.0,
             1.0 / 3.0, 2.0 / 3.0;
    assert(filter.getState().isApprox(x_upd, kTol) && "Linear: updated x should be [5/3, 4/3].");
    assert(filter.getStateCovariance().isApprox(P_upd, kTol) && "Linear: updated P should be [2/3 1/3; 1/3 2/3].");
}

// ---------------------------------------------------------------------------
// predict(): the Jacobian of f must be evaluated at the state BEFORE the
// prediction.
//
//   f(x) = x^2, F(x) = 2x, x0 = 2, P0 = 1, Q = 0
//   correct: F = F(2) = 4  ->  x = 4, P = 4 * 1 * 4 = 16
//   wrong:   F = F(4) = 8  ->  P = 64
// ---------------------------------------------------------------------------
void testPredictLinearizesAtPriorState() {
    using EKF = kalman::ExtendedKalmanFilter<double, 1, 1>;
    EKF filter;

    filter.setStateTransitionFunction([](const EKF::State& x, const EKF::Control&) -> EKF::State {
        return x.array().square().matrix();
    });
    filter.setStateTransitionJacobian([](const EKF::State& x, const EKF::Control&) {
        EKF::StateTransitionMatrix F;
        F << 2.0 * x(0);
        return F;
    });

    EKF::State x0;
    x0 << 2.0;
    EKF::StateCovariance P0;
    P0 << 1.0;
    filter.init(x0, P0);

    filter.predict();
    assert(near(filter.getState()(0), 4.0) && "x^2: predicted x should be 4.");
    assert(near(filter.getStateCovariance()(0, 0), 16.0) &&
           "x^2: predicted P should be 16 (Jacobian taken at x = 2, not at x = 4).");
}

// ---------------------------------------------------------------------------
// predict(u) passes u to f and to its Jacobian, linearizing at the prior
// state; predict() passes u = 0.
//
//   f(x, u) = u x^2, F(x, u) = 2 u x, x0 = 2, P0 = 1, Q = 0
//   predict(u = 3): F = 2 * 3 * 2 = 12  ->  x = 3 * 4 = 12, P = 144
//                   (Jacobian taken after the state update: F = 72, P = 5184)
//   predict():      u = 0  ->  x = 0, P = 0
// ---------------------------------------------------------------------------
void testControlIsPassedToModel() {
    using EKF = kalman::ExtendedKalmanFilter<double, 1, 1, 1>;
    EKF filter;

    filter.setStateTransitionFunction([](const EKF::State& x, const EKF::Control& u) -> EKF::State {
        return u(0) * x.array().square().matrix();
    });
    filter.setStateTransitionJacobian([](const EKF::State& x, const EKF::Control& u) {
        EKF::StateTransitionMatrix F;
        F << 2.0 * u(0) * x(0);
        return F;
    });

    EKF::State x0;
    x0 << 2.0;
    EKF::StateCovariance P0;
    P0 << 1.0;
    filter.init(x0, P0);

    EKF::Control u;
    u << 3.0;
    filter.predict(u);
    assert(near(filter.getState()(0), 12.0) && "Control: predicted x should be 12.");
    assert(near(filter.getStateCovariance()(0, 0), 144.0) &&
           "Control: predicted P should be 144 (Jacobian taken at x = 2, not at x = 12).");

    filter.predict();
    assert(near(filter.getState()(0), 0.0) && "predict() should pass u = 0 to f.");
    assert(near(filter.getStateCovariance()(0, 0), 0.0) && "predict() should pass u = 0 to the Jacobian.");
}

// ---------------------------------------------------------------------------
// update() with a nonlinear measurement: range from the origin (radar).
// The Jacobian of h must be evaluated at the predicted state.
//
//   x = [px, py] = [3, 4], P = I, R = 1, z = 6
//   h(x) = r = sqrt(px^2 + py^2) = 5      H = [px/r, py/r] = [0.6, 0.8]
//   y = 6 - 5 = 1
//   S = H P H^T + R = 0.36 + 0.64 + 1 = 2
//   K = P H^T / S = [0.3, 0.4]
//   x = [3 + 0.3, 4 + 0.4] = [3.3, 4.4]
//   P = (I - K H) P = I - [0.18 0.24; 0.24 0.32] = [0.82 -0.24; -0.24 0.68]
// ---------------------------------------------------------------------------
void testRangeMeasurementUpdate() {
    using EKF = kalman::ExtendedKalmanFilter<double, 2, 1>;
    EKF filter;

    filter.setMeasurementFunction([](const EKF::State& x) -> EKF::Measurement {
        EKF::Measurement z;
        z << x.norm();
        return z;
    });
    filter.setMeasurementJacobian([](const EKF::State& x) {
        EKF::MeasurementMatrix H;
        H << x(0) / x.norm(), x(1) / x.norm();
        return H;
    });

    EKF::State x0;
    x0 << 3.0, 4.0;
    filter.init(x0, EKF::StateCovariance::Identity());

    EKF::Measurement z;
    z << 6.0;
    filter.update(z);

    EKF::State x_expected;
    x_expected << 3.3, 4.4;
    EKF::StateCovariance P_expected;
    P_expected << 0.82, -0.24,
                  -0.24, 0.68;
    assert(filter.getState().isApprox(x_expected, kTol) && "Range: updated x should be [3.3, 4.4].");
    assert(filter.getStateCovariance().isApprox(P_expected, kTol) &&
           "Range: updated P should be [0.82 -0.24; -0.24 0.68].");
}

// ---------------------------------------------------------------------------
// Two measurements (range and bearing) of a fixed point: the estimate must
// approach the true position and P must stay symmetric.
//
//   true position = [3, 4], initial guess = [2, 5], P0 = I, R = 0.01 I, Q = 0
//   h(x) = [r, theta] = [sqrt(px^2 + py^2), atan2(py, px)]
//   H    = [ px/r     py/r   ]
//          [ -py/r^2  px/r^2 ]
//
// With Q = 0 the filter treats the point as perfectly still, so each new
// measurement gets a weight of roughly 1/k: the error shrinks steadily but
// slowly (about 0.27 after 1 step, 0.06 after 20, 0.025 after 50). The test
// checks that the error goes down and ends small, not that it reaches zero.
// ---------------------------------------------------------------------------
void testRangeBearingConverges() {
    using EKF = kalman::ExtendedKalmanFilter<double, 2, 2>;
    EKF filter;

    // The point does not move: f(x) = x, Jacobian = I.
    filter.setStateTransitionFunction([](const EKF::State& x, const EKF::Control&) { return x; });
    filter.setStateTransitionJacobian([](const EKF::State&, const EKF::Control&) {
        return EKF::StateTransitionMatrix::Identity().eval();
    });

    auto h = [](const EKF::State& x) -> EKF::Measurement {
        EKF::Measurement z;
        z << x.norm(), std::atan2(x(1), x(0));
        return z;
    };
    filter.setMeasurementFunction(h);
    filter.setMeasurementJacobian([](const EKF::State& x) {
        const double r2 = x.squaredNorm();
        const double r = std::sqrt(r2);
        EKF::MeasurementMatrix H;
        H << x(0) / r,   x(1) / r,
             -x(1) / r2, x(0) / r2;
        return H;
    });
    filter.setMeasurementNoiseCovariance(EKF::MeasurementCovariance::Identity() * 0.01);

    EKF::State truth;
    truth << 3.0, 4.0;
    EKF::State guess;
    guess << 2.0, 5.0;
    filter.init(guess, EKF::StateCovariance::Identity());

    const EKF::Measurement z = h(truth);  // noiseless measurements
    const double initial_error = (guess - truth).norm();  // sqrt(2) ~ 1.41
    for (int k = 0; k < 50; ++k) {
        filter.predict();
        filter.update(z);

        const EKF::StateCovariance& P = filter.getStateCovariance();
        assert(near(P(0, 1), P(1, 0), 1e-9) && "Range-bearing: P should stay symmetric.");
        assert(P(0, 0) > 0.0 && P(1, 1) > 0.0 && "Range-bearing: P should keep a positive diagonal.");
    }

    const double final_error = (filter.getState() - truth).norm();
    assert(final_error < initial_error / 10.0 && "Range-bearing: error should shrink by at least 10x.");
    assert(final_error < 0.05 && "Range-bearing: estimate should end within 0.05 of [3, 4].");
}

}  // namespace

int main() {
    testThrowsWhenModelIsMissing();
    testLinearModelMatchesKalmanFilter();
    testPredictLinearizesAtPriorState();
    testControlIsPassedToModel();
    testRangeMeasurementUpdate();
    testRangeBearingConverges();

    std::cout << "All ExtendedKalmanFilter tests passed" << std::endl;
}
