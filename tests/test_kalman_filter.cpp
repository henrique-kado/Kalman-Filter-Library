// Tests for kalman::KalmanFilter.
//
// Every expected value below is computed by hand (shown in the comments),
// never read back from the filter, so a wrong implementation cannot pass.

// Keep assert() active even in Release builds: these are tests.
#undef NDEBUG
#include <kalman/kalman_filter.hpp>

#include <cassert>
#include <cmath>
#include <iostream>

namespace {

constexpr double kTol = 1e-12;

bool near(double a, double b, double tol = kTol) { return std::abs(a - b) < tol; }

// ---------------------------------------------------------------------------
// 1 state, 1 measurement, 1 control input: defaults and setters
// ---------------------------------------------------------------------------
void testDefaultsAndSetters() {
    using KF = kalman::KalmanFilter<double, 1, 1, 1>;
    KF filter;

    assert(filter.getStateTransitionMatrix().isIdentity() && "Default F should be identity.");
    assert(filter.getControlMatrix().isZero() && "Default B should be zero.");
    assert(filter.getMeasurementMatrix().isZero() && "Default H should be zero.");

    KF::StateTransitionMatrix F;
    F << 2.0;
    KF::ControlMatrix B;
    B << 1.0;
    KF::MeasurementMatrix H;
    H << 0.5;
    filter.setStateTransitionMatrix(F);
    filter.setControlMatrix(B);
    filter.setMeasurementMatrix(H);

    assert(filter.getStateTransitionMatrix().isApprox(F) && "F should match.");
    assert(filter.getControlMatrix().isApprox(B) && "B should match.");
    assert(filter.getMeasurementMatrix().isApprox(H) && "H should match.");
}

// ---------------------------------------------------------------------------
// 1D: one predict + one update
//
//   F = 2, x0 = 1, P0 = 1, Q = 0, H = 0.5, R = 1, z = 1.5
//   predict: x = 2,  P = 4
//   update:  y = 1.5 - 0.5 * 2       = 0.5
//            S = 0.5 * 4 * 0.5 + 1   = 2
//            K = 4 * 0.5 / 2         = 1
//            x = 2 + 1 * 0.5         = 2.5
//            P = (1 - 0.5) * 4 * (1 - 0.5) + 1 * 1 * 1 = 2
// ---------------------------------------------------------------------------
void testScalarPredictUpdate() {
    using KF = kalman::KalmanFilter<double, 1, 1>;
    KF filter;

    KF::StateTransitionMatrix F;
    F << 2.0;
    KF::MeasurementMatrix H;
    H << 0.5;
    filter.setStateTransitionMatrix(F);
    filter.setMeasurementMatrix(H);

    KF::State x0;
    x0 << 1.0;
    KF::StateCovariance P0;
    P0 << 1.0;
    filter.init(x0, P0);

    filter.predict();
    assert(near(filter.getState()(0), 2.0) && "1D: predicted x should be 2.");
    assert(near(filter.getStateCovariance()(0, 0), 4.0) && "1D: predicted P should be 4.");

    KF::Measurement z;
    z << 1.5;
    filter.update(z);
    assert(near(filter.getState()(0), 2.5) && "1D: updated x should be 2.5.");
    assert(near(filter.getStateCovariance()(0, 0), 2.0) && "1D: updated P should be 2.");
}

// ---------------------------------------------------------------------------
// 2 states (position, velocity), 1 measurement (position only)
//
//   F = [1 1; 0 1], H = [1 0], x0 = [0, 1], P0 = I, Q = 0, R = 1, z = 2
//   predict: x = [1, 1]
//            P = F F^T = [2 1; 1 1]
//   update:  y = 2 - 1 = 1
//            S = 2 + 1 = 3
//            K = [2, 1] / 3
//            x = [1 + 2/3, 1 + 1/3] = [5/3, 4/3]
//            P = (I - K H) P = [2/3 1/3; 1/3 2/3]
//
// The velocity is corrected (1 -> 4/3) although it is never measured:
// that comes from the off-diagonal term of P. This case also catches
// dimension mistakes, since n != m.
// ---------------------------------------------------------------------------
void testPositionVelocity() {
    using KF = kalman::KalmanFilter<double, 2, 1>;
    KF filter;

    KF::StateTransitionMatrix F;
    F << 1.0, 1.0,
         0.0, 1.0;
    KF::MeasurementMatrix H;
    H << 1.0, 0.0;
    filter.setStateTransitionMatrix(F);
    filter.setMeasurementMatrix(H);

    KF::State x0;
    x0 << 0.0, 1.0;
    filter.init(x0, KF::StateCovariance::Identity());

    filter.predict();
    KF::State x_pred;
    x_pred << 1.0, 1.0;
    KF::StateCovariance P_pred;
    P_pred << 2.0, 1.0,
              1.0, 1.0;
    assert(filter.getState().isApprox(x_pred, kTol) && "2D: predicted x should be [1, 1].");
    assert(filter.getStateCovariance().isApprox(P_pred, kTol) && "2D: predicted P should be [2 1; 1 1].");

    KF::Measurement z;
    z << 2.0;
    filter.update(z);
    KF::State x_upd;
    x_upd << 5.0 / 3.0, 4.0 / 3.0;
    KF::StateCovariance P_upd;
    P_upd << 2.0 / 3.0, 1.0 / 3.0,
             1.0 / 3.0, 2.0 / 3.0;
    assert(filter.getState().isApprox(x_upd, kTol) && "2D: updated x should be [5/3, 4/3].");
    assert(filter.getStateCovariance().isApprox(P_upd, kTol) && "2D: updated P should be [2/3 1/3; 1/3 2/3].");
}

// ---------------------------------------------------------------------------
// predict(u) with a control input (acceleration)
//
//   F = [1 1; 0 1], B = [0.5, 1], x0 = [0, 1], P0 = I, Q = diag(0.5, 0.25), u = 2
//   x = F x0 + B u     = [1, 1] + [1, 2] = [2, 3]
//   P = F P0 F^T + Q   = [2.5 1; 1 1.25]   (the control input does not change P)
// ---------------------------------------------------------------------------
void testPredictWithControl() {
    using KF = kalman::KalmanFilter<double, 2, 1, 1>;
    KF filter;

    KF::StateTransitionMatrix F;
    F << 1.0, 1.0,
         0.0, 1.0;
    KF::ControlMatrix B;
    B << 0.5,
         1.0;
    filter.setStateTransitionMatrix(F);
    filter.setControlMatrix(B);

    KF::StateCovariance Q;
    Q << 0.5, 0.0,
         0.0, 0.25;
    filter.setProcessNoiseCovariance(Q);

    KF::State x0;
    x0 << 0.0, 1.0;
    filter.init(x0, KF::StateCovariance::Identity());

    KF::Control u;
    u << 2.0;
    filter.predict(u);

    KF::State x_expected;
    x_expected << 2.0, 3.0;
    KF::StateCovariance P_expected;
    P_expected << 2.5, 1.0,
                  1.0, 1.25;
    assert(filter.getState().isApprox(x_expected, kTol) && "Control: predicted x should be [2, 3].");
    assert(filter.getStateCovariance().isApprox(P_expected, kTol) && "Control: predicted P should be [2.5 1; 1 1.25].");
}

// ---------------------------------------------------------------------------
// predict() adds the process noise Q to the covariance
//
//   F = [1 1; 0 1], P0 = I, Q = diag(0.5, 0.25)
//   P = F P0 F^T + Q = [2 1; 1 1] + [0.5 0; 0 0.25] = [2.5 1; 1 1.25]
// ---------------------------------------------------------------------------
void testPredictAddsProcessNoise() {
    using KF = kalman::KalmanFilter<double, 2, 1>;
    KF filter;

    KF::StateTransitionMatrix F;
    F << 1.0, 1.0,
         0.0, 1.0;
    filter.setStateTransitionMatrix(F);

    KF::StateCovariance Q;
    Q << 0.5, 0.0,
         0.0, 0.25;
    filter.setProcessNoiseCovariance(Q);
    filter.init(KF::State::Zero(), KF::StateCovariance::Identity());

    filter.predict();

    KF::StateCovariance P_expected;
    P_expected << 2.5, 1.0,
                  1.0, 1.25;
    assert(filter.getStateCovariance().isApprox(P_expected, kTol) && "Q: predicted P should be [2.5 1; 1 1.25].");
}

// ---------------------------------------------------------------------------
// Many predict/update cycles: P must stay symmetric with a positive diagonal.
// Guards the Joseph-form covariance update against round-off drift.
// ---------------------------------------------------------------------------
void testCovarianceStaysValid() {
    using KF = kalman::KalmanFilter<double, 2, 1>;
    KF filter;

    KF::StateTransitionMatrix F;
    F << 1.0, 0.1,
         0.0, 1.0;
    KF::MeasurementMatrix H;
    H << 1.0, 0.0;
    filter.setStateTransitionMatrix(F);
    filter.setMeasurementMatrix(H);
    filter.setProcessNoiseCovariance(KF::StateCovariance::Identity() * 1e-3);
    filter.init(KF::State::Zero(), KF::StateCovariance::Identity() * 100.0);

    KF::Measurement z;
    for (int k = 1; k <= 1000; ++k) {
        filter.predict();
        z << 0.3 * k;  // object moving at 3 units/s, sampled every 0.1 s
        filter.update(z);

        const KF::StateCovariance& P = filter.getStateCovariance();
        assert(near(P(0, 1), P(1, 0), 1e-9) && "P should stay symmetric.");
        assert(P(0, 0) > 0.0 && P(1, 1) > 0.0 && "P should keep a positive diagonal.");
    }

    // With noiseless measurements of a constant-velocity motion, the
    // velocity estimate must converge to 3.
    assert(near(filter.getState()(1), 3.0, 1e-3) && "Velocity should converge to 3.");
}

}  // namespace

int main() {
    testDefaultsAndSetters();
    testScalarPredictUpdate();
    testPositionVelocity();
    testPredictWithControl();
    testPredictAddsProcessNoise();
    testCovarianceStaysValid();

    std::cout << "All KalmanFilter tests passed" << std::endl;
}
