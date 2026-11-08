// Tests for kalman::UnscentedKalmanFilter.
//
// Expected values are computed by hand (shown in the comments) or taken from
// the linear KalmanFilter, never read back from the UKF itself.

// Keep assert() active even in Release builds: these are tests.
#undef NDEBUG
#include <kalman/kalman_filter.hpp>
#include <kalman/unscented_kalman_filter.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

bool near(double a, double b, double tol) { return std::abs(a - b) < tol; }

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
// Sigma points must reproduce the mean and covariance they were drawn from:
//   sum Wm_i chi_i = x    and    sum Wc_i (chi_i - x)(chi_i - x)^T = P
// With f(x) = x and Q = 0, predict() rebuilds x and P from the sigma points,
// so both must come back unchanged. Checked for several alpha values; with the
// default alpha = 1e-3 the weights are about +-1e6 and some precision is lost.
// ---------------------------------------------------------------------------
void testSigmaPointsReproduceMeanAndCovariance() {
    using UKF = kalman::UnscentedKalmanFilter<double, 3, 1>;

    UKF::State x0;
    x0 << 1.0, 2.0, 3.0;
    UKF::StateCovariance P0;
    P0 << 4.0, 1.0, 0.5,
          1.0, 3.0, 0.2,
          0.5, 0.2, 2.0;

    struct Case { double alpha; double tol; };
    for (const Case c : {Case{1e-3, 1e-8}, Case{0.5, 1e-12}, Case{1.0, 1e-12}}) {
        UKF filter;
        filter.setParameters(c.alpha, 2.0, 0.0);
        filter.setStateTransitionFunction([](const UKF::State& x, const UKF::Control&) { return x; });
        filter.init(x0, P0);

        filter.predict();
        assert(filter.getState().isApprox(x0, c.tol) && "Sigma points should reproduce the mean.");
        assert(filter.getStateCovariance().isApprox(P0, c.tol) && "Sigma points should reproduce the covariance.");
    }
}

// ---------------------------------------------------------------------------
// With a linear model the UKF must give the same result as the linear KF,
// for any alpha, over several predict/update cycles (Q and R not zero).
// The first case uses the default parameters, so it also checks that the
// constructor computes the weights.
// ---------------------------------------------------------------------------
void testLinearModelMatchesKalmanFilter() {
    using UKF = kalman::UnscentedKalmanFilter<double, 2, 1>;
    using KF = kalman::KalmanFilter<double, 2, 1>;

    KF::StateTransitionMatrix F;
    F << 1.0, 1.0,
         0.0, 1.0;
    KF::MeasurementMatrix H;
    H << 1.0, 0.0;
    const KF::StateCovariance Q = KF::StateCovariance::Identity() * 0.1;
    KF::MeasurementCovariance R;
    R << 0.5;
    KF::State x0;
    x0 << 0.0, 1.0;

    struct Case { bool use_defaults; double alpha; double tol; };
    for (const Case c : {Case{true, 1e-3, 1e-8}, Case{false, 0.5, 1e-12}, Case{false, 1.0, 1e-12}}) {
        UKF ukf;
        if (!c.use_defaults) ukf.setParameters(c.alpha, 2.0, 0.0);
        ukf.setStateTransitionFunction([F](const UKF::State& x, const UKF::Control&) -> UKF::State { return F * x; });
        ukf.setMeasurementFunction([H](const UKF::State& x) -> UKF::Measurement { return H * x; });
        ukf.setProcessNoiseCovariance(Q);
        ukf.setMeasurementNoiseCovariance(R);
        ukf.init(x0, KF::StateCovariance::Identity());

        KF kf;
        kf.setStateTransitionMatrix(F);
        kf.setMeasurementMatrix(H);
        kf.setProcessNoiseCovariance(Q);
        kf.setMeasurementNoiseCovariance(R);
        kf.init(x0, KF::StateCovariance::Identity());

        for (int k = 0; k < 10; ++k) {
            ukf.predict();
            kf.predict();
            assert(ukf.getState().isApprox(kf.getState(), c.tol) && "Linear: predicted x should match the KF.");
            assert(ukf.getStateCovariance().isApprox(kf.getStateCovariance(), c.tol) &&
                   "Linear: predicted P should match the KF.");

            KF::Measurement z;
            z << 1.0 + 0.9 * k;
            ukf.update(z);
            kf.update(z);
            assert(ukf.getState().isApprox(kf.getState(), c.tol) && "Linear: updated x should match the KF.");
            assert(ukf.getStateCovariance().isApprox(kf.getStateCovariance(), c.tol) &&
                   "Linear: updated P should match the KF.");
        }
    }
}

// ---------------------------------------------------------------------------
// Nonlinear predict: f(x) = x^2 with x ~ N(2, 1). The exact moments are
//   E[x^2]   = mu^2 + sigma^2 = 5
//   Var[x^2] = 4 mu^2 sigma^2 + 2 sigma^4 = 18
// (the EKF would give mean 4 and variance 16).
//
// With alpha = 1, kappa = 0: sigma points 1, 2, 3; Wm = [0, 1/2, 1/2].
//   mean = (1 + 9) / 2 = 5
//   beta = 2: Wc0 = 2  ->  var = 2 * (4-5)^2 + ((1-5)^2 + (9-5)^2) / 2 = 18
//   beta = 0: Wc0 = 0  ->  var = 16
// The second case also checks that setParameters() recomputes the weights.
// ---------------------------------------------------------------------------
void testSquareMatchesExactMoments() {
    using UKF = kalman::UnscentedKalmanFilter<double, 1, 1>;

    UKF::State x0;
    x0 << 2.0;
    UKF::StateCovariance P0;
    P0 << 1.0;

    struct Case { double beta; double variance; };
    for (const Case c : {Case{2.0, 18.0}, Case{0.0, 16.0}}) {
        UKF filter;
        filter.setParameters(1.0, c.beta, 0.0);
        filter.setStateTransitionFunction([](const UKF::State& x, const UKF::Control&) -> UKF::State {
            return x.array().square().matrix();
        });
        filter.init(x0, P0);

        filter.predict();
        assert(near(filter.getState()(0), 5.0, 1e-12) && "x^2: predicted mean should be exactly 5.");
        assert(near(filter.getStateCovariance()(0, 0), c.variance, 1e-12) &&
               "x^2: predicted variance should be 18 (beta = 2) or 16 (beta = 0).");
    }
}

// ---------------------------------------------------------------------------
// predict(u) passes u to f; predict() passes u = 0.
//
//   f(x, u) = x + u, x0 = 1, P0 = 1, Q = 0
//   predict(u = 2): x = 3, P = 1      predict(): x = 3, P = 1
// ---------------------------------------------------------------------------
void testControlIsPassedToModel() {
    using UKF = kalman::UnscentedKalmanFilter<double, 1, 1, 1>;
    UKF filter;
    filter.setParameters(1.0, 2.0, 0.0);
    filter.setStateTransitionFunction([](const UKF::State& x, const UKF::Control& u) -> UKF::State {
        return x + u;
    });

    UKF::State x0;
    x0 << 1.0;
    UKF::StateCovariance P0;
    P0 << 1.0;
    filter.init(x0, P0);

    UKF::Control u;
    u << 2.0;
    filter.predict(u);
    assert(near(filter.getState()(0), 3.0, 1e-12) && "Control: predicted x should be 3.");
    assert(near(filter.getStateCovariance()(0, 0), 1.0, 1e-12) && "Control: predicted P should stay 1.");

    filter.predict();
    assert(near(filter.getState()(0), 3.0, 1e-12) && "predict() should pass u = 0 to f.");
}

// ---------------------------------------------------------------------------
// Range and bearing measurements of a fixed point (same scenario as the EKF
// test): the estimate must approach the true position and P must stay valid.
// ---------------------------------------------------------------------------
void testRangeBearingConverges() {
    using UKF = kalman::UnscentedKalmanFilter<double, 2, 2>;
    UKF filter;
    filter.setParameters(1.0, 2.0, 0.0);

    filter.setStateTransitionFunction([](const UKF::State& x, const UKF::Control&) { return x; });
    auto h = [](const UKF::State& x) -> UKF::Measurement {
        UKF::Measurement z;
        z << x.norm(), std::atan2(x(1), x(0));
        return z;
    };
    filter.setMeasurementFunction(h);
    filter.setMeasurementNoiseCovariance(UKF::MeasurementCovariance::Identity() * 0.01);

    UKF::State truth;
    truth << 3.0, 4.0;
    UKF::State guess;
    guess << 2.0, 5.0;
    filter.init(guess, UKF::StateCovariance::Identity());

    const UKF::Measurement z = h(truth);  // noiseless measurements
    const double initial_error = (guess - truth).norm();
    for (int k = 0; k < 50; ++k) {
        filter.predict();
        filter.update(z);

        const UKF::StateCovariance& P = filter.getStateCovariance();
        assert(near(P(0, 1), P(1, 0), 1e-9) && "Range-bearing: P should stay symmetric.");
        assert(P(0, 0) > 0.0 && P(1, 1) > 0.0 && "Range-bearing: P should keep a positive diagonal.");
    }

    const double final_error = (filter.getState() - truth).norm();
    assert(final_error < initial_error / 10.0 && "Range-bearing: error should shrink by at least 10x.");
    assert(final_error < 0.05 && "Range-bearing: estimate should end within 0.05 of [3, 4].");
}

// ---------------------------------------------------------------------------
// Errors: missing functions, P not positive definite (no Cholesky), and a
// singular innovation covariance (h constant and R = 0 -> Pzz = 0).
// ---------------------------------------------------------------------------
void testErrors() {
    using UKF = kalman::UnscentedKalmanFilter<double, 2, 1>;
    UKF::Measurement z;
    z << 1.0;

    {
        UKF filter;  // nothing set
        assert(throwsRuntimeError([&] { filter.predict(); }) && "predict() without f should throw.");
        assert(throwsRuntimeError([&] { filter.update(z); }) && "update() without h should throw.");
    }
    {
        UKF filter;
        filter.setStateTransitionFunction([](const UKF::State& x, const UKF::Control&) { return x; });
        filter.init(UKF::State::Zero(), UKF::StateCovariance::Zero());
        assert(throwsRuntimeError([&] { filter.predict(); }) && "predict() with P = 0 should throw.");
        filter.init(UKF::State::Zero(), -UKF::StateCovariance::Identity());
        assert(throwsRuntimeError([&] { filter.predict(); }) && "predict() with P negative should throw.");
    }
    {
        UKF filter;
        filter.setMeasurementFunction([](const UKF::State&) { return UKF::Measurement::Zero().eval(); });
        filter.setMeasurementNoiseCovariance(UKF::MeasurementCovariance::Zero());
        assert(throwsRuntimeError([&] { filter.update(z); }) && "update() with Pzz = 0 should throw.");
    }
}

}  // namespace

int main() {
    testSigmaPointsReproduceMeanAndCovariance();
    testLinearModelMatchesKalmanFilter();
    testSquareMatchesExactMoments();
    testControlIsPassedToModel();
    testRangeBearingConverges();
    testErrors();

    std::cout << "All UnscentedKalmanFilter tests passed" << std::endl;
}
