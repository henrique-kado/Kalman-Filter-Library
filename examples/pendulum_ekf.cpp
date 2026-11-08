// Example: estimating the angle and angular velocity of a pendulum released
// from a large angle, measuring only a noisy angle (extended Kalman filter).
//
//   state        x = [theta, omega]       (angle [rad], angular velocity [rad/s])
//   dynamics     theta' = omega
//                omega' = -(g / L) sin(theta)
//   measurement  z = theta + noise
//
// Why not the linear filter? For small angles sin(theta) ~ theta and the model
// becomes linear, but at theta = 1 rad that approximation is off by about 16%.
// The EKF uses the real sin(theta) and linearizes it at every step.

#include <kalman/kalman.hpp>

#include <cmath>
#include <cstdio>
#include <random>

namespace {

using EKF = kalman::ExtendedKalmanFilter<double, 2, 1>;  // 2 states, 1 measurement

constexpr double g = 9.81;   // gravity [m/s^2]
constexpr double L = 1.0;    // pendulum length [m]
constexpr double dt = 0.01;  // time step [s]

// Discretized dynamics: semi-implicit (symplectic) Euler step.
//   omega' = omega - (g / L) sin(theta) dt
//   theta' = theta + omega' dt          <- uses the NEW omega
// The plain Euler step (theta' = theta + omega dt) adds a little energy at
// every step, so the swing would slowly grow; this variant keeps it steady.
// Used both by the filter and to simulate the true pendulum, so any difference
// comes from the filter, not from the model.
EKF::State pendulumStep(const EKF::State& x) {
    const double theta = x(0);
    const double omega = x(1);
    const double omega_next = omega - (g / L) * std::sin(theta) * dt;
    EKF::State next;
    next << theta + omega_next * dt,
            omega_next;
    return next;
}

// Jacobian of pendulumStep with respect to the state. With
// a = -(g / L) cos(theta) dt  (the derivative of omega' with respect to theta):
//   d(omega')/d(theta) = a              d(omega')/d(omega) = 1
//   d(theta')/d(theta) = 1 + a dt       d(theta')/d(omega) = dt
EKF::StateTransitionMatrix pendulumJacobian(const EKF::State& x) {
    const double a = -(g / L) * std::cos(x(0)) * dt;
    EKF::StateTransitionMatrix F;
    F << 1.0 + a * dt, dt,
         a,            1.0;
    return F;
}

}  // namespace

int main() {
    // ---- Scenario ---------------------------------------------------------
    const int steps = 500;           // 5 seconds
    const double meas_sigma = 0.1;   // angle sensor noise [rad] (~6 degrees)

    EKF::State truth;
    truth << 1.0, 0.0;               // released from 1 rad (~57 degrees), at rest

    // ---- Filter -----------------------------------------------------------
    EKF filter;
    filter.setStateTransitionFunction(
        [](const EKF::State& x, const EKF::Control&) { return pendulumStep(x); });
    filter.setStateTransitionJacobian(
        [](const EKF::State& x, const EKF::Control&) { return pendulumJacobian(x); });

    // The angle is measured directly: h(x) = theta, a linear measurement.
    // The EKF is still needed because of the nonlinear dynamics.
    EKF::MeasurementMatrix H;
    H << 1.0, 0.0;
    filter.setMeasurementFunction([H](const EKF::State& x) -> EKF::Measurement { return H * x; });
    filter.setMeasurementJacobian([H](const EKF::State&) { return H; });

    EKF::StateCovariance Q;
    Q << 1e-6, 0.0,
         0.0,  1e-4;
    filter.setProcessNoiseCovariance(Q);

    EKF::MeasurementCovariance R;
    R << meas_sigma * meas_sigma;
    filter.setMeasurementNoiseCovariance(R);

    // Deliberately wrong initial guess: half the true angle.
    EKF::State guess;
    guess << 0.5, 0.0;
    filter.init(guess, EKF::StateCovariance::Identity());

    // ---- Simulation -------------------------------------------------------
    std::mt19937 rng(42);  // fixed seed: same output on every run
    std::normal_distribution<double> noise(0.0, meas_sigma);

    double sq_err_measurement = 0.0;
    double sq_err_theta = 0.0;
    double sq_err_omega = 0.0;

    std::printf("%6s %10s %10s %10s %10s %10s\n",
                "t [s]", "true th", "measured", "est th", "true om", "est om");

    for (int k = 1; k <= steps; ++k) {
        truth = pendulumStep(truth);

        EKF::Measurement z;
        z << truth(0) + noise(rng);

        filter.predict();
        filter.update(z);

        const EKF::State& est = filter.getState();
        sq_err_measurement += std::pow(z(0) - truth(0), 2);
        sq_err_theta += std::pow(est(0) - truth(0), 2);
        sq_err_omega += std::pow(est(1) - truth(1), 2);

        if (k % 25 == 0) {
            std::printf("%6.2f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
                        k * dt, truth(0), z(0), est(0), truth(1), est(1));
        }
    }

    std::printf("\nRMS error\n");
    std::printf("  angle, raw measurements: %.3f rad\n", std::sqrt(sq_err_measurement / steps));
    std::printf("  angle, EKF:              %.3f rad\n", std::sqrt(sq_err_theta / steps));
    std::printf("  angular velocity, EKF:   %.3f rad/s (never measured)\n",
                std::sqrt(sq_err_omega / steps));
    return 0;
}
