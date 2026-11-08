// Example: tracking an object that moves at constant velocity along a line,
// using only noisy position measurements (linear Kalman filter).
//
//   state        x = [position, velocity]
//   measurement  z = position + noise
//
// The filter starts without knowing the velocity (0) and has to work it out
// from the positions alone.

#include <kalman/kalman.hpp>

#include <cmath>
#include <cstdio>
#include <random>

int main() {
    using KF = kalman::KalmanFilter<double, 2, 1>;  // 2 states, 1 measurement

    // ---- Scenario ---------------------------------------------------------
    const double dt = 0.1;              // sampling period [s]
    const int steps = 50;               // 5 seconds
    const double true_velocity = 2.0;   // [m/s]
    const double meas_sigma = 0.5;      // position sensor noise [m]
    const double accel_sigma = 0.05;    // unmodelled acceleration [m/s^2]

    // ---- Model ------------------------------------------------------------
    // position' = position + velocity * dt
    // velocity' = velocity
    KF::StateTransitionMatrix F;
    F << 1.0, dt,
         0.0, 1.0;

    // Only the position is measured.
    KF::MeasurementMatrix H;
    H << 1.0, 0.0;

    // Process noise: a small random acceleration a ~ N(0, accel_sigma^2) moves
    // the state by G * a, with G = [dt^2 / 2, dt]  ->  Q = G G^T accel_sigma^2.
    KF::State G;
    G << 0.5 * dt * dt, dt;
    const KF::StateCovariance Q = G * G.transpose() * accel_sigma * accel_sigma;

    KF::MeasurementCovariance R;
    R << meas_sigma * meas_sigma;

    // ---- Filter -----------------------------------------------------------
    KF filter;
    filter.setStateTransitionMatrix(F);
    filter.setMeasurementMatrix(H);
    filter.setProcessNoiseCovariance(Q);
    filter.setMeasurementNoiseCovariance(R);

    // Initial guess: at the origin, standing still, with a large uncertainty.
    filter.init(KF::State::Zero(), KF::StateCovariance::Identity() * 10.0);

    // ---- Simulation -------------------------------------------------------
    std::mt19937 rng(42);  // fixed seed: same output on every run
    std::normal_distribution<double> noise(0.0, meas_sigma);

    double true_position = 0.0;
    double sq_err_measurement = 0.0;
    double sq_err_filter = 0.0;

    std::printf("%6s %10s %10s %10s %10s %10s\n",
                "t [s]", "true pos", "measured", "est pos", "est vel", "std pos");

    for (int k = 1; k <= steps; ++k) {
        true_position += true_velocity * dt;

        KF::Measurement z;
        z << true_position + noise(rng);

        filter.predict();
        filter.update(z);

        const double est_position = filter.getState()(0);
        const double est_velocity = filter.getState()(1);
        const double std_position = std::sqrt(filter.getStateCovariance()(0, 0));

        sq_err_measurement += std::pow(z(0) - true_position, 2);
        sq_err_filter += std::pow(est_position - true_position, 2);

        if (k % 5 == 0) {
            std::printf("%6.1f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
                        k * dt, true_position, z(0), est_position, est_velocity, std_position);
        }
    }

    std::printf("\nRMS position error\n");
    std::printf("  raw measurements: %.3f m\n", std::sqrt(sq_err_measurement / steps));
    std::printf("  Kalman filter:    %.3f m\n", std::sqrt(sq_err_filter / steps));
    std::printf("Estimated velocity: %.3f m/s (true: %.1f m/s)\n",
                filter.getState()(1), true_velocity);
    return 0;
}
