# Kalman-Filter-Library

A header-only Kalman filter library for C++17, built on [Eigen](https://eigen.tuxfamily.org).

Three filters with the same interface:

| Filter | Class | When to use |
|---|---|---|
| Linear Kalman filter (KF) | `kalman::KalmanFilter` | linear model and measurements: `x = F x + B u`, `z = H x` |
| Extended Kalman filter (EKF) | `kalman::ExtendedKalmanFilter` | nonlinear model whose Jacobians you can compute |
| Unscented Kalman filter (UKF) | `kalman::UnscentedKalmanFilter` | nonlinear model, no Jacobians needed; usually more accurate than the EKF when the nonlinearity is strong |

All of them derive from `kalman::FilterBase`, so code written for one filter works with the others: you set up the model once and then just call `predict()` and `update(z)`.

## Requirements

- A C++17 compiler
- CMake ≥ 3.16
- Eigen ≥ 3.3

Installing Eigen:

```bash
sudo apt install libeigen3-dev     # Ubuntu / Debian
sudo dnf install eigen3-devel      # Fedora
brew install eigen                 # macOS
```

## Quick example

Track an object moving along a line, measuring only its position. The state is `[position, velocity]`:

```cpp
#include <kalman/kalman.hpp>
#include <iostream>

int main() {
    using KF = kalman::KalmanFilter<double, 2, 1>;   // 2 states, 1 measurement
    const double dt = 0.1;

    KF filter;

    KF::StateTransitionMatrix F;
    F << 1.0, dt,
         0.0, 1.0;
    KF::MeasurementMatrix H;
    H << 1.0, 0.0;

    KF::MeasurementCovariance R;
    R << 0.25;

    filter.setStateTransitionMatrix(F);
    filter.setMeasurementMatrix(H);
    filter.setProcessNoiseCovariance(KF::StateCovariance::Identity() * 1e-4);
    filter.setMeasurementNoiseCovariance(R);
    filter.init(KF::State::Zero(), KF::StateCovariance::Identity() * 10.0);

    for (double position : {0.21, 0.38, 0.65, 0.79, 1.02}) {
        KF::Measurement z;
        z << position;

        filter.predict();
        filter.update(z);
    }

    std::cout << "state:\n" << filter.getState() << "\n";
    std::cout << "covariance:\n" << filter.getStateCovariance() << "\n";
}
```

## Template parameters

```cpp
kalman::KalmanFilter<Scalar, N, M, C>
```

| Parameter | Meaning |
|---|---|
| `Scalar` | floating-point type: `float`, `double` or `long double` |
| `N` | state dimension |
| `M` | measurement dimension |
| `C` | control input dimension (default `0`, no control) |

Each class exposes the types you need, with the right dimensions: `State` (N×1), `StateCovariance` (N×N), `Measurement` (M×1), `MeasurementCovariance` (M×M) and `Control` (C×1). Since the dimensions are fixed at compile time, multiplying matrices of incompatible sizes is a compile error rather than a runtime bug.

## Common interface (`FilterBase`)

| Method | What it does |
|---|---|
| `init(x0, P0)` | sets the initial state and its covariance |
| `setProcessNoiseCovariance(Q)` | model noise (N×N) |
| `setMeasurementNoiseCovariance(R)` | measurement noise (M×M) |
| `predict()` | propagates the state one step, without control |
| `predict(u)` | propagates the state one step with control input `u` |
| `update(z)` | corrects the state with measurement `z` |
| `getState()`, `getStateCovariance()` | current estimate and its uncertainty |
| `getProcessNoiseCovariance()`, `getMeasurementNoiseCovariance()` | current `Q` and `R` |

Defaults: `x = 0`, `P = I`, `Q = 0`, `R = I`.

Because the three filters share the same interface, a single function can take any of them:

```cpp
template <typename Scalar, int N, int M, int C>
void step(kalman::FilterBase<Scalar, N, M, C>& filter,
          const typename kalman::FilterBase<Scalar, N, M, C>::Measurement& z) {
    filter.predict();
    filter.update(z);
}
```

## Linear Kalman filter

```
x = F x + B u + w,    w ~ N(0, Q)
z = H x + v,          v ~ N(0, R)
```

| Method | Matrix | Default |
|---|---|---|
| `setStateTransitionMatrix(F)` | N×N | identity |
| `setControlMatrix(B)` | N×C | zero |
| `setMeasurementMatrix(H)` | M×N | zero |

There are getters for all three matrices. The covariance is updated in Joseph form, which keeps `P` symmetric and positive semi-definite.

With a control input (here, an acceleration):

```cpp
using KF = kalman::KalmanFilter<double, 2, 1, 1>;   // 2 states, 1 measurement, 1 control
KF filter;

KF::ControlMatrix B;
B << 0.5 * dt * dt,
     dt;
filter.setControlMatrix(B);

KF::Control u;
u << 2.0;          // 2 m/s² acceleration
filter.predict(u);
```

## Extended Kalman filter

```
x = f(x, u) + w
z = h(x) + v
```

You provide `f`, `h` and the Jacobians `∂f/∂x` and `∂h/∂x` as functions (anything a `std::function` accepts, lambdas included):

| Method | Function signature |
|---|---|
| `setStateTransitionFunction(f)` | `State f(const State& x, const Control& u)` |
| `setStateTransitionJacobian(F)` | `StateTransitionMatrix F(const State& x, const Control& u)` |
| `setMeasurementFunction(h)` | `Measurement h(const State& x)` |
| `setMeasurementJacobian(H)` | `MeasurementMatrix H(const State& x)` |

The Jacobians are recomputed at every step: the one of `f` at the state before `predict`, and the one of `h` at the predicted state, before the correction.

Example: a radar at the origin measures the range to an object at `(px, py)`:

```cpp
using EKF = kalman::ExtendedKalmanFilter<double, 2, 1>;
EKF filter;

// The object does not move.
filter.setStateTransitionFunction([](const EKF::State& x, const EKF::Control&) { return x; });
filter.setStateTransitionJacobian([](const EKF::State&, const EKF::Control&) {
    return EKF::StateTransitionMatrix::Identity().eval();
});

// h(x) = range from the origin;  ∂h/∂x = [px/r, py/r]
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
```

Calling `predict` or `update` without the required functions throws `std::runtime_error`.

## Unscented Kalman filter

Same idea as the EKF, but without Jacobians: the filter passes 2N+1 points (*sigma points*) through the nonlinear function and rebuilds the mean and covariance from them.

| Method | What it does |
|---|---|
| `setStateTransitionFunction(f)` | same as the EKF |
| `setMeasurementFunction(h)` | same as the EKF |
| `setParameters(alpha, beta, kappa)` | spread of the sigma points |

```cpp
using UKF = kalman::UnscentedKalmanFilter<double, 2, 1>;
UKF filter;
filter.setParameters(1.0, 2.0, 0.0);
filter.setStateTransitionFunction([](const UKF::State& x, const UKF::Control&) { return x; });
filter.setMeasurementFunction([](const UKF::State& x) -> UKF::Measurement {
    UKF::Measurement z;
    z << x.norm();
    return z;
});
```

Parameters (defaults `alpha = 1e-3`, `beta = 2`, `kappa = 0`):
- **`alpha`** controls how far the points spread from the mean. Small values, like the default, produce weights around ±10⁶ and lose a few digits of precision; values between `0.5` and `1` tend to be more stable.
- **`beta = 2`** is optimal when the distribution is Gaussian.
- **`kappa`** is usually left at `0`.

## Errors

The filters throw `std::runtime_error` when:
- a required function (`f`, `h` or the EKF Jacobians) has not been set;
- the innovation covariance (`S` in the KF and EKF, `Pzz` in the UKF) is not positive definite, for example with `P = 0` and `R = 0`;
- in the UKF, `P` is not positive definite, so the sigma points cannot be generated.

## Using it in your project

The library is header-only: there is nothing to compile, you only need the include path and Eigen.

**With CMake, installing the library (`find_package`):**

```bash
git clone https://github.com/henrique-kado/Kalman-Filter-Library.git
cmake -S Kalman-Filter-Library -B build
sudo cmake --install build                 # installs to /usr/local
```

```cmake
find_package(kalman 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE kalman::kalman)
```

To install somewhere else, use `cmake --install build --prefix ~/libs` and, in your project, `-DCMAKE_PREFIX_PATH=~/libs`. Eigen must also be installed on the machine that uses the library: `find_package(kalman)` looks for it automatically.

**With CMake, as a subdirectory:**

```cmake
add_subdirectory(external/Kalman-Filter-Library)
target_link_libraries(my_app PRIVATE kalman::kalman)
```

**With CMake, downloading straight from GitHub:**

```cmake
include(FetchContent)
FetchContent_Declare(kalman
    GIT_REPOSITORY https://github.com/henrique-kado/Kalman-Filter-Library.git
    GIT_TAG        main)
FetchContent_MakeAvailable(kalman)

target_link_libraries(my_app PRIVATE kalman::kalman)
```

**Without CMake:** copy the `include/kalman` folder into your project and add `include/` and Eigen to the include path.

Then a single include brings in all three filters:

```cpp
#include <kalman/kalman.hpp>
```

## Building and running the tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

The tests live in `tests/`, one executable per filter. The expected values were computed by hand and are shown in the comments of each test.

CMake options:

| Option | Default | What it does |
|---|---|---|
| `KALMAN_BUILD_TESTS` | `ON` when the library is the main project | builds the tests |
| `KALMAN_BUILD_EXAMPLES` | `ON` when the library is the main project | builds the examples |
| `KALMAN_INSTALL` | `ON` when the library is the main project | generates the install rules |

When the library is pulled into another project with `add_subdirectory` or `FetchContent`, all three are off: neither the tests, the examples nor the library's install rules end up in the consumer's project.

## Examples

Each example simulates the real system and a noisy sensor (with a fixed seed, so the output is always the same), runs the filter, and prints a table plus the RMS error at the end. After building:

```bash
./build/examples/constant_velocity
./build/examples/pendulum_ekf
```

| Example | Filter | What it shows |
|---|---|---|
| [`constant_velocity.cpp`](examples/constant_velocity.cpp) | KF | An object moving along a line at 2 m/s, with only its position measured (0.5 m noise). The filter starts without knowing the velocity and works it out from the positions alone; the position error drops from 0.47 m (raw measurements) to 0.19 m. |
| [`pendulum_ekf.cpp`](examples/pendulum_ekf.cpp) | EKF | A pendulum released from 1 rad, with only its angle measured (0.1 rad noise). The model is nonlinear (sin θ); the angle error drops from 0.10 rad to 0.017 rad, and the angular velocity, which is never measured, is estimated with a 0.07 rad/s error. |

## Layout

```
include/kalman/
  kalman.hpp                    includes all three filters
  filter_base.hpp               abstract base class (common interface)
  kalman_filter.hpp             linear KF
  extended_kalman_filter.hpp    EKF
  unscented_kalman_filter.hpp   UKF
tests/                          tests for each filter
examples/                       usage examples
cmake/                          template of the file used by find_package
```

## License

[MIT](LICENSE) © 2026 Henrique Yuiti Kado
