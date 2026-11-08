# Kalman-Filter-Library

Biblioteca header-only de filtros de Kalman em C++17, construída sobre o [Eigen](https://eigen.tuxfamily.org).

Três filtros com a mesma interface:

| Filtro | Classe | Quando usar |
|---|---|---|
| Kalman linear (KF) | `kalman::KalmanFilter` | modelo e medidas lineares: `x = F x + B u`, `z = H x` |
| Kalman estendido (EKF) | `kalman::ExtendedKalmanFilter` | modelo não linear, com jacobianas que você sabe calcular |
| Kalman unscented (UKF) | `kalman::UnscentedKalmanFilter` | modelo não linear, sem jacobianas; costuma ser mais preciso que o EKF quando a não linearidade é forte |

Todos herdam de `kalman::FilterBase`, então código escrito para um filtro funciona com os outros: você configura o modelo uma vez e depois só chama `predict()` e `update(z)`.

## Requisitos

- Compilador com suporte a C++17
- CMake ≥ 3.16
- Eigen ≥ 3.3

Instalação do Eigen:

```bash
sudo apt install libeigen3-dev     # Ubuntu / Debian
sudo dnf install eigen3-devel      # Fedora
brew install eigen                 # macOS
```

## Exemplo rápido

Rastrear um objeto em linha reta medindo só a posição. O estado é `[posição, velocidade]`:

```cpp
#include <kalman/kalman.hpp>
#include <iostream>

int main() {
    using KF = kalman::KalmanFilter<double, 2, 1>;   // 2 estados, 1 medida
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

    std::cout << "estado:\n" << filter.getState() << "\n";
    std::cout << "covariância:\n" << filter.getStateCovariance() << "\n";
}
```

## Os parâmetros de template

```cpp
kalman::KalmanFilter<Scalar, N, M, C>
```

| Parâmetro | Significado |
|---|---|
| `Scalar` | tipo de ponto flutuante: `float`, `double` ou `long double` |
| `N` | dimensão do estado |
| `M` | dimensão da medida |
| `C` | dimensão da entrada de controle (padrão `0`, sem controle) |

Cada classe expõe os tipos de que você precisa, com as dimensões certas: `State` (N×1), `StateCovariance` (N×N), `Measurement` (M×1), `MeasurementCovariance` (M×M) e `Control` (C×1). Como as dimensões são fixas na compilação, multiplicar matrizes de tamanhos incompatíveis é um erro de compilação, e não um bug em tempo de execução.

## Interface comum (`FilterBase`)

| Método | O que faz |
|---|---|
| `init(x0, P0)` | define o estado inicial e a sua covariância |
| `setProcessNoiseCovariance(Q)` | ruído do modelo (N×N) |
| `setMeasurementNoiseCovariance(R)` | ruído da medida (M×M) |
| `predict()` | propaga o estado um passo, sem controle |
| `predict(u)` | propaga o estado um passo com a entrada de controle `u` |
| `update(z)` | corrige o estado com a medida `z` |
| `getState()`, `getStateCovariance()` | estimativa atual e sua incerteza |
| `getProcessNoiseCovariance()`, `getMeasurementNoiseCovariance()` | `Q` e `R` atuais |

Valores padrão: `x = 0`, `P = I`, `Q = 0`, `R = I`.

Como os três filtros têm a mesma interface, uma função pode receber qualquer um deles:

```cpp
template <typename Scalar, int N, int M, int C>
void step(kalman::FilterBase<Scalar, N, M, C>& filter,
          const typename kalman::FilterBase<Scalar, N, M, C>::Measurement& z) {
    filter.predict();
    filter.update(z);
}
```

## Filtro de Kalman linear

```
x = F x + B u + w,    w ~ N(0, Q)
z = H x + v,          v ~ N(0, R)
```

| Método | Matriz | Padrão |
|---|---|---|
| `setStateTransitionMatrix(F)` | N×N | identidade |
| `setControlMatrix(B)` | N×C | zero |
| `setMeasurementMatrix(H)` | M×N | zero |

Também há getters para as três matrizes. A covariância é atualizada na forma de Joseph, que mantém `P` simétrica e positiva semidefinida.

Com entrada de controle (aqui, uma aceleração):

```cpp
using KF = kalman::KalmanFilter<double, 2, 1, 1>;   // 2 estados, 1 medida, 1 controle
KF filter;

KF::ControlMatrix B;
B << 0.5 * dt * dt,
     dt;
filter.setControlMatrix(B);

KF::Control u;
u << 2.0;          // aceleração de 2 m/s²
filter.predict(u);
```

## Filtro de Kalman estendido

```
x = f(x, u) + w
z = h(x) + v
```

Você fornece `f`, `h` e as jacobianas `∂f/∂x` e `∂h/∂x` como funções (qualquer coisa que um `std::function` aceite, inclusive lambdas):

| Método | Assinatura da função |
|---|---|
| `setStateTransitionFunction(f)` | `State f(const State& x, const Control& u)` |
| `setStateTransitionJacobian(F)` | `StateTransitionMatrix F(const State& x, const Control& u)` |
| `setMeasurementFunction(h)` | `Measurement h(const State& x)` |
| `setMeasurementJacobian(H)` | `MeasurementMatrix H(const State& x)` |

As jacobianas são recalculadas a cada passo: a de `f` no estado anterior ao `predict`, e a de `h` no estado previsto, antes da correção.

Exemplo: um radar na origem mede a distância até um objeto em `(px, py)`:

```cpp
using EKF = kalman::ExtendedKalmanFilter<double, 2, 1>;
EKF filter;

// O objeto está parado.
filter.setStateTransitionFunction([](const EKF::State& x, const EKF::Control&) { return x; });
filter.setStateTransitionJacobian([](const EKF::State&, const EKF::Control&) {
    return EKF::StateTransitionMatrix::Identity().eval();
});

// h(x) = distância até a origem;  ∂h/∂x = [px/r, py/r]
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

Chamar `predict` ou `update` sem as funções necessárias lança `std::runtime_error`.

## Filtro de Kalman unscented

Mesma ideia do EKF, mas sem jacobianas: o filtro passa 2N+1 pontos (*sigma points*) pela função não linear e reconstrói a média e a covariância a partir deles.

| Método | O que faz |
|---|---|
| `setStateTransitionFunction(f)` | igual ao EKF |
| `setMeasurementFunction(h)` | igual ao EKF |
| `setParameters(alpha, beta, kappa)` | espalhamento dos sigma points |

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

Parâmetros (padrão `alpha = 1e-3`, `beta = 2`, `kappa = 0`):
- **`alpha`** controla o quanto os pontos se afastam da média. Valores pequenos, como o padrão, geram pesos da ordem de ±10⁶ e perdem alguns dígitos de precisão; valores entre `0.5` e `1` costumam ser mais estáveis.
- **`beta = 2`** é o valor ótimo quando a distribuição é gaussiana.
- **`kappa`** costuma ficar em `0`.

## Erros

Os filtros lançam `std::runtime_error` quando:
- uma função obrigatória (`f`, `h` ou as jacobianas do EKF) não foi definida;
- a covariância da inovação (`S` no KF e no EKF, `Pzz` no UKF) não é positiva definida, por exemplo com `P = 0` e `R = 0`;
- no UKF, `P` não é positiva definida, o que impede gerar os sigma points.

## Como usar no seu projeto

A biblioteca é header-only: não há nada para compilar, basta o caminho de includes e o Eigen.

**Com CMake, como subdiretório:**

```cmake
add_subdirectory(external/Kalman-Filter-Library)
target_link_libraries(meu_app PRIVATE kalman::kalman)
```

**Com CMake, baixando direto do GitHub:**

```cmake
include(FetchContent)
FetchContent_Declare(kalman
    GIT_REPOSITORY https://github.com/henrique-kado/Kalman-Filter-Library.git
    GIT_TAG        main)
FetchContent_MakeAvailable(kalman)

target_link_libraries(meu_app PRIVATE kalman::kalman)
```

**Sem CMake:** copie a pasta `include/kalman` para o seu projeto e adicione `include/` e o Eigen ao caminho de includes.

Depois, um único include traz os três filtros:

```cpp
#include <kalman/kalman.hpp>
```

## Compilar e rodar os testes

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

Os testes ficam em `tests/`, um executável por filtro. Os valores esperados foram calculados à mão e estão nos comentários de cada teste.

## Estrutura

```
include/kalman/
  kalman.hpp                    inclui os três filtros
  filter_base.hpp               classe base abstrata (interface comum)
  kalman_filter.hpp             KF linear
  extended_kalman_filter.hpp    EKF
  unscented_kalman_filter.hpp   UKF
tests/                          testes de cada filtro
```

## Licença

[MIT](LICENSE) © 2026 Henrique Yuiti Kado
