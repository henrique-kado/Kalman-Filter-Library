add_executable(constant_velocity constant_velocity.cpp)
target_link_libraries(constant_velocity PRIVATE kalman::kalman)

add_executable(pendulum_ekf pendulum_ekf.cpp)
target_link_libraries(pendulum_ekf PRIVATE kalman::kalman)
