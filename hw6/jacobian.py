import numpy as np

from linear_system_solver import Linear_System_Solver

def jacobian(theta1, theta2, theta3, L1, L2, L3):
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c12 = np.cos(theta1 + theta2)
    s12 = np.sin(theta1 + theta2)
    c123 = np.cos(theta1 + theta2 + theta3)
    s123 = np.sin(theta1 + theta2 + theta3)
    
    J = np.array([
        [-L1 * s1 - L2 * s12 - L3 * s123, -L2 * s12 - L3 * s123, -L3 * s123],
        [ L1 * c1 + L2 * c12 + L3 * c123,  L2 * c12 + L3 * c123,  L3 * c123],
        [1, 1, 1]
    ])
    
    return J

L1, L2, L3 = 4, 3, 2
theta1, theta2, theta3 = np.pi / 18, np.pi / 9, np.pi / 6
x_dot, y_dot, theta_dot = 0.2, -0.3, -0.2
dt = 0.1
T = 2.0

theta_values = [(0, theta1, theta2, theta3)]

lss = Linear_System_Solver()

for t in np.arange(0.1, T + dt, dt):
    J = jacobian(theta1, theta2, theta3, L1, L2, L3)
    
    V = np.array([x_dot, y_dot, theta_dot])
    
    L, U, P = lss.LUdcmp(J)
    theta_dots = lss.LUbksub(L, U, P, V)
    
    theta1 += dt * theta_dots[0]
    theta2 += dt * theta_dots[1]
    theta3 += dt * theta_dots[2]
    
    theta_values.append((t, theta1, theta2, theta3))

print("t      theta_1    theta_2    theta_3")
for t, th1, th2, th3 in theta_values:
    print(f"{t:.1f}   {th1:.5f}   {th2:.5f}   {th3:.5f}")