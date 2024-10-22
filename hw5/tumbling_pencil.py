import numpy as np

g = np.array([0, 0, -9.8])  # gravity vector
m = 1
r = 0.5
h1 = 3
h2 = 0.5
t1 = 0.4
t2 = 0.8
dt = 0.1  # Time step for simulation
h = (6*h1**2 + 12*h1*h2 + 3*h2**2)/(12*h1 + 4*h2)

p1 = np.array([h * np.cos(np.pi/3), 0, h * np.sin(np.pi/3)])
v_minus_1 = np.array([-5 * np.cos(np.pi/6), 0, -5 * np.sin(np.pi/6)])
w_minus_1 = np.array([1.0, 5, 0.5])
v_plus_1 = np.array([-1.80954, -0.546988, 1.2076])
w_plus_1 = np.array([0.09957, -0.04174, 0.5])

def quaternion_rotation(q, v):
    q0 = q[0]
    q_vec = np.array(q[1:])
    v_rot = (q0**2 - np.dot(q_vec, q_vec)) * v + 2 * np.dot(q_vec, v) * q_vec + 2 * q0 * np.cross(q_vec, v)
    return v_rot

def update_quaternion(q, w, dt):
    w_norm = np.linalg.norm(w)
    delta_phi = w_norm * dt
    if w_norm != 0:
        w_hat = w / w_norm
        r = np.hstack((np.cos(delta_phi / 2), np.sin(delta_phi / 2) * w_hat))
        q = np.hstack((r[0] * q[0] - np.dot(r[1:], q[1:]), r[0] * q[1:] + q[0] * r[1:] + np.cross(r[1:], q[1:])))
    return q / np.linalg.norm(q)  # Normalize quaternion

def tumble(v_init, w_init,t_start, t_end, dt):
    q = np.array([np.cos(np.pi/12), 0, np.sin(np.pi/12), 0])  # Initial quaternion for pencil orientation
    p = p1.copy()
    v = v_init.copy()
    w = w_init.copy()
    
    time = t_start
    results = []
    
    while time <= t_end:
        # Store position, velocity, and angular velocity
        results.append((time, p.copy(), v.copy(), w.copy()))
        
        # Update velocity with gravity
        v = v + g * dt
        
        # Update angular velocity using Euler's equation
        w_dot = np.linalg.inv(Q) @ (-np.cross(w, Q @ w))
        w = w + w_dot * dt
        
        # Update position
        p = p + v * dt
        
        q = update_quaternion(q, w, dt)
        
        time += dt

    return results

# Moment of inertia matrix (Q) - simplified for diagonal matrix
Q = np.diag([m * (r**2 + h1**2 / 12), m * (r**2 + h1**2 / 12), m * r**2 / 2])

results_before = tumble(v_minus_1, w_minus_1, 0.0, t1, dt)
results_after = tumble(v_plus_1, w_plus_1, t1, t2, dt)

results = results_before + results_after

for i in range(len(results)):
    if i % int(0.1 / dt) == 0:
        time, pos, vel, ang_vel = results[i]
        print(f"{time:.1f} Position: {pos} Velocity: {vel} Angular Velocity: {ang_vel}")
