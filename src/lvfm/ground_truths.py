import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj

class LinearOscillator2D(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, oscillation_speed, u_bound, d_bound, control_mode="min", disturbance_mode="max"):
        self.oscillation_speed = oscillation_speed

        control_space = hj.sets.Box(jnp.array([-u_bound]), jnp.array([u_bound]))
        disturbance_space = hj.sets.Box(jnp.array([-d_bound]), jnp.array([d_bound]))

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)
    
    def open_loop_dynamics(self, state, time):
        x1, x2 = state 
        w = self.oscillation_speed
        return jnp.array([x2, -w**2 * x1])

    def control_jacobian(self, state, time):
        x1, x2 = state
        return jnp.array([[0], [1]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0], [1]])

class Air3DRelative(hj.ControlAndDisturbanceAffineDynamics):
    def __init__(self, vp, ve, u_bound, d_bound, control_mode="max", disturbance_mode="min"):
        self.vp = vp
        self.ve = ve

        control_space = hj.sets.Box(jnp.array([-u_bound]), jnp.array([u_bound]))
        disturbance_space = hj.sets.Box(jnp.array([-d_bound]), jnp.array([d_bound]))

        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        xr, yr, theta = state
        return jnp.array([
            -self.ve + self.vp * jnp.cos(theta),
            self.vp * jnp.sin(theta),
            0.0,
        ])

    def control_jacobian(self, state, time):
        xr, yr, theta = state
        return jnp.array([
            [yr],
            [-xr],
            [-1.0],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.0],
            [0.0],
            [1.0],
        ])

class Dubins3DAvoid(hj.ControlAndDisturbanceAffineDynamics):
    """Single Dubins car avoiding a circle: xdot=v cos th, ydot=v sin th, thetadot=u,
    u in [-omega_max, omega_max]. Avoid BRT: control maximizes value (stays safe)."""
    def __init__(self, velocity, omega_max, control_mode="max", disturbance_mode="min"):
        self.velocity = velocity
        control_space = hj.sets.Box(jnp.array([-omega_max]), jnp.array([omega_max]))
        disturbance_space = hj.sets.Box(jnp.array([0.0]), jnp.array([0.0]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x, y, theta = state
        return jnp.array([self.velocity * jnp.cos(theta), self.velocity * jnp.sin(theta), 0.0])

    def control_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0], [1.0]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0], [0.0]])
