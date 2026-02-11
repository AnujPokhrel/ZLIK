import random, math
import pdb
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class SegmentParams:
    mode: str          # "CTRV" or "CTRA"
    accln: float           # const accel (CTRV: near 0 with tiny noise)
    omega: float       # const yaw rate
    hold_s: float      # duration


class CTRV_CTRA_Driver:
    def __init__(self, cfg, veh_model):
        self.cfg = cfg
        vp = cfg["vehicle_params"].get(veh_model, cfg["vehicle_params"]["_default"])

        self.L = vp["L"]
        self.max_steer = vp["max_steer_rad"]

        # Initialize state
        self.v = 0.0     # m/s
        self.yaw = 0.0   # rad

        self.seg = self._sample_segment(step_no=0)
        self.seg_t = 0.0

        self.dt = float(1/cfg["sim_frequency"])

    def _sample_segment(self, step_no) -> SegmentParams:
        if step_no <= 90:
            mode = "CTRA"
            accln = random.uniform(*self.cfg["a_bounds"]) 
        elif random.random() < self.cfg["mode_probs"]["CTRV"]:
            mode = "CTRV"
            accln = 0.0
        else:
            mode = "CTRA"            
            accln = random.uniform(self.cfg["a_bounds"][0], self.cfg["a_bounds"][1])

        omega = random.uniform(self.cfg["omega_bounds"][0], self.cfg["omega_bounds"][1])
        hold = np.random.exponential(0.5 / self.cfg["segment_rate_hz"])
        print(f"[INFO] Sampled segment: mode={mode}, a={accln:.2f} m/s^2 o={omega:.2f} rad/s, hold: {hold}s")
        return SegmentParams(mode, accln, omega, hold)

    def step(self, step_no: int):
        # Resample segment on expiry
        self.seg_t += self.dt
        if self.seg_t >= self.seg.hold_s:
            self.seg = self._sample_segment(step_no)
            self.seg_t = 0.0

        # Noisy parameters each step
        accln = self.seg.accln + np.random.randn() * self.cfg["a_sigma"]
        if self.seg.mode == "CTRV":
            accln += np.random.randn() * (0.15 * self.cfg["a_sigma"])  # tiny accel noise #So that this not constant for constant 

        omega = self.seg.omega + np.random.randn() * self.cfg["omega_sigma"]

        # Integrate simple kinematics (world frame)
        self.v = np.clip(self.v + accln * self.dt, *self.cfg["v_bounds"])
        self.yaw += omega * self.dt

        return dict(v=self.v, omega=omega, accln=accln, yaw=self.yaw)

class SpeedPID:
    def __init__(self, kp=0.5, ki=0.05, kd=0.1, dt=0.05, throttle_bounds=(0.0, 1.0), brake_bounds = (0.0, 1.0)):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.dt = dt
        self.min_throttle, self.max_throttle = throttle_bounds
        self.min_brake, self.max_brake = brake_bounds

        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_err = 0.0
    
    def step(self, v_target, v_current):
        error = v_target - v_current
        self.integral += error * self.dt
        derivative = (error - self.prev_err) / self.dt
        self.prev_err = error

        u = self.kp * error + self.ki * self.integral + self.kd * derivative

        if u >= 0:
            throttle = float(np.clip(u, self.min_throttle, self.max_throttle))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(u, self.min_brake, self.max_brake))
        return throttle, brake