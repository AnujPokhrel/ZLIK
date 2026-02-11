import math, random, os, shutil
import numpy as np
import pdb
import time
from typing import Dict, Tuple, List

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import AdvancedIMU as AIMU  # type: ignoreof
from scipy.spatial.transform import Rotation as R


def yaw_to_quat(yaw: float) -> tuple:
    """Roll=Pitch=0, yaw-only quaternion (x,y,z,w)."""
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))

def quat_to_yaw(qxyzw: Tuple[float, float, float, float]) -> float:
    x, y, z, w = qxyzw
    # yaw from quaternion (assuming z-up); robust enough for spawn
    t3 = 2.0*(w*z + x*y)
    t4 = 1.0 - 2.0*(y*y + z*z)
    return math.atan2(t3, t4) 

def normalize_quat(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    else:
        return tuple((q/n).tolist())

def rand_in(a, b):
    return random.uniform(a, b)


#-------------------Vehicle state/IMU Helpers-------------
def _poll_vehicle_once(vehicle: Vehicle):
    vehicle.sensors.poll()

def _get_imu_world_pos(vehicle: Vehicle):
    _poll_vehicle_once(vehicle)
    imu_pos = vehicle.state['pos'][:]
    imu_pos[2] += 0.5
    
    imu_pos = np.array([0.0, -0.5, 0.0])
    return imu_pos

def get_vehicle_params(cfg, model):
    return cfg["vehicle_params"].get(model, cfg["vehicle_params"]["_default"])

def ensure_dir_for(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def pose_to_rotation_matrix(x, y, z, yaw, pitch, roll, degrees=False):
    """Convert pose (x,y,z,rpy) into a 4x4 homogeneous transform."""
    if degrees:
        r = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
    else:
        r = R.from_euler('zyx', [yaw, pitch, roll])
    T = np.eye(4)
    T[:3,:3] = r.as_matrix()
    T[:3, 3] = [x, y, z]
    return T

def to_robot_6_dof(pose1, pose2, degrees=False):
    T1 = pose_to_rotation_matrix(*pose1, degrees=degrees)
    T2 = pose_to_rotation_matrix(*pose2, degrees=degrees)
    T_relative = np.linalg.inv(T1) @ T2

    t_rel = T_relative[:3, 3]

    r_rel = R.from_matrix(T_relative[:3, :3])
    rpy = r_rel.as_euler('xyz', degrees=degrees)

    six_dof_robot_frame = np.concatenate((t_rel, rpy)).tolist()
    return six_dof_robot_frame


def make_subsample_indices_from_time(time_array, sim_hz, target_hz):
    stride = max(1, int(round(sim_hz / target_hz)))
    idxs = np.arange(0, len(time_array), stride, dtype=int)
    if idxs[-1] != len(time_array) - 1:
        idxs = np.append(idxs, len(time_array) - 1)
    return idxs

def future_robot_deltas_from_now(pose_deltas, hist_steps, K_futures, delay_steps):
    Tm1 = int(pose_deltas.shape[0])     #length of the data stream
    F = int(pose_deltas.shape[1])

    #not enough data to fill history
    if Tm1 < hist_steps:
        return np.zeros((0, hist_steps, pose_deltas.shape[1]), dtype=pose_deltas.dtype), np.zeros((0,), dtype=int)

    #where to start and where to end
    i_start = hist_steps - 1
    i_end = (Tm1 - 1) - K_futures 

    N = i_end - i_start + 1  #number of valid data points
    resultant_windows = np.zeros((Tm1, K_futures, F), dtype=pose_deltas.dtype)
    idx_map = np.zeros((Tm1,), dtype=int)

    for n, i in enumerate(range(i_start, i_end + 1)):
        res_end = i + K_futures 
        resultant_windows[i] = pose_deltas[i:res_end]
        idx_map[n] = i
    
    return resultant_windows, idx_map
