# Utility Script to launch BeamNG, spawn vehicle, attach Advanced IMU sensor.
# Also includes damage application functions (tyre puncture, shock/spring break, axle break


import math, random, os, shutil
import numpy as np
import pdb
import time
from typing import Dict, Tuple, List

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamngpy import BeamNGpy, Scenario, Vehicle 
from beamngpy.sensors import AdvancedIMU as AIMU, GPS, Damage as DamageSensor  # type: ignoreof
from scipy.spatial.transform import Rotation as R
import DataGeneration.data_utilities as DU

def bng_init(cfg, model, color, damage_choice: str):
    """
    Creates/opens BeamNG, builds a scenario, spawns vehicle.
    If damage_choice == 'fall', spawns with a 'fall' orientation/height,
    otherwise spawns normally (near ground).
    Returns: (bng, vehicle, scenario, IMU1, yaw0)
    """
    bng = BeamNGpy(cfg["host"], cfg["port"])
    try:
        bng.open(None, listen_ip='*', launch=False)
    except Exception:
        bng.open(None, listen_ip='*', launch=True)

    scenario = Scenario(cfg["map"], cfg["scenario_name"], description="CTRV/CTRA data collection")

    vehicle = Vehicle("ego_vehicle", model=model, license="DATA", color=color)

    # Spawn
    if damage_choice == "fall":
        pos, quat, yaw0 = fall_spawn(cfg)   # high Z + given quat (with noise)
    else:
        pos, quat, yaw0 = random_spawn(cfg) # near ground, random yaw

    scenario.add_vehicle(vehicle, pos=pos, rot_quat=quat)
    scenario.make(bng)
    bng.load_scenario(scenario)
    bng.start_scenario()

    #attach Damage Sensor
    dmg = DamageSensor()
    vehicle.attach_sensor("damage", dmg)
    
    for each in range(30):
        # _poll_vehicle_once(vehicle)
        bng.step(1)
    
    # Warm up state (for IMU pos fallback etc.)
    if damage_choice == "fall":
        qx, qy, qz, qw = quat
        x, y, z = pos
        yaw = 0.0 # True North
        qz = math.sin(yaw/2.0)
        qw = math.cos(yaw/2.0)
        desired_quat = (0, 0, qz, qw)
        vehicle.teleport((x, y, 0.7), desired_quat, reset=False)
        # IMU aligned with vehicle frame at (approx) COG

        for each in range(60):
            # _poll_vehicle_once()
            time.sleep(0.05)
            bng.step(1)

    vehicle.sensors.poll()
    qv = vehicle.state['rotation']
    # Get a rotated quaternion
    R_from_quat = R.from_quat(qv)
    r_rot = R.from_euler('z', 270, degrees=True)
    r_new_local = r_rot * R_from_quat
    Rv = r_new_local.as_matrix()

    forward = Rv[:, 0]
    left = Rv[:, 1]
    up = Rv[:, 2]

    forward_vec = tuple(forward.tolist())
    left_vec = tuple(left.tolist())
    up_vec = tuple(up.tolist())

    imu_pos = DU._get_imu_world_pos(vehicle)   # uses CoG or vehicle.state['pos']
    IMU1 = AIMU(
        "imu",
        bng,
        vehicle,
        is_send_immediately=True,
        pos=tuple(imu_pos),
        dir=forward_vec, 
        up=left_vec,
        physics_update_time=0.01,  # 100 Hz target
        gfx_update_time=0.01,
        is_using_gravity=True,
        is_allow_wheel_nodes=False,
        is_dir_world_space=False,
        is_snapping_desired=True
    )

    qx, qy, qz, qw = quat
    x, y, z = pos
    yaw = DU.rand_in(-math.pi, math.pi)
    qz = math.sin(yaw/2.0)
    qw = math.cos(yaw/2.0)
    desired_quat = (0, 0, qz, qw)
    vehicle.teleport((x, y, 0.7), desired_quat, reset=False)

    yaw0 = DU.quat_to_yaw(desired_quat)

    return bng, vehicle, scenario, IMU1, yaw0, pos

def choose_damage(cfg, model) -> str:
    """
    Choose a damage scenario string. Accepts your existing pool AND adds
    extra rare cases if desired.
    """
    # Compose weighted pool
    weighted = []
    for each in list(cfg["damage_pool"].keys()):
        weighted.append((each, cfg["damage_pool"][each])) #name and weight
  
    total = sum(w for _, w in weighted)
    rand_no = random.random() * total
    acc = 0.0
    for name, weight in weighted:
        acc += weight
        if rand_no <= acc:
            return name
    return list(cfg["damage_pool"].keys())[0]

def apply_damage(cfg, model: str, vehicle: Vehicle, damage_choice: str, pos: tuple):
    """
    Dispatch to the right damage action.
    """
    tyre_puncture = []
    if damage_choice == "fall":
        # already spawned high with noisy quat in bng_init(); no post-spawn damage
        print("[INFO] Damage: fall spawn (done at spawn time)")
        return {"fall": list(pos)[-1]}, tyre_puncture

    if damage_choice == "multi_tss":
        # Choose one adjacent pair (max 2, must be adjacent)
        pair = random.choice(cfg["adjacent_pairs"])
        for side in pair:
            wheel_id = cfg["wheel_ids"][side]
            puncture_tire(cfg, vehicle, wheel_id, side)
            tyre_puncture.append(f"{cfg['wheel_side_2_text'][side]} type punctured")
            break_shock_spring(cfg, model, vehicle, side)
        return {"multi_tss": pair}, tyre_puncture

    side = random.choice(list(cfg["wheel_ids"].keys()))
    if damage_choice == "axle_break":
        # pick one axle corner uniformly and break it
        break_axle(cfg, model, vehicle, side)
        return {"axle": side}, tyre_puncture

    wheel_id = cfg["wheel_ids"][side]
    if damage_choice == "tyre_puncture":
        puncture_tire(cfg, vehicle, wheel_id, side)
        tyre_puncture.append(f"{cfg['wheel_side_2_text'][side]} type punctured")
        return {"tyre_puncture": side}, tyre_puncture
    

    if damage_choice == "tyre&shock&spring":
        puncture_tire(cfg, vehicle, wheel_id, side)
        tyre_puncture.append(f"{cfg['wheel_side_2_text'][side]} type punctured")
        break_shock_spring(cfg, model, vehicle, side)
        return {"tyre&shock&spring": side}, tyre_puncture

    print(f"[WARN] Unknown damage type '{damage_choice}', no damage applied.")
    if damage_choice == 'none':
        return "none", "none"

    

# ----------------- Actions: fall spawn -----------------
def fall_spawn(cfg):
    """
    Build a high spawn using spawn_angle + noise and spawn_height_bound.
    YAML:
      spawn_angle: [(qx, qy, qz, qw), ...]  (we'll add noise to qx,qy)
      spawn_angle_sigma: float (noise std applied to first two components)
      spawn_height_bound: [min, max]  (Z)
    """
    # Base XY sampled like random_spawn; Z from bound
    xb = cfg["spawn_box"]["x"] 
    yb = cfg["spawn_box"]["y"]
    x = random.uniform(*xb) 
    y = random.uniform(*yb)

    z_bounds = cfg["spawn_box"]["z"]
    z = random.uniform(*z_bounds)
    print(f"[INFO] Spawning from {z}m height")
    angles = cfg["spawn_angle"]
    sigma = float(cfg["spawn_angle_sigma"])
    qx, qy, qz, qw = random.choice(angles)
    if sigma > 0.0:
        qx += random.gauss(0.0, sigma)
        qy += random.gauss(0.0, sigma)
    # Normalize quaternion (important)
    print(f"[INFO] Spawning at qx: {qx} and qy: {qy}")
    qx, qy, qz, qw = DU.normalize_quat((qx, qy, qz, qw))

    yaw = DU.quat_to_yaw((qx, qy, qz, qw))
    return (x, y, z), (qx, qy, qz, qw), yaw


# ----------------- Actions: tyre / shock_spring / axle -----------------
def puncture_tire(cfg, vehicle: Vehicle, wid: int, side: str):
    """
    Try a few known Lua hooks to 'puncture' tire for a given corner.
    Uses cfg['wheel_ids'][side] to identify wheel.
    """
    lua = f'''
    beamstate.deflateTire({wid})
    '''
    vehicle.queue_lua_command(lua)
    print(f"[INFO] Punctured tire on side: {side} (wheel id {wid})")

def break_shock_spring(cfg, model: str, vehicle: Vehicle, side: str):
    """
    Break shock + spring beams for a given wheel side, using YAML part_beam_id mapping.
    Expects keys like shock_spring_FR etc. If not present (e.g., pessima), skip.
    """
    model_map: Dict = cfg.get(model, {})
    key = f"shock_spring_{side}"
    ids = model_map.get(key, [])
    if not ids:
        print(f"[WARN] No shock/spring IDs for {model} {side}, skipping SS break")
        return
    for bid in ids:
        vehicle.queue_lua_command(f'obj:breakBeam({int(bid)})')
        time.sleep(0.01)
    print(f"[INFO] Broke shock+spring on {model} {side}: {ids}")

def break_axle(cfg, model: str, vehicle: Vehicle, side: str):
    """
    Break axle beams for a given wheel side (axle_FR etc.). Safe if model lacks IDs.
    """
    model_map: Dict = cfg.get(model, {})
    key = f"axle_{side}"
    ids = model_map.get(key, [])
    if not ids:
        print(f"[WARN] No axle IDs for {model} {side}, skipping axle break")
        return
    for bid in ids:
        vehicle.queue_lua_command(f'if obj and obj.breakBeam then obj:breakBeam({int(bid)}) end')
        time.sleep(0.01)
    print(f"[INFO] Broke axle on {model} {side}: {ids}")

# yaw could be needed, Dont know
def random_spawn(cfg):
    xb = cfg["spawn_box"]["x"]
    yb = cfg["spawn_box"]["y"]
    x = DU.rand_in(*xb)
    y =  DU.rand_in(*yb)
    z = 1.0

    yaw = 0.0

    qz = math.sin(yaw/2.0) 
    qw = math.cos(yaw/2.0)
    return (x, y, z), (0.0, 0.0, qz, qw), yaw