# random_walk.py
import math, random, pickle, os, datetime
from typing import Dict, List
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
import pdb
import time
import argparse

from random_walk_algorithms import CTRV_CTRA_Driver, SpeedPID
import data_utilities as DU
import beamng_init as binit

def main(cfg):
    # timebase
    dt = 1.0/float(cfg["sim_frequency"])
    steps = int(float(cfg["run_time_s"]) / dt)
    for each in range(cfg["num_trials"]):
        # Pick a random vehicle
        vinfo = random.choice(cfg["vehicle_pool"])
        model = vinfo["model"]; color = vinfo.get("color", "White")
        vp = DU.get_vehicle_params(cfg, model)
        print(f"[INFO] Selected vehicle: {model} (L={vp['L']:.2f} m, max_steer={vp['max_steer_rad']:.2f} rad)")

        # Pick a damage scenario
        damage_choice = "None"
        print(f"[INFO] Damage scenario: No Damage")

        # BeamNG init + spawn (fall handled inside if chosen)
        bng, vehicle, scenario, IMU1, yaw0, pos = binit.bng_init(cfg, model, color, damage_choice)

        # Apply damage (puncture / shock+spring / axle)
        # damage_info, tyre_damage_info = binit.apply_damage(cfg, model, vehicle, damage_choice, pos)
        tyre_damage_info = {}
        damage_info = "No Damage"
        # Driver
        driver = CTRV_CTRA_Driver(cfg, model)
        driver.yaw = yaw0

        pid = SpeedPID(kp=0.4, ki=0.04, kd=0.1, dt=0.05, throttle_bounds=cfg["throttle_bounds"], brake_bounds=cfg["brake_bounds"])

        # Logs
        saving_output = {
            'scenario_meta': [],
            'time': [],
            'vehicle_model': [],
            'ctrl': [],
            'ctrv_a': [],
            'pose': [],   # x,y,z, roll,pitch,yaw (IMU-derived)
            'twist': [],  # vx,vy,vz (world), wx,wy,wz (body)
            'accln': [],   # ax,ay,az (body), awx,awy,awz (body)
            'v_rotation': [],
            'd_robot_frame': [],
            'damage_text': []
        }

        last_recorded_pose = np.zeros(6)
        print("[INFO] Running...")
        time.sleep(0.5)
        for k in range(steps):
            vehicle.sensors.poll()
            vstate = vehicle.state
            curr_dmg_text = vehicle.sensors.data.get('damage', {})

            # CTRV/CTRA control
            out = driver.step(dt)
            v_target, omega_target = out["v"], out["omega"]
            v_current = np.linalg.norm(vehicle.state['vel'][:2])

            if k < 20:
                throttle, brake = 0.2, 0.0
            else:
                throttle, brake = pid.step(v_target, v_current)
            if abs(v_target) < 0.2:
                steering = 0.0
            else:
                delta = math.atan(driver.L * omega_target / v_target)
                steering = np.clip(delta / driver.max_steer, -1.0, 1.0)

            # Apply & advance
            if k%10 == 0:
                print(f"{throttle = }, {brake =}, {steering = }")
            vehicle.control(throttle=throttle, steering=steering, brake=brake)
            bng.step(1)

            # State + IMU
            I = IMU1.poll()
            if not I:
                continue
            if 0.0 in I.keys():
                imu = I[list(I.keys())[-1]]
            else:
                imu = I  # already a single reading

            # Pose from IMU basis
            x, y, z = imu['pos']
            dir_x = np.array(imu['dirX']); dir_y = np.array(imu['dirY']); dir_z = np.array(imu['dirZ'])
            Rmat = np.column_stack((dir_x, dir_y, dir_z))
            r = R.from_matrix(Rmat)
            yaw, pitch, roll= r.as_euler('zyx', degrees=False)

            # Velocities
            vx, vy, vz = vstate['vel']
            wx, wy, wz = imu.get('angVelSmooth', imu.get('angVel', [0.0,0.0,0.0]))
            ax, ay, az = imu.get('accSmooth', imu.get('accRaw', [0.0,0.0,0.0]))
            awx, awy, awz = imu.get('angAccel', [0.0,0.0,0.0])

            v_quart = vstate['rotation']
            rotation = R.from_quat(v_quart)

            v_roll,v_pitch,v_yaw = rotation.as_euler('zyx', degrees=False)
            last_recorded_pose = np.zeros(6)
            # Log
            curr_dmg_text['tyre_damage'] = tyre_damage_info

            if (k % int(cfg["log_every_n_steps"])) == 0 and k >= 20:
                saving_output['time'].append(k * dt)
                saving_output['ctrl'].append([float(throttle), float(brake), float(steering)])

                saving_output['ctrv_a'].append([float(out["v"]), float(out["omega"]), float(out["a"]), float(out["yaw"])])
                saving_output['pose'].append([float(x), float(y), float(z), float(yaw), float(pitch), float(roll)])
                saving_output['twist'].append([float(vx), float(vy), float(vz), float(wx), float(wy), float(wz)])
                saving_output['accln'].append([float(ax), float(ay), float(az), float(awx), float(awy), float(awz)])
                saving_output['v_rotation'].append([float(v_roll), float(v_pitch), float(v_yaw)])
                saving_output['d_robot_frame'].append(DU.to_robot_6_dof(np.array(saving_output['pose'][-1]), last_recorded_pose, degrees=False))
                saving_output['damage_text'].append(curr_dmg_text)
                last_recorded_pose = np.array(saving_output['pose'][-1])
        
        saving_output['scenario_meta'] = {
            "vehicle_model": model,
            "damage": damage_choice, 
            "damage_type": damage_info
        }

        # Save
        DU.ensure_dir_for(cfg["out_pickle_folder"])
        ts = datetime.datetime.now().strftime("%m%d_%H%M")
        out_path = os.path.join(cfg["out_pickle_folder"], f"{damage_choice}_{model}_{ts}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(saving_output, f)

        print(f"[INFO] Saved {len(saving_output['time'])} samples to {out_path}")
    bng.close()

if __name__ == "__main__":
    with open("config/data_gen.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
