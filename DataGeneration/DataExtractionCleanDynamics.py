import numpy as np
import pickle
import os
import json
import argparse
import yaml
# from typing import List, Dict, Str, Any
import threading
import pdb
import copy
import text_generation as text_gen
from scipy.spatial.transform import Rotation as R
import data_utilities as DU

def body_increments_6dof(p1_batch, p2_batch, order='zyx', degrees=False):
    if p1_batch.shape != p2_batch.shape or p1_batch.shape[1] != 6:
        raise ValueError("Input arrays must have the same shape and be of shape (N, 6)")
    
    N = p1_batch.shape[0]
    out = np.zeros((N, 6), dtype=np.float32)
    
    p1 = p1_batch[:, :3]
    p2 = p2_batch[:, :3]

    r1 = R.from_euler(order, p1_batch[:, 3:], degrees=degrees).as_matrix()
    r2 = R.from_euler(order, p2_batch[:, 3:], degrees=degrees).as_matrix()

    #body frame relative translation : v = R1^T (p2 - p1)
    dp_world = p2 - p1
    v_body = np.einsum('nij,nj->ni', r1.transpose(0, 2, 1), dp_world)

    #Body frame relative rotation: R_rel = R1^T R2
    dr_body = np.einsum('nij,njk->nik', r1.transpose(0, 2, 1), r2)
    omega = R.from_matrix(dr_body).as_rotvec()

    out[:, :3] = v_body
    out[:, 3:] = omega

    return out

def save_data(raw_data_dictionary, output_folder, file_name, cfg):
    clean_data = put_it_in_sequence(raw_data_dictionary, file_name, cfg)
    file_path = os.path.join(output_folder, file_name)
    pickle.dump(clean_data, open(file_path, 'wb'))
    print(f"[INFO] Extracted from {file_name} and Data saved to {file_path}")

def get_dict(data):
    extracted_data = {
        
        'time': data['time'], #.tolist(),
        'pose': np.array(data['pose']), #.tolist(),
        'vel': np.array(data['twist']), #.tolist(),
        'accln': np.array(data['accln']), #.tolist(),
        'ctrv_a': np.array(data['ctrv_a']), #.tolist(),
        'ctrl': np.array(data['ctrl']), #.tolist(),
        'v_rotation': np.array(data['v_rotation']), #.tolist(),
        'd_robot_frame': np.array(data['d_robot_frame']), #.tolist(),
    }
    return extracted_data

def put_it_in_sequence(data_dict, file_name, cfg):
    #----------get all the configs -----------
    sim_hz      = float(cfg.get("sim_frequency"))
    target_hz   = float(cfg.get("data_frequency"))
    hist_secs      = float(cfg.get("horizon_length"))
    act_hist_secs  = float(cfg.get("action_sequence"))
    K_futures      = int(cfg.get("resultant_count"))
    delay_s        = float(cfg.get("resultant_delay_s", 0.2))

    # ----------create all the inputs as np arrays --------
    t_raw = np.asarray(data_dict['time'], dtype=float)
    pose = np.asarray(data_dict['pose'], dtype=float)
    ctrv_a = np.asarray(data_dict['ctrv_a'], dtype=float)
    ctrl = np.asarray(data_dict['ctrl'], dtype=float)

    #  ------------resample to target_hz -----------
    idx = DU.make_subsample_indices_from_time(t_raw, sim_hz, target_hz)
    t = t_raw[idx]
    poses_stride = pose[idx]
    ctrv_stride = ctrv_a[idx]
    ctrl_stride = ctrl[idx]
    T = len(t)

    # --- Length of the window for history and actions in number of steps -----
    hist_steps = int(round(hist_secs * target_hz))
    act_hist_steps = int(round(act_hist_secs * target_hz))
    delay_steps = max(0, int(round(delay_s * target_hz)))

    # -----------Data to Save ------------
    data_to_save = {
        'del_pose_seq': [],
        'del_action_seq': [],
        'action_seq': [],
        'ctrl_seq': [],
        'res_del_pose': [],       # single horizon (we’ll store the first valid future)
        'res_del_pose_seq': [],   # K futures
        'fut_action_seq': [],     # K future action seq
        'pose_full': poses_stride.tolist(),
    }

    pose_deltas = body_increments_6dof(poses_stride[:-1], poses_stride[1:], order='zyx', degrees=False)  #N-1 x 6
    act_hist = ctrv_stride[:-1, :2]  #N-1 x 2
    act_diff = np.diff(ctrv_stride, axis=0)[:, :2]  #N-2 x 2

    ctrl_diff = np.diff(ctrl_stride, axis=0)[:, [0, 2]]
    ctrl_hist = ctrl_stride[:-1, [0, 2]] 

    i_start = max(hist_steps -1, act_hist_steps -1)
    i_end = T -  K_futures - delay_steps - 1

    for i in range(i_start, i_end):
        dp_hist = pose_deltas[i - (hist_steps - 1) : i + 1]
        ad_hist = act_diff[i - (act_hist_steps - 1) : i + 1]
        a_hist = act_hist[i - (act_hist_steps -1) : i + 1]
        fut_pose_seq = pose_deltas[i + delay_steps + 1 : i + 1 + delay_steps + K_futures] 
        fut_act_seq = act_hist[i + delay_steps + 1 : i + 1 + delay_steps + K_futures] 
        ctrl_seq = ctrl_hist[i - (act_hist_steps - 1): i + 1]

        data_to_save['del_pose_seq'].append(dp_hist)
        data_to_save['del_action_seq'].append(ad_hist)
        data_to_save['action_seq'].append(a_hist)
        data_to_save['ctrl_seq'].append(ctrl_seq)
        data_to_save['res_del_pose_seq'].append(fut_pose_seq)
        data_to_save['res_del_pose'].append(fut_pose_seq[0])  # first valid future
        data_to_save['fut_action_seq'].append(fut_act_seq)

    return data_to_save



def threading_function(file_path, save_path, file_name, cfg):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    data_dict = get_dict(data)
    if len(data_dict['time']) > 0:
        save_data(copy.deepcopy(data_dict), save_path, file_name, cfg)
       

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg.get("data_folder")  # e.g.,
    save_path = cfg.get("output_folder")  # e.g.,


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file in os.listdir(data_dir):
        if file.endswith(".pkl"):
            file_path = os.path.join(data_dir, file)
            if cfg.get('enable_threading'):
                threading_array = []                
                threading_array.append(threading.Thread(target = threading_function, args=(file_path, save_path, file, cfg)))
                for thread in threading_array:
                    thread.start()
                for thread in threading_array:
                    thread.join()
            else:
                #no threads though,  sad affair
                threading_function(file_path, save_path, file, cfg)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    # arguments for both modes
    ap.add_argument("--config", type=str, default="config/data_extraction.yaml", help="YAML config file")
    args = ap.parse_args()

    main(args)
