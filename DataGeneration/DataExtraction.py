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

def save_data(raw_data_dictionary, output_folder, file_name, cfg):
    clean_data = put_it_in_sequence(raw_data_dictionary, file_name, cfg)
    print(f"[INFO] Data extracted from {file_name}, saving to {output_folder}... No of entries: {len(clean_data['del_pose_seq'])}")
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
        #all the damage text, there is 
        'damage_text': data['damage_text'], #.tolist()
        #get all embeddings
        'embedding_none': np.array(data['embedding_none']),
        'embedding_mid': np.array(data['embedding_mid']),
        'embedding_all': np.array(data['embedding_all'])
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
    del_pose = np.asarray(data_dict['d_robot_frame'], dtype=float)
    ctrv_a = np.asarray(data_dict['ctrv_a'], dtype=float)
    ctrl = np.asarray(data_dict['ctrl'], dtype=float)
    emb_all = np.asarray(data_dict['embedding_all'])
    emb_none = np.asarray(data_dict['embedding_none'])
    emb_mid = np.asarray(data_dict['embedding_mid'])
    dmg_txt = data_dict['damage_text']
    vel = np.asarray(data_dict['vel'], dtype=float)

    # --- Length of the window for history and actions in number of steps -----
    hist_steps = int(round(hist_secs * target_hz))
    act_hist_steps = int(round(act_hist_secs * target_hz))
    delay_steps = max(0, int(round(delay_s * target_hz)))


    #  ------------resample to target_hz -----------
    idx = DU.make_subsample_indices_from_time(t_raw, sim_hz, target_hz)
    t = t_raw[idx]
    poses_stride = pose[idx]
    ctrv_stride = ctrv_a[idx]
    vel = vel[idx]
    
    ctrl_stride = ctrl[idx]
    emb_all_stride = emb_all[idx]
    emb_none_stride = emb_none[idx]
    emb_mid_stride = emb_mid[idx]
    dmg_txtS = [dmg_txt[i] for i in idx]

    T = len(t)

    # -----------Data to Save ------------
    data_to_save = {
        'del_pose_seq': [],
        'del_action_seq': [],
        'del_ctrl_seq': [],
        'vel_seq': [],

        'pose_seq': [],
        'action_seq': [],
        'ctrl_seq': [],
        'pose_full': poses_stride.tolist(),

        'res_del_pose': [],       # single horizon (we’ll store the first valid future)
        'res_del_pose_seq': [],   # K futures
        'fut_ctrl_seq': [],
        'fut_del_ctrl_seq': [],
        'fut_action_seq': [],     # K future action seq
        'res_pose_seq': [],

        'damage_text_all': [],
        'damage_text_mid': [],
        'damage_text_parsed': [],
        'text_embedding_all': [],
        'text_embedding_mid': [],
        'text_embedding_none': []
    }

    act_hist = ctrv_stride[:-1, :2]  #N-1 x 2
    act_diff = np.diff(ctrv_stride, axis=0)[:, :2]  #N-2 x 2
    ctrl_diff = np.diff(ctrl_stride, axis=0)[:, [0, 2]]
    ctrl_hist = ctrl_stride[:-1, [0, 2]]

    i_start = max(hist_steps -1, act_hist_steps -1)
    i_end = T -  K_futures - delay_steps

    damage_same_flag = False
    if len(dmg_txtS) > 1 and (dmg_txtS[0] == dmg_txtS[-1]):
        damage_same_flag = True
        mid_damage_text = text_gen.mid_text_from_vehicle_sensors(dmg_txtS[0], K=48, keep_cosmetic=False)
        all_damage_text = text_gen.all_text_from_vehicle_sensors(dmg_txtS[0], K=48, keep_cosmetic=False)

    for i in range(i_start, i_end):
        dp_hist = del_pose[i - (hist_steps - 1) : i + 1]
        pose_hist = poses_stride[i + 1 - (hist_steps-1): i + 2]

        vel_hist = vel[i - (hist_steps -1) : i + 1] 

        a_hist = act_hist[i - (act_hist_steps -1) : i + 1]
        ad_hist = act_diff[i - (act_hist_steps - 1) : i + 1]

        ctrl_seq = ctrl_hist[i - (act_hist_steps - 1): i + 1]
        ctrl_diff_seq = ctrl_diff[i - (act_hist_steps - 1) : i + 1]

        fut_del_pose_seq = del_pose[i + delay_steps + 1 : i + 1 + delay_steps + K_futures] 
        fut_pose_seq = poses_stride[i + delay_steps + 2 : i + 2 + delay_steps + K_futures]

        fut_act_seq = act_hist[i + delay_steps + 1 : i + 1 + delay_steps + K_futures] 
        fut_ctrl_seq = ctrl_hist[i + delay_steps + 1 : i + 1 + delay_steps + K_futures] 
        fut_del_ctrl_seq = ctrl_diff[i + delay_steps + 1 : i + 1 + delay_steps + K_futures]

        # text embeddings aligned at pose index i
        data_to_save['text_embedding_all'].append(emb_all_stride[i])        #HERE HERE: all means fully parsed
        data_to_save['text_embedding_mid'].append(emb_mid_stride[i])
        data_to_save['text_embedding_none'].append(emb_none_stride[i])

        # damage text variants
        if damage_same_flag:
            data_to_save['damage_text_all'].append(dmg_txtS[0])
            data_to_save['damage_text_mid'].append(mid_damage_text)
            data_to_save['damage_text_parsed'].append(all_damage_text)
        else:
            data_to_save['damage_text_all'].append(dmg_txtS[i])
            data_to_save['damage_text_mid'].append(
                text_gen.mid_text_from_vehicle_sensors(dmg_txtS[i], K=48, keep_cosmetic=False)
            )
            data_to_save['damage_text_parsed'].append(
                text_gen.all_text_from_vehicle_sensors(dmg_txtS[i], K=48, keep_cosmetic=False)
            )

        data_to_save['del_pose_seq'].append(dp_hist)
        data_to_save['pose_seq'].append(pose_hist)
        data_to_save['vel_seq'].append(vel_hist)

        data_to_save['del_action_seq'].append(ad_hist)
        data_to_save['action_seq'].append(a_hist)

        data_to_save['ctrl_seq'].append(ctrl_seq)
        data_to_save['del_ctrl_seq'].append(ctrl_diff_seq)

        data_to_save['res_del_pose_seq'].append(fut_del_pose_seq)
        data_to_save['res_del_pose'].append(fut_del_pose_seq[0])  # first valid future
        data_to_save['res_pose_seq'].append(fut_pose_seq)

        data_to_save['fut_action_seq'].append(fut_act_seq)
        data_to_save['fut_ctrl_seq'].append(fut_ctrl_seq)
        data_to_save['fut_del_ctrl_seq'].append(fut_del_ctrl_seq)

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
