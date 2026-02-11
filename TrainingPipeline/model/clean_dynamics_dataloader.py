from pathlib import Path
import pickle
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
from tqdm import tqdm

def normalize_standard(data, mean, std):
    return (data - mean)/ (std + 1e-8)

def feature_dim(pose_dim, act_dim):
    return int(pose_dim + act_dim)

def expected_steps(hz, window_secs):
    return int(hz * window_secs)


class CleanDyamicsFnDataset(Dataset):
    '''get items returns
    del_pose: floatTensor 1 time step so (1, 6)
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.pose_key = cfg.get("pose_key_10s", "del_pose_sequence")
        self.action_key = cfg.get("actions_key_10s", "action")
        self.gt_pose_key = cfg.get("gt_pose_key", "resultant_pose")
        self.fut_actions_key = cfg.get("future_actions_key", "fut_action_seq")
        self.prediction_horizon = cfg.get("prediction_horizon", 3) 
        self.attack = cfg.get("attack", True)
        self.add_noise = cfg.get("noise", False)

        data_root = Path(self.cfg.root)
        all_files = sorted(list(data_root.glob("*.pkl")))
        print(f"Found {len(all_files)} data files in {data_root}")

        self.records = []
        for pkl_file in tqdm(all_files, desc="Loading data files"):
            with open(pkl_file, "rb") as f:
                self.records.append(pickle.load(f))
        
        self.index = []
        print(f"length of the records: {len(self.records)}")
        for file_idx, rec in enumerate(self.records):
            # Check if the record is valid and has the required keys
            if not rec or self.pose_key not in rec:
                print(f"Skipping empty or invalid record at file index {file_idx}")
                continue
        
            num_sequences = len(rec[self.pose_key])
            for seq_idx in range(num_sequences):
                gt_seq = np.array(rec[self.gt_pose_key][seq_idx], dtype=np.float32)
                if gt_seq.ndim == 1:
                    gt_seq = gt_seq[None, :]
                K_i = gt_seq.shape[0]
                if K_i != self.prediction_horizon:
                    continue

                self.index.append({"file_idx": file_idx, "seq_idx": seq_idx})
        
        with open(self.cfg.data_stats, 'rb') as f:
            stats_data = pickle.load(f)
            self.stats = stats_data.get("stats", {}) # Handle nested stats key

        # Extract stats for normalization, converting them to torch tensors
        self.del_pose_mean = torch.tensor(self.stats['del_pose']['mean'], dtype=torch.float32)
        self.del_pose_std = torch.tensor(self.stats['del_pose']['std'], dtype=torch.float32)

        self.action_mean = torch.tensor(self.stats['ctrl']['mean'], dtype=torch.float32)
        self.action_std = torch.tensor(self.stats['ctrl']['std'], dtype=torch.float32)

        self.hz = int(cfg.get("data_frequency", 20))
        self.window_secs = cfg.get("horizon_length", 10.0)
        self.T = expected_steps(self.hz, self.window_secs)
    
        self.pose_dim = int(cfg.get("pose_dim", 6))
        self.act_dim = int(cfg.get("action_dim", 2))
        self.feature_dim = feature_dim(self.pose_dim, self.act_dim)

        self.batch_size = int(cfg.get("batch_size", 64))

        print(f"No of data points: {len(self.index)}")

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        """
        Retrieves, normalizes, and returns one sample of behavior data and
        the corresponding raw damage text.
        """

        map_info = self.index[idx]
        file_idx, seq_idx = map_info["file_idx"], map_info["seq_idx"]
        record = self.records[file_idx]

        del_pose = torch.tensor(record[self.pose_key][seq_idx], dtype=torch.float32)
        action = torch.tensor(record[self.action_key][seq_idx], dtype=torch.float32)
        gt_pose = torch.tensor(record[self.gt_pose_key][seq_idx], dtype=torch.float32)
        fut_action = torch.tensor(record[self.fut_actions_key][seq_idx], dtype=torch.float32)

        del_pose_norm = normalize_standard(del_pose, self.del_pose_mean, self.del_pose_std)
        action_norm = normalize_standard(action, self.action_mean, self.action_std)
        fut_action_norm = normalize_standard(fut_action, self.action_mean, self.action_std)
        gt_pose_norm = normalize_standard(gt_pose, self.del_pose_mean, self.del_pose_std)

        del_pose_noise_norm, action_noise_norm = self.apply_robustness(del_pose_norm, action_norm)

        del_pose_T = self.pad_to_T(del_pose_noise_norm, feat = self.pose_dim)
        action_T = self.pad_to_T(action_noise_norm, feat = self.act_dim)

        history = torch.cat([del_pose_T, action_T], dim=1)

        return {
            "history": history,
            "gt_pose": gt_pose_norm,
            "future_action": fut_action_norm
        }
    
    def pad_to_T(self, x, feat):
        if x.shape[0] >= self.T:
            x = x[-self.T:]
        else:
            pad = torch.zeros(self.T - x.shape[0], feat)
            x = torch.cat([pad, x], dim=0)
        return x
    
    def apply_robustness(self, del_pose, action):
        """
        Applies Noise and Attacks as per AnyCar paper.
        """
        #add gaussian Noise
        if self.add_noise:
            del_pose += torch.randn_like(del_pose) * 0.01 
            action += torch.randn_like(action) * 0.01

        #attack 
        if self.attack:
            T, D = del_pose.shape
            attack_mask = torch.rand(T) < 0.05
            
            if attack_mask.any():
                #select random dimensions to attack 
                dim_indices = torch.randint(0, D, (attack_mask.sum(),))
                
                noise_vals = (torch.rand(attack_mask.sum()) * 60 - 30) 
                
                #apply attack
                step_indices = torch.where(attack_mask)[0]
                del_pose[step_indices, dim_indices] += noise_vals

        return del_pose, action
    
def collate_clean_behavior(batch):
    """Collate into
    del_pose : floatTensor [B, T, 6]
    action : floatTensor [B, T, 2]
    gt_pose : floatTensor [B, K, 6]
    """

    history = torch.stack([item["history"] for item in batch], dim=0)
    future_actions = torch.stack([item["future_action"] for item in batch], dim=0)
    gt_pose = torch.stack([item["gt_pose"] for item in batch], dim=0)
    
    B, H, _ = history.shape
    _, P, _ = future_actions.shape

    history_padding_mask = torch.ones(B, H, dtype=torch.bool, device=history.device)
    action_padding_mask = torch.ones(B, P, dtype=torch.bool, device=history.device)


    return {"history": history, 
            "future_action": future_actions,
            'gt_pose': gt_pose,
            "history_padding_mask": history_padding_mask,
            "action_padding_mask": action_padding_mask
            }

