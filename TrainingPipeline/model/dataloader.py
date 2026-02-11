from pathlib import Path
import copy
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


class DamageBehaviorPreTrain(Dataset):
    '''get items returns
    text_raw: str,
    del_pose: floatTensor
    del_action: floatTensor
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.pose_key = cfg.get("pose_key_10s", "del_pose_sequence")
        self.del_action_key = cfg.get("del_actions_key_10s", "del_action_sequence")
        self.action_key = cfg.get("actions_key_10s", "action")
        self.text_key = cfg.get("damage_text_key", "damage_text_parsed")
        self.embedding_key = cfg.get("damage_embedding_key", "embedding_none")

        data_root = Path(self.cfg.root)
        all_files = sorted(list(data_root.glob("*.pkl")))
        print(f"Found {len(all_files)} data files in {data_root}")

        self.records = []
        for pkl_file in tqdm(all_files, desc="Loading data files"):
            with open(pkl_file, "rb") as f:
                self.records.append(pickle.load(f))
        
        self.index = []
        for file_idx, rec in enumerate(self.records):
            # Check if the record is valid and has the required keys
            if not rec or self.pose_key not in rec:
                print(f"Skipping empty or invalid record at file index {file_idx}")
                continue
        
            num_sequences = len(rec[self.pose_key])
            for seq_idx in range(num_sequences):
                self.index.append({"file_idx": file_idx, "seq_idx": seq_idx})
        
        with open(self.cfg.data_stats, 'rb') as f:
            stats_data = pickle.load(f)
            self.stats = stats_data.get("stats", {}) # Handle nested stats key


        # Extract stats for normalization, converting them to torch tensors
        self.del_pose_mean = torch.tensor(self.stats['del_pose']['mean'], dtype=torch.float32)
        self.del_pose_std = torch.tensor(self.stats['del_pose']['std'], dtype=torch.float32)
        self.del_action_mean = torch.tensor(self.stats['del_ctrl']['mean'], dtype=torch.float32)
        self.del_action_std = torch.tensor(self.stats['del_ctrl']['std'], dtype=torch.float32)
        self.action_mean = torch.tensor(self.stats['ctrl']['mean'], dtype=torch.float32)
        self.action_std = torch.tensor(self.stats['ctrl']['std'], dtype=torch.float32)

        self.hz = int(cfg.get("data_frequency", 20))
        self.window_secs = cfg.get("horizon_length", 10.0)
        self.T = expected_steps(self.hz, self.window_secs)
    
        self.pose_dim = int(cfg.get("pose_dim", 6))
        self.act_dim = int(cfg.get("action_dim", 2))
        self.feature_dim = feature_dim(self.pose_dim, self.act_dim)

        self.batch_size = cfg.get


    def __len__(self):
        return len(self.index)
    
    def load_window_arrays(self, rec, t):
        del_pose = np.array(rec[self.pose_key][t], dtype=np.float32, copy=False)  # (L, 6)
        del_action = np.array(rec[self.actions_key][t], dtype=np.float32, copy=False)

        assert del_pose.ndim == 2 and del_pose.shape[1] == self.pose_dim, \
            f"Expected del_pose shape (L, {self.pose_dim}), got {del_pose.shape}"
        assert del_action.ndim == 2 and del_action.shape[1] == self.act_dim, \
            f"Expected del_action shape (L, {self.act_dim}), got {del_action.shape}"
        assert del_pose.shape[0] == del_action.shape[0], \
            f"Expected del_pose and del_action to have same length, got {del_pose.shape[0]} and {del_action.shape[0]}"
        
        return {"del_pose": del_pose, "del_action": del_action}
    
    def load_text(self, rec, t):
        txt = rec[self.text_key][t]
        if isinstance(txt, (dict, list)):
            txt = json.dumps(txt)
        
        assert isinstance(txt, str), f"Expected txt to be a string, got {type(txt)}"
        return txt
    
    def __getitem__(self, idx):
        """
        Retrieves, normalizes, and returns one sample of behavior data and
        the corresponding raw damage text.
        """

        map_info = self.index[idx]
        file_idx, seq_idx = map_info["file_idx"], map_info["seq_idx"]
        record = self.records[file_idx]

        del_pose = torch.tensor(record[self.pose_key][seq_idx], dtype=torch.float32)
        del_action = torch.tensor(record[self.del_action_key][seq_idx], dtype=torch.float32)
        action = torch.tensor(record[self.action_key][seq_idx], dtype=torch.float32)

        del_pose_norm = normalize_standard(del_pose, self.del_pose_mean, self.del_pose_std)
        del_action_norm = normalize_standard(del_action, self.del_action_mean, self.del_action_std)
        action_norm = normalize_standard(action, self.action_mean, self.action_std)

        # --- Damage Text ---
        text_raw = record[self.text_key][seq_idx]
        if isinstance(text_raw, (dict, list)):
            text_raw = json.dumps(text_raw) # Ensure text is a string
        
        text_embedding = torch.tensor(record[self.embedding_key][seq_idx], dtype=torch.float32)

        return {
            "del_pose": del_pose_norm,
            "del_action": del_action_norm,
            "action": action_norm,
            "text_raw": text_raw,
            "text_embedding": text_embedding
        }
    
    def pad_to_T(self, x, feat):
        if x.shape[0] >= self.T:
            x = x[-self.T:]
        else:
            pad = torch.zeros(self.T - x.shape[0], feat)
            x = torch.cat([pad, x], dim=0)
        return x
    
def collate_damage_behavior(batch):
    """Collate into
    del_pose : floatTensor [B, T, 6]
    del_action : floatTensor [B, T, 2]
    text_raw : list of str [B]
    """

    del_poses = torch.stack([item["del_pose"] for item in batch], dim=0)
    del_actions = torch.stack([item["del_action"] for item in batch], dim=0)
    actions = torch.stack([item["action"] for item in batch], dim=0)
    text_embeddings = torch.stack([item["text_embedding"] for item in batch], dim=0)
    texts_raw = [item["text_raw"] for item in batch]

    return {"del_pose": del_poses, 
            "del_action": del_actions,
            'action': actions,
            "text_raw": texts_raw,
            "text_embedding": text_embeddings}

def make_dataloader(records, cfg) -> DataLoader:
    dataset = DamageBehaviorPreTrain(records, cfg)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=bool(cfg.get("shuffle", True)),
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_damage_behavior,
    )

    return loader