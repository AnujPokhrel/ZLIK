#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
import pdb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="ExtractedPkl/etk", help="Folder with DataExtraction pickles")
    ap.add_argument("--out", type=str, default="damaged_stats_etk.pkl", help="Output pickle path")
    args = ap.parse_args()

    del_pose_last = []          # each: (40, 6)
    del_action_last = []        # each: (40, 2)
    velocity_last = []
    del_ctrl_last = []          # each: (40, 2)
    action_last = []            # each: (40, 2)
    control_last = []
    resultant_pose_rows = []      # each: (6,)
    resultant_pose_seq_last = []# each: (20, 6)
    pose_last = []                # each: (6,)

    # Load & aggregate
    files = sorted([f for f in os.listdir(args.data_dir) if f.endswith(".pkl")])
    for fname in files:
        with open(os.path.join(args.data_dir, fname), "rb") as f:
            d = pickle.load(f)

        # Taking only the last element from each window 
        del_pose_last.extend([np.asarray(a)[-1] for a in d["del_pose_seq"]])            # -> list of (6,)
        del_action_last.extend([np.asarray(a)[-1] for a in d["del_action_seq"]])        # -> list of (2,)
        velocity_last.extend([np.asarray(a)[-1] for a in d["vel_seq"]])
        del_ctrl_last.extend([np.asarray(a)[-1] for a in d["del_ctrl_seq"]])          # -> list of (2,)
        action_last.extend([np.asarray(a)[-1] for a in d["action_seq"]])                # -> list of (2,)
        control_last.extend([np.asarray(a)[-1] for a in d["ctrl_seq"]])                # -> list of (2,)
        resultant_pose_rows.extend([np.asarray(r) for r in d["res_del_pose"]])             # already (6,)
        resultant_pose_seq_last.extend([np.asarray(a)[-1] for a in d["res_del_pose_seq"]]) # -> list of (6,)
        pose_last.extend([np.asarray(a)[-1] for a in d["pose_seq"]])                # -> list of (6,)
    

    # Stack into big matrices
    del_pose = np.vstack([r.reshape(1, -1) for r in del_pose_last])                     # (N, 6)
    del_action = np.vstack([r.reshape(1, -1) for r in del_action_last])                 # (N, 2)
    del_ctrl = np.vstack([r.reshape(1, -1) for r in del_ctrl_last])                   # (N, 2)
    velocity = np.vstack([r.reshape(1, -1) for r in velocity_last])                          # (N, 6)
    action = np.vstack([r.reshape(1, -1) for r in action_last])                         # (N, 2)
    control = np.vstack([r.reshape(1, -1) for r in control_last])                         # (N, 2)
    resultant_pose = np.vstack([r.reshape(1, -1) for r in resultant_pose_rows])  # (N, 6)
    resultant_pose_seq = np.vstack([r.reshape(1, -1) for r in resultant_pose_seq_last]) # (N, 6)
    pose_last = np.vstack([r.reshape(1, -1) for r in pose_last])                     # (N, 6)

    # Helper to pack stats as vectors
    def pack(arr):
        return {
            "mean": arr.mean(axis=0),
            "std":  arr.std(axis=0),
            "min":  arr.min(axis=0),
            "max":  arr.max(axis=0),
            "median": np.median(arr, axis=0)
        }

    # Everything under ONE top-level key
    stats = {
        "del_pose":           pack(del_pose),            # vectors (6,)
        "del_action":         pack(del_action),          # vectors (2,)
        "del_ctrl":           pack(del_ctrl),            # vectors (2,)
        "velocity":           pack(velocity),            # vectors (6,)
        "action":             pack(action),              # vectors (2,)
        "ctrl":               pack(control),              # vectors (2,)
        "resultant_pose":     pack(resultant_pose),      # vectors (6,)
        "resultant_pose_seq": pack(resultant_pose_seq),  # vectors (6,)
        "pose_last":          pack(pose_last),           # vectors (6,)
        # Optional simple counts for sanity
        "counts": {
            "del_pose_rows":           del_pose.shape[0],
            "del_action_rows":         del_action.shape[0],
            "del_ctrl_rows":           del_ctrl.shape[0],
            "velocity_rows":           velocity.shape[0],
            "action_rows":             action.shape[0],
            "control_rows":            control.shape[0],
            "resultant_pose_rows":     resultant_pose.shape[0],
            "resultant_pose_seq_rows": resultant_pose_seq.shape[0],
            "pose_last_rows":          pose_last.shape[0],
            "files":                   len(files),
        },
        # Column semantics for the 6-vecs
        "labels_6d": ["dx", "dy", "dz", "d_rx", "d_ry", "d_rz"],
        "labels_2d": ["a0", "a1"],
    }

    with open(args.out, "wb") as f:
        pickle.dump({"stats": stats}, f)

    # Quick echo: pose_mean is a (6,) ndarray as requested
    print("Wrote", args.out)
    print("pose_mean:", stats["del_pose"]["mean"].shape, stats["del_pose"]["mean"])

if __name__ == "__main__":
    main()
