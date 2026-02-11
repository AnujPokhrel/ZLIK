"""
Quick demo: turn BeamNG `vehicle.sensors` damage dicts into text, encode with
DistilBERT, and compute cosine similarity between two damage states.

Usage:
  pip install torch transformers
  python distilbert_damage_embedding_pipeline.py --demo  # runs a tiny demo

To use your own dicts, import `to_text_from_vehicle_sensors` and
`encode_text_distilbert`, or run as a script with --json paths.
"""
from __future__ import annotations
from sentence_transformers import SentenceTransformer
import argparse
import json
import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D # noqa: F401
import pdb
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TrainingPipeline.model.dmv_behavior_model import DamagedVehicleBehaviorModel
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import yaml

# ------------------------------
# 1) Canonicalization helpers
# ------------------------------

# Parts that are mostly cosmetic / weakly coupled to dynamics.
COSMETIC_KEYWORDS = {
    "glass", "headlight", "taillight", "lightbar", "foglight", "mirror",
    "door", "seat", "hood", "trunk", "tailgate", "bumper", "roof", "panel",
    "quarterglass", "display", "interior", "floorjack", "sparewheel", "sparetire", "42x12.50r17_crawler_spare_tire"
}

# Map raw keys/names to system labels (very light rules; extend as needed).
SYSTEM_RULES: List[Tuple[str, str]] = [
    (r"\b(suspension)\b", "suspension"),
    (r"\b(steering|tie_rod|rack)\b", "steering"),
    (r"\b(driveshaft|halfshaft|wheelaxle|axle)\b", "drivetrain"),
    (r"\b(differential|transfer_case)\b", "drivetrain"),
    (r"\b(tire|tyre|pressure|lowpressure)\b", "tire"),
    (r"\b(engine|intake|turbo|intercooler|oilpan|radiator)\b", "powertrain"),
    (r"\b(wheel|hub|spindle|knuckle|brake)\b", "unsprung"),
]

SIDE_RULES: List[Tuple[str, str]] = [
    (r"\bFL\b|front\s*left", "FL"),
    (r"\bFR\b|front\s*right", "FR"),
    (r"\bRL\b|rear\s*left", "RL"),
    (r"\bRR\b|rear\s*right", "RR"),
    (r"\bF\b|\bfront\b", "F"),
    (r"\bR\b|\brear\b", "R"),
]

SEVERITY_BINS = [
    (0.0, "none"), (0.1, "mild"), (0.3, "moderate"), (0.6, "high"), (1.01, "severe")
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _is_cosmetic(text: str) -> bool:
    t = _norm(text)
    return any(k in t for k in COSMETIC_KEYWORDS)


def _pick_system(text: str) -> str:
    t = _norm(text)
    for pat, lab in SYSTEM_RULES:
        if re.search(pat, t):
            return lab
    return "misc"


def _pick_side(text: str) -> str:
    t = _norm(text)
    for pat, lab in SIDE_RULES:
        if re.search(pat, t):
            return lab
    return "NA"


def _bin_severity(x: float) -> str:
    for thr, name in SEVERITY_BINS:
        if x < thr:
            return name
    return "severe"


@dataclass
class CanonicalRecord:
    system: str
    side: str
    part: str
    severity_raw: float
    severity_bin: str
    source: str  # 'part_damage' | 'deform_group_damage' | 'flag'
    original: str

    def to_phrase(self) -> str:
        side_map = {
            "RL": "rear left",
            "RR": "rear right",
            "F": "front",
            "R": "rear",
            "NA": "",
        }
        side_words = side_map.get(self.side, "").strip()


        # Part normalization (very light heuristics; extend as needed)
        t = _norm(self.part).replace("_", " ")
        part = t
        if any(k in t for k in ["halfshaft", "wheelaxle", "axle shaft"]):
            part = "halfshaft"
        elif "driveshaft" in t:
            part = "driveshaft"
        elif "differential" in t:
            part = "differential"
        elif "strut" in t:
            part = "strut"
        elif "shock" in t:
            part = "shock"
        elif "suspension" in t:
            part = "suspension"
        elif any(k in t for k in ["hub", "spindle", "knuckle"]):
            part = "hub"
        elif any(k in t for k in ["steering", "tie rod", "rack"]):
            part = "steering"
        elif "radiator" in t:
            part = "radiator"
        elif "oilpan" in t or "oil pan" in t:
            part = "oil pan"
        elif "intercooler" in t:
            part = "intercooler"
        elif "turbo" in t:
            part = "turbocharger"
        elif "intake" in t:
            part = "intake"
        elif "engine mount" in t or "enginemount" in t or "engine mounts" in t:
            part = "engine mount"
        elif "tire" in t or "tyre" in t:
            part = "tire"


        sev = self.severity_bin
        # assemble phrase
        bits = [side_words, part, sev]
        phrase = " ".join([b for b in bits if b])
        return phrase


def extract_records(payload: Dict[str, Any], *, keep_cosmetic: bool = False) -> List[CanonicalRecord]:
    recs: List[CanonicalRecord] = []

    # 1) part_damage: authoritative, richer names
    pd: Dict[str, Any] = payload.get("part_damage", {}) or {}
    for path, info in pd.items():
        name = str(info.get("name", Path(path).name))
        dmg = float(info.get("damage", 0.0))
        if not keep_cosmetic and _is_cosmetic(name):
            continue
        recs.append(
            CanonicalRecord(
                system=_pick_system(name),
                side=_pick_side(name or path),
                part=name,
                severity_raw=max(0.0, min(1.0, dmg)),
                severity_bin=_bin_severity(dmg),
                source="part_damage",
                original=path,
            )
        )

    # 2) deform_group_damage: keep only dynamics-relevant keys
    dgd: Dict[str, Any] = payload.get("deform_group_damage", {}) or {}
    for key, d in dgd.items():
        name = key
        if not keep_cosmetic and _is_cosmetic(name):
            continue
        if _pick_system(name) in {"drivetrain", "suspension", "steering", "powertrain", "tire", "unsprung"}:
            dmg = float(d.get("damage", 0.0))
            recs.append(
                CanonicalRecord(
                    system=_pick_system(name),
                    side=_pick_side(name),
                    part=name,
                    severity_raw=max(0.0, min(1.0, dmg)),
                    severity_bin=_bin_severity(dmg),
                    source="deform_group_damage",
                    original=key,
                )
            )

    # 3) top-level flags (e.g., lowpressure)
    if payload.get("lowpressure") is True:
        recs.append(
            CanonicalRecord(
                system="tire",
                side="NA",
                part="lowpressure",
                severity_raw=0.8,
                severity_bin=_bin_severity(0.8),
                source="flag",
                original="lowpressure",
            )
        )

    return recs


def all_text_from_vehicle_sensors(payload: Dict[str, Any], *, K: int = 48, keep_cosmetic: bool = False) -> str:
    """Build a compact text summary from a BeamNG damage payload.
    Select top-K records by severity (ties arbitrary).
    """
    recs = extract_records(payload, keep_cosmetic=keep_cosmetic)
    if not recs:
        return "no_dynamic_damage"
    # Impact-aware ranking could be added; start with severity only
    recs.sort(key=lambda r: r.severity_raw, reverse=True)
    phrases = [r.to_phrase() for r in recs[:K]]
    # Include a header to mark vehicle/model if present
    model = str(payload.get("model", payload.get("vehicle_model", "unknown_model")))
    header = f"vehicle:{_norm(model)}"
    return " [SEP] ".join(phrases)

def mid_text_from_vehicle_sensors(payload: Dict[str, Any], *, K: int = 48, keep_cosmetic: bool = False) -> str:
    damaged_parts: List[Tuple[float, str]] = []
    part_damage_dict: Dict[str, Any] = payload.get("part_damage", {}) or {}
    #check cosmetic and low damage
    for _, info in part_damage_dict.items():
        name = str(info.get("name", "unknown_part"))
        damage = float(info.get("damage", 0.0))

        if damage < 1e-3:
            continue

        if not keep_cosmetic and _is_cosmetic(name):
            continue
        
        damaged_parts.append((damage, name))
    if not damaged_parts:
        return "no_damage"
    
    #Sort damage
    damaged_parts.sort(key=lambda item: item[0], reverse=True)
    
    #Create the text format of the string
    phrases = []
    for dmg, name in damaged_parts[:K]:
        normalized_name = _norm(name).replace(' ', '_')
        phrase = f"{normalized_name} {dmg:.3f}"
        phrases.append(phrase) 
    
    return " [SEP] ".join(phrases)

#Embeeding gemma Encoding
def load_embeedinggemma(device: str | None=None):
    model = SentenceTransformer("google/embeddinggemma-300m").to(device=device) 
    return model, device


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def batch_encode_gemma(texts: List[str], mdl, device:str, batch_size: int =  16) -> np.ndarray:
    embs = []
    # pdb.set_trace()
    for i in range(0, len(texts), batch_size):
        chunk = texts[i: i+batch_size]
        with torch.no_grad():
            out = mdl.encode(chunk)
        embs.append(out)
    return np.concatenate(embs, axis=0)
            

def batch_encode(texts: List[str], tok, mdl, device: str, max_len: int = 512, batch_size: int = 16) -> np.ndarray:
    embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        inputs = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = mdl(**inputs)
            reps = mean_pool(out.last_hidden_state, inputs["attention_mask"])
            reps = torch.nn.functional.normalize(reps, p=2, dim=1)
        embs.append(reps.cpu().numpy())
    return np.concatenate(embs, axis=0)

def main():
    ap = argparse.ArgumentParser()
    # arguments for both modes
    ap.add_argument("--out", type=str, default="evaluation_results/similarities") #"DamageTrials/DamageJsons/", help="Output directory")
    ap.add_argument("--ndim", type=int, default=3, choices=[2,3], help="PCA Dimension")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for encoding")
    ap.add_argument("--model", type=str, default="distilbert", choices=["distilbert", "embeddinggemma", "projector"], help="Model to use for encoding")
    ap.add_argument("--checkpoint", type=str, help="Path to your trained .pth model checkpoint")
    ap.add_argument("--config", type=str, help="Path to your pre_train_config.yaml file")

    #mode flag
    ap.add_argument("--from-jsons", action="store_true", help="Enable JSON processing mode. Reads from --folder")

    #json specific arguments
    ap.add_argument("--folder", type=str, help="Folder containing *.json", default="evaluation_results") #DamageTrials/DamageJsons/")
    ap.add_argument("--pattern", type=str, default="*.json", help="Type of file to get")
    ap.add_argument("--keep-cosmetic", action="store_true", help="Keep cosmetic parts in text", default=False)
    ap.add_argument("--topk", type=int, default=48, help="Top-K records to keep in text")
    ap.add_argument("--parse-level", type=str, default="none", choices=["none", "mid", "all"], help="Which level of damage info to parse from JSONs")

    #text file specific arguments
    ap.add_argument("--texts", type=str, default='DamageTrials/DamageJsons/texts_vivid.txt',
                help="Path to texts.txt (each line: '<label>: <text>'); "
                         "defaults to <out>/texts.txt if omitted.")

    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    labels: List[str] = []
    texts: List[str] = []
    laddition : List[str] = [] 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.from_jsons:
        print("Running in JSON processing node...")


        dmg_folder = Path(args.folder) / "damage_reports"
        traj_folder = Path(args.folder) / "trajectories"
        json_files = sorted(list(dmg_folder.glob(str(args.pattern))))
        if not json_files:
            print(f"Error: No files found matching '{args.pattern} in '{dmg_folder}'.")
            return
        
        payloads: List[Dict[str, Any]] = []
        trajectories: List[np.ndarray] = []
        for f_json in json_files:
            data = json.loads(f_json.read_text())
            payloads.append(data)
            labels.append(f_json.stem)

            #load the trajectory
            f_traj = traj_folder / f"{f_json.stem}.pkl"
            if f_traj.exists():
                with open(f_traj, "rb") as f:
                    traj_data = pickle.load(f)
                    trajectories.append(np.array(traj_data["trajectory"]))
            else:
                print(f"Warning: Trajectory file not found for {f_json.stem}")
                trajectories.append(np.array([[0,0]]))
        
        if args.parse_level == "all": 
            laddition.append("_all")
            texts = [all_text_from_vehicle_sensors(pl, K=args.topk, keep_cosmetic=args.keep_cosmetic) for pl in payloads]
        elif args.parse_level == "mid":
            laddition.append("_mid")
            texts = [mid_text_from_vehicle_sensors(pl, K=args.topk, keep_cosmetic=args.keep_cosmetic) for pl in payloads]
        elif args.parse_level == "none":
            #So that texts is a list of string and not json
            laddition.append("_none")
            texts = [json.dumps(pl) for pl in payloads]
        
        if args.keep_cosmetic:
            laddition[0] += "_y_cosmetic"
        else:
            laddition[0] += "_n_cosmetic"

        generated_texts_path =  outdir / f"texts_{laddition[0]}.txt"
        generated_texts_path.write_text("\n".join(f"{lab}: {txt}" for lab, txt in zip(labels, texts)))
        print(f"Generated {len(texts)} text records and saved to {generated_texts_path}")
    
    else:
        laddition.append("_file")
        print("Running in text file analysis mode...")
        txt_file =  Path(args.texts) if args.texts else (outdir / "texts.txt")
        if not txt_file.exists():
            print(f"Error: Text file is not found at {txt_file}")
            return
        
        with txt_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if ":" in line:
                    lab, txt = line.split(":", 1)
                    labels.append(lab.strip())
                    texts.append(txt.strip())
                else:
                    labels.append(f"item_{len(labels)}")
                    texts.append(line)
        print(f"Loaded {len(labels)} items from {txt_file}")

    projector = None
    if args.model == "projector":
        if not args.checkpoint or not args.config:
            print("Error: --checkpoint and --config are required when using --model projector")
            return

        print(f"Loading trained projector from {args.checkpoint}...")
        
        # 1. Load the training configuration
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

        # 2. Instantiate the full model
        # The device will be determined later, so instantiate on CPU for now

        model = DamagedVehicleBehaviorModel(cfg).to(device)
        
        # 3. Load the saved weights (the state dictionary)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        
        # 4. Extract just the damage_encoder, set it to evaluation mode
        projector = model.damage_encoder
        projector.eval()
        print("Projector loaded successfully.")


    # Encode
    if args.model == "embeddinggemma":
        laddition.append("_embeddinggemma")
        mdl, device = load_embeedinggemma(device)
        embs = batch_encode_gemma(texts, mdl, device)
    elif args.model == "distilbert": 
        laddition.append("_distilbert")
        tok, mdl, device = load_distilbert(device)
        embs = batch_encode(texts, tok, mdl, device, batch_size=args.batch_size)
    elif args.model == "projector":
        laddition.append("_projector")
        # Step 1: Get the base embeddings from embedding-gemma
        print("Step 1: Getting base embeddings from embedding-gemma...")
        base_model, device = load_embeedinggemma(device)
        raw_embs_np = batch_encode_gemma(texts, base_model, device, batch_size=args.batch_size)
        
        # Step 2: Pass the base embeddings through your trained projector
        print("Step 2: Projecting embeddings with your trained model...")
        raw_embs_torch = torch.from_numpy(raw_embs_np).to(device)
        
        projected_embs_list = []
        with torch.no_grad():
            for i in range(0, len(raw_embs_torch), args.batch_size):
                chunk = raw_embs_torch[i:i+args.batch_size].to(device)
                projected_chunk = projector(chunk)
                projected_embs_list.append(projected_chunk.cpu().numpy())
        
        embs = np.concatenate(projected_embs_list, axis=0)


    # Similarity
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    sim = embs_norm @ embs_norm.T

    # PCA & plots
    proj, _, evr = pca_project(embs, ndim=args.ndim)
    tag = laddition[0] + "_" + laddition[1]
    if args.ndim == 2:
        scatter_2d(proj, labels, outdir / f"pca2_scatter_{tag}.png", args.model, tag)
    else:
        scatter_3d(proj, labels, outdir / f"pca3_scatter_{tag}.png", args.model, tag)
    heatmap(sim, labels, args.model, outdir / f"sim_heatmap_{tag}.png", tag)

    print("\n Calculating trajectory distance matrix.....")
    num_trajectories = len(trajectories)
    traj_dist_matrix = np.zeros((num_trajectories, num_trajectories))
    for i in range(num_trajectories):
        for j in range(i, num_trajectories):
            dist = calculate_trajectory_distance(trajectories[i], trajectories[j])
            traj_dist_matrix[i, j] = dist
            traj_dist_matrix[j, i] = dist
    
    max_dist = traj_dist_matrix.max()
    traj_sim_matrix = 1.0 - (traj_dist_matrix / (max_dist + 1e-9))

    heatmap(traj_sim_matrix, labels, args.model, outdir / f"trajectory_sim_heatmap_{tag}.png", "Trajectory Similarity")
    print(f"Genrated trajectory similarity heatmap")

    # Console summary
    print(f"Top-{args.ndim} PCs explained variance ratio: {evr}")
    for i, lab in enumerate(labels):
        row = sim[i].copy(); row[i] = -1
        j = int(np.argmax(row))
        print(f"NN({lab}) -> {labels[j]} (cos={sim[i, j]:.3f})")
    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()

