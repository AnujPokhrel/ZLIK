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
from mpl_toolkits.mplot3d import Axes3D # noqa: F401
import pdb

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
    x = float(np.clip(x, 0.0, 1.0))
    if x == 0:
        return "none"
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

        #doing this because I am getting "front front left" etc
        if any(tok in part for tok in ["front", "rear", "left", "right", "fl", "fr", "rl", "rr"]):
            side_words = ""

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