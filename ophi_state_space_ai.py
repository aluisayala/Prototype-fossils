#!/usr/bin/env python3
"""
OPHI State Space Image AI: Deterministic Ledger-First Manifold Renderer

This script generates a deterministic OPHI state-space run from explicit parameters,
fossilizes admissible states into a SHA-256 chained ledger, records rejected states
in a mutable shell, and renders images strictly from the ledger data.

Required output package:
  01_trajectory.png
  02_phase_portrait.png
  03_density_heatmap.png
  04_rgb_state_space_image.png
  ophi_state_space_ledger.json
  ophi_state_space_ledger.csv
  mutable_shell_rejections.json
  verification_report.json
  verification_report.txt

Example:
  python ophi_state_space_ai.py \
    --steps 200 \
    --seed 987654321 \
    --initial 0.731283 \
    --bias -0.00371 \
    --alpha 1.00297 \
    --reliability 0.99810 \
    --grounding 1.0 \
    --size 1024 \
    --out outputs_random_run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# -----------------------------
# OPHI constants / gate settings
# -----------------------------

SCALE = 10_000
COHERENCE_MIN = 9_850       # 0.985 scaled by 10,000
ENTROPY_MAX = 100           # 0.01 scaled by 10,000
DRIFT_MAX = 10              # 0.001 scaled by 10,000
GENESIS_HASH = "0" * 64
BASE_TIMESTAMP = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

CODON_CYCLE = ["ATG", "CCC", "TTG", "AAA", "TTC", "GAC", "ATC", "CTT"]
GLYPH_MAP = {
    "ATG": "⧖⧖",   # bootstrap / origin
    "CCC": "⧃⧃",   # fossil lock
    "TTG": "⧖⧊",   # uncertainty translator
    "AAA": "⧉",    # signal carrier
    "TTC": "⧈",    # collapse suppression
    "GAC": "⧆",    # grounding / binding
    "ATC": "⧇",    # drift branch
    "CTT": "⧊",    # transition path
}


@dataclass(frozen=True)
class LedgerEntry:
    step: int
    scaled_z: int
    scaled_omega: int
    scaled_candidate: int
    scaled_noise: int
    coherence: int
    entropy: int
    drift: int
    admissible: bool
    promotion_status: str
    timestamp_utc: str
    codon: str
    glyphs: str
    previous_hash_sha256: str
    hash_sha256: str


# -----------------------------
# Deterministic helpers
# -----------------------------

def scaled(value: float) -> int:
    return int(round(value * SCALE))


def unscaled(value: int) -> float:
    return value / SCALE


def timestamp_for_step(step: int) -> str:
    return (BASE_TIMESTAMP + timedelta(seconds=step)).isoformat().replace("+00:00", "Z")


def codon_for_step(step: int) -> Tuple[str, str]:
    codon = CODON_CYCLE[step % len(CODON_CYCLE)]
    return codon, GLYPH_MAP[codon]


def canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_entry_payload(entry_without_hash: Dict[str, Any]) -> str:
    """Hash the previous hash plus canonical payload to form a Merkle-style chain."""
    previous_hash = entry_without_hash["previous_hash_sha256"]
    return sha256_bytes(previous_hash.encode("utf-8") + canonical_json_bytes(entry_without_hash))


def deterministic_noise_scaled(seed: int, step: int, amplitude: int) -> int:
    """
    Generate deterministic pseudo-random integer perturbation.

    The seed is always active. With the default amplitude of 1, perturbation is
    tiny enough to remain within SE44-style drift gates for stable parameters,
    but it still makes --seed materially affect the trajectory and final hashes.
    """
    if amplitude <= 0:
        return 0
    raw = hashlib.sha256(f"OPHI|seed={seed}|step={step}".encode("utf-8")).digest()
    value = int.from_bytes(raw[:8], byteorder="big", signed=False)
    return int(value % (2 * amplitude + 1)) - amplitude


def gate_values(scaled_z: int, scaled_candidate: int, prior_delta: int) -> Tuple[int, int, int, bool]:
    drift = abs(scaled_candidate - scaled_z)
    delta = scaled_candidate - scaled_z
    entropy = abs(delta - prior_delta)
    coherence = max(0, 10_000 - drift * 10 - entropy)
    admissible = coherence >= COHERENCE_MIN and entropy <= ENTROPY_MAX and drift <= DRIFT_MAX
    return coherence, entropy, drift, admissible


def make_entry(
    *,
    step: int,
    s_z: int,
    s_omega: int,
    s_candidate: int,
    s_noise: int,
    coherence: int,
    entropy: int,
    drift: int,
    admissible: bool,
    status: str,
    prev_hash: str,
) -> LedgerEntry:
    codon, glyphs = codon_for_step(step)
    payload: Dict[str, Any] = {
        "step": step,
        "scaled_z": s_z,
        "scaled_omega": s_omega,
        "scaled_candidate": s_candidate,
        "scaled_noise": s_noise,
        "coherence": coherence,
        "entropy": entropy,
        "drift": drift,
        "admissible": admissible,
        "promotion_status": status,
        "timestamp_utc": timestamp_for_step(step),
        "codon": codon,
        "glyphs": glyphs,
        "previous_hash_sha256": prev_hash,
    }
    payload["hash_sha256"] = hash_entry_payload(payload)
    return LedgerEntry(**payload)


# -----------------------------
# Core OPHI run
# -----------------------------

def generate_run(args: argparse.Namespace) -> Tuple[List[LedgerEntry], List[LedgerEntry]]:
    fossil_ledger: List[LedgerEntry] = []
    mutable_shell: List[LedgerEntry] = []

    current_z = float(args.initial)
    prior_delta = 0
    prev_hash = GENESIS_HASH

    for step in range(args.steps):
        omega = (current_z + args.bias) * args.alpha * args.reliability * args.grounding
        # Promotion is damped: Ω is the raw operator output, while candidate is
        # the SE44-compatible committed transition proposed from the current state.
        candidate = current_z + (omega - current_z) * args.promotion_gain

        s_z = scaled(current_z)
        s_omega = scaled(omega)
        s_noise = deterministic_noise_scaled(args.seed, step, args.noise_amplitude_scaled)
        s_candidate = scaled(candidate) + s_noise

        coherence, entropy, drift, admissible = gate_values(s_z, s_candidate, prior_delta)
        status = "FOSSILIZED" if admissible else "REJECTED_TO_MUTABLE_SHELL"

        entry = make_entry(
            step=step,
            s_z=s_z,
            s_omega=s_omega,
            s_candidate=s_candidate,
            s_noise=s_noise,
            coherence=coherence,
            entropy=entropy,
            drift=drift,
            admissible=admissible,
            status=status,
            prev_hash=prev_hash,
        )

        if admissible:
            fossil_ledger.append(entry)
            prev_hash = entry.hash_sha256
            prior_delta = s_candidate - s_z
            current_z = unscaled(s_candidate)
        else:
            mutable_shell.append(entry)
            # Rejections do not mutate the fossilized state, prior delta, or hash chain.

    return fossil_ledger, mutable_shell


# -----------------------------
# Rendering
# -----------------------------

def style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor("#05070d")
    ax.set_title(title, color="white", fontfamily="monospace", fontsize=12)
    ax.tick_params(colors="#cfd6ff", labelsize=8)
    ax.xaxis.label.set_color("#cfd6ff")
    ax.yaxis.label.set_color("#cfd6ff")
    for spine in ax.spines.values():
        spine.set_color("#39fff2")
        spine.set_linewidth(0.8)
    ax.grid(color="#1f3350", linewidth=0.45, alpha=0.6)


def ledger_series(fossil_ledger: List[LedgerEntry], initial: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = [scaled(initial)] + [entry.scaled_candidate for entry in fossil_ledger]
    steps = np.arange(len(z), dtype=np.int64)
    coherence = np.array([entry.coherence for entry in fossil_ledger], dtype=np.int64)
    entropy = np.array([entry.entropy for entry in fossil_ledger], dtype=np.int64)
    drift = np.array([entry.drift for entry in fossil_ledger], dtype=np.int64)
    return steps, np.array(z, dtype=np.int64), coherence, entropy, drift


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def render_images(out: Path, fossil_ledger: List[LedgerEntry], initial: float, size: int) -> None:
    steps, z, coherence, entropy, drift = ledger_series(fossil_ledger, initial)

    # 01 Trajectory
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor("#02030a")
    style_axes(ax, "01 Trajectory: scaled z[n] vs fossil index")
    ax.plot(steps, z, linewidth=1.7)
    ax.set_xlabel("Fossil index")
    ax.set_ylabel("z × 10,000")
    save_fig(fig, out / "01_trajectory.png")

    # 02 Phase Portrait
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor("#02030a")
    style_axes(ax, "02 Phase Portrait: z[n] vs z[n+1]")
    if len(z) > 1:
        ax.scatter(z[:-1], z[1:], c=np.arange(len(z) - 1), cmap="turbo", s=18)
    ax.set_xlabel("z[n] × 10,000")
    ax.set_ylabel("z[n+1] × 10,000")
    save_fig(fig, out / "02_phase_portrait.png")

    # 03 Density Heatmap
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor("#02030a")
    style_axes(ax, "03 Manifold Density Heatmap: fossil index vs scaled state")
    if len(z) > 2:
        bins_x = max(8, min(96, int(math.sqrt(len(z))) * 2))
        bins_y = max(8, min(96, int(math.sqrt(len(z))) * 2))
        ax.hist2d(steps, z, bins=[bins_x, bins_y], cmap="magma")
    else:
        ax.scatter(steps, z, s=18)
    ax.set_xlabel("Fossil index")
    ax.set_ylabel("z × 10,000")
    save_fig(fig, out / "03_density_heatmap.png")

    # 04 RGB State Space Image
    rgb = make_rgb_state_space_image(fossil_ledger, initial, size)
    Image.fromarray(rgb, mode="RGB").save(out / "04_rgb_state_space_image.png")


def normalize_to_uint8(values: np.ndarray, fallback: int = 0) -> np.ndarray:
    if values.size == 0:
        return np.array([fallback], dtype=np.uint8)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax == vmin:
        return np.full(values.shape, fallback, dtype=np.uint8)
    return np.clip(((values - vmin) / (vmax - vmin)) * 255, 0, 255).astype(np.uint8)


def make_rgb_state_space_image(fossil_ledger: List[LedgerEntry], initial: float, size: int) -> np.ndarray:
    """
    Construct a deterministic RGB image from ledger fields only.

    R channel: normalized fossilized state.
    G channel: normalized coherence.
    B channel: inverse normalized entropy + drift signature.
    """
    if size < 64:
        raise ValueError("--size must be at least 64")

    _, z, coherence, entropy, drift = ledger_series(fossil_ledger, initial)
    n = max(1, len(z))
    pixels = size * size

    r_base = normalize_to_uint8(z)
    if coherence.size == 0:
        g_base = np.array([0], dtype=np.uint8)
    else:
        g_base = normalize_to_uint8(coherence, fallback=255)
    if entropy.size == 0 and drift.size == 0:
        b_base = np.array([0], dtype=np.uint8)
    else:
        e = entropy if entropy.size else np.array([0], dtype=np.int64)
        d = drift if drift.size else np.array([0], dtype=np.int64)
        max_len = max(len(e), len(d))
        e_rep = np.resize(e, max_len)
        d_rep = np.resize(d, max_len)
        b_base = 255 - normalize_to_uint8(e_rep + d_rep, fallback=0)

    r = np.resize(r_base, pixels)
    g = np.resize(g_base, pixels)
    b = np.resize(b_base, pixels)

    rgb = np.stack([r, g, b], axis=1).reshape(size, size, 3)

    # Add deterministic structural folding from the ledger hash sequence.
    hash_stream = "".join(entry.hash_sha256 for entry in fossil_ledger) or GENESIS_HASH
    digest = hashlib.sha256(hash_stream.encode("utf-8")).digest()
    roll_x = digest[0] % size
    roll_y = digest[1] % size
    rgb = np.roll(rgb, shift=roll_x, axis=0)
    rgb = np.roll(rgb, shift=roll_y, axis=1)
    return rgb


# -----------------------------
# Artifact writers
# -----------------------------

def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: Iterable[LedgerEntry]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            writer = csv.writer(f)
            writer.writerow([field for field in LedgerEntry.__dataclass_fields__.keys()])
            return
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_outputs(args: argparse.Namespace, fossil_ledger: List[LedgerEntry], mutable_shell: List[LedgerEntry]) -> Dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ledger_json = out / "ophi_state_space_ledger.json"
    ledger_csv = out / "ophi_state_space_ledger.csv"
    rejections_json = out / "mutable_shell_rejections.json"

    write_json(ledger_json, [asdict(entry) for entry in fossil_ledger])
    write_csv(ledger_csv, fossil_ledger)
    write_json(rejections_json, [asdict(entry) for entry in mutable_shell])

    render_images(out, fossil_ledger, args.initial, args.size)

    artifact_names = [
        "01_trajectory.png",
        "02_phase_portrait.png",
        "03_density_heatmap.png",
        "04_rgb_state_space_image.png",
        "ophi_state_space_ledger.json",
        "ophi_state_space_ledger.csv",
        "mutable_shell_rejections.json",
    ]
    artifact_hashes = {name: file_sha256(out / name) for name in artifact_names}

    params = {
        "steps": args.steps,
        "seed": args.seed,
        "initial": args.initial,
        "bias": args.bias,
        "alpha": args.alpha,
        "reliability": args.reliability,
        "grounding": args.grounding,
        "size": args.size,
        "promotion_gain": args.promotion_gain,
        "noise_amplitude_scaled": args.noise_amplitude_scaled,
        "out": str(out),
    }

    report = {
        "script": "ophi_state_space_ai.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "data_schema_version": "v1.1",
        "model": "OPHI deterministic ledger-first state-space renderer",
        "operator": "Omega = (state + bias) * alpha * reliability * grounding",
        "scale": SCALE,
        "gate": {
            "coherence_min_scaled": COHERENCE_MIN,
            "coherence_min": COHERENCE_MIN / SCALE,
            "entropy_max_scaled": ENTROPY_MAX,
            "entropy_max": ENTROPY_MAX / SCALE,
            "drift_max_scaled": DRIFT_MAX,
            "drift_max": DRIFT_MAX / SCALE,
        },
        "parameters": params,
        "accepted_count": len(fossil_ledger),
        "rejected_count": len(mutable_shell),
        "genesis_hash_sha256": GENESIS_HASH,
        "ledger_root_hash_sha256": fossil_ledger[-1].hash_sha256 if fossil_ledger else GENESIS_HASH,
        "artifact_hashes_sha256": artifact_hashes,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }

    report_json = out / "verification_report.json"
    report_txt = out / "verification_report.txt"
    write_json(report_json, report)

    lines = [
        "OPHI STATE SPACE VERIFICATION REPORT",
        "====================================",
        f"Generated UTC: {report['generated_at_utc']}",
        f"Operator: {report['operator']}",
        f"Scale: {SCALE}",
        f"Gate: C >= {COHERENCE_MIN / SCALE:.4f}; S <= {ENTROPY_MAX / SCALE:.4f}; RMS/Drift <= {DRIFT_MAX / SCALE:.4f}",
        "",
        "Parameters:",
    ]
    for key, value in params.items():
        lines.append(f"  {key}: {value}")
    lines.extend([
        "",
        f"Accepted fossil states: {len(fossil_ledger)}",
        f"Rejected mutable-shell states: {len(mutable_shell)}",
        f"Ledger root/final hash: {report['ledger_root_hash_sha256']}",
        "",
        "Artifact SHA-256 hashes:",
    ])
    for name, digest in artifact_hashes.items():
        lines.append(f"  {name}: {digest}")
    lines.append("")
    report_txt.write_text("\n".join(lines), encoding="utf-8")

    # Add reports themselves after writing text/json.
    report["artifact_hashes_sha256"]["verification_report.json"] = file_sha256(report_json)
    report["artifact_hashes_sha256"]["verification_report.txt"] = file_sha256(report_txt)
    write_json(report_json, report)

    return report


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OPHI deterministic ledger-first state-space renderer")
    parser.add_argument("--steps", type=int, default=200, help="Number of attempted state transitions")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic perturbation vector")
    parser.add_argument("--initial", type=float, default=0.731283, help="Initial state")
    parser.add_argument("--bias", type=float, default=-0.00371, help="Bias term")
    parser.add_argument("--alpha", type=float, default=1.00297, help="Amplification scalar")
    parser.add_argument("--reliability", type=float, default=0.99810, help="Reliability scalar")
    parser.add_argument("--grounding", type=float, default=1.0, help="Grounding scalar")
    parser.add_argument("--size", type=int, default=1024, help="Width/height of RGB state-space image")
    parser.add_argument("--out", type=str, default="outputs_random_run", help="Output directory")
    parser.add_argument(
        "--promotion-gain",
        type=float,
        default=0.25,
        help="Damping factor from raw Ω output to committed candidate transition",
    )
    parser.add_argument(
        "--noise-amplitude-scaled",
        type=int,
        default=1,
        help="Deterministic seed-derived perturbation amplitude in scaled integer units",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Derive initial, bias, alpha, reliability from seed before running",
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.size < 64:
        raise ValueError("--size must be at least 64")
    if args.noise_amplitude_scaled < 0:
        raise ValueError("--noise-amplitude-scaled must be >= 0")
    if not (0.0 < args.promotion_gain <= 1.0):
        raise ValueError("--promotion-gain must be in the interval (0, 1]")

    if args.randomize:
        rng = np.random.default_rng(args.seed)
        args.initial = float(rng.uniform(0.1, 0.9))
        args.bias = float(rng.uniform(-0.01, 0.01))
        args.alpha = float(rng.uniform(0.999, 1.004))
        args.reliability = float(rng.uniform(0.996, 1.0))

    return args


def main() -> None:
    args = parse_args()
    fossil_ledger, mutable_shell = generate_run(args)
    report = write_outputs(args, fossil_ledger, mutable_shell)

    print("OPHI run complete.")
    print(f"Accepted fossil states: {report['accepted_count']}")
    print(f"Rejected mutable-shell states: {report['rejected_count']}")
    print(f"Ledger root/final hash: {report['ledger_root_hash_sha256']}")
    print(f"Outputs written to: {args.out}")


if __name__ == "__main__":
    main()
