"""Validated paired design and resumable generation with immutable run configuration."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import wavfile

METRICS = {"loudness": ("rms_dbfs", 1.0), "pitch": ("f0_hz", 50.0),
           "rate": ("event_rate_hz", 0.25)}


def save_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temporary.replace(path)


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    required = {"model_id", "revision", "duration_s", "steps", "guidance_scales", "seeds", "pairs"}
    if not required <= cfg.keys():
        raise ValueError(f"Missing config keys: {required-cfg.keys()}")
    if any(not isinstance(cfg[k], str) or not cfg[k].strip() for k in ("model_id", "revision")):
        raise ValueError("model_id and revision must be nonempty strings")
    if not isinstance(cfg["duration_s"], (int, float)) or not 1 <= cfg["duration_s"] <= 30:
        raise ValueError("duration_s must be between 1 and 30")
    if type(cfg["steps"]) is not int or cfg["steps"] < 1:
        raise ValueError("steps must be a positive integer")
    for key in ("seeds", "guidance_scales", "pairs"):
        if not isinstance(cfg[key], list) or not cfg[key]:
            raise ValueError(f"{key} must be a nonempty list")
    if any(type(s) is not int or s < 0 or s >= 2**32 for s in cfg["seeds"]):
        raise ValueError("seeds must be integers in [0, 2**32)")
    if len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Duplicate seeds")
    gs = cfg["guidance_scales"]
    if any(not isinstance(g, (int, float)) or not math.isfinite(g) or g < 0 for g in gs):
        raise ValueError("guidance_scales must be finite nonnegative numbers")
    if len(set(gs)) != len(gs):
        raise ValueError("Duplicate guidance scales")
    ids = set()
    for pair in cfg["pairs"]:
        pid = pair["id"]
        if not re.fullmatch(r"[a-z0-9_-]+", pid) or pid in ids:
            raise ValueError("Pair ids must be unique safe slugs")
        ids.add(pid)
        if pair["attribute"] not in METRICS:
            raise ValueError("Unknown attribute")
        if any(not isinstance(pair.get(k), str) or not pair[k].strip() for k in ("low", "high")):
            raise ValueError("Each pair needs two nonempty prompts")
        if pair["low"] == pair["high"]:
            raise ValueError("Paired prompts must differ")
    return cfg


def design(cfg):
    for pair in cfg["pairs"]:
        for guidance in cfg["guidance_scales"]:
            for seed in cfg["seeds"]:
                for level in ("low", "high"):
                    key = f'{pair["id"]}_g{guidance:g}_s{seed}_{level}'
                    yield {"id": key, "pair_id": pair["id"], "attribute": pair["attribute"],
                           "level": level, "prompt": pair[level], "seed": seed,
                           "guidance_scale": guidance, "file": f"audio/{key}.wav"}


def synthetic_control(item, duration, sample_rate=16000):
    """Known waveforms for software calibration ONLY, not a learned generator."""
    t = np.arange(round(sample_rate*duration))/sample_rate
    high = item["level"] == "high"
    # Separate control families from the natural-language prompt semantics.
    if item["attribute"] == "pitch":
        x = 0.2*np.sin(2*np.pi*(660 if high else 220)*t)
    elif item["attribute"] == "loudness":
        x = (0.2 if high else 0.05)*np.sin(2*np.pi*440*t)
    else:
        x = np.zeros_like(t)
        for onset in np.arange(0.3, duration-0.15, 1/(4 if high else 1)):
            envelope = np.exp(-0.5*((t-onset)/0.018)**2)
            x += 0.4*envelope*np.sin(2*np.pi*440*t)
    x *= np.minimum(1, t/0.02)*np.minimum(1, (duration-t)/0.02)
    return x.astype(np.float32), sample_rate


class AudioLDM2Generator:
    """Thin inference adapter: no training, fine-tuning or audio editing is claimed."""
    def __init__(self, cfg, device, offload):
        try:
            import torch
            from diffusers import AudioLDM2Pipeline
            from huggingface_hub import model_info
        except ImportError as e:
            raise RuntimeError("Install the generation extra: pip install -e '.[generation]'") from e
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use a GPU runtime or explicitly --device cpu")
        if offload and device != "cuda":
            raise ValueError("--offload requires CUDA")
        self.torch = torch
        self.device = device
        # Resolve a branch/tag ONCE, then load all weights from that exact commit.
        self.revision = model_info(cfg["model_id"], revision=cfg["revision"]).sha
        self.pipe = AudioLDM2Pipeline.from_pretrained(
            cfg["model_id"], revision=self.revision,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        if offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.sample_rate = int(self.pipe.vocoder.config.sampling_rate)

    def __call__(self, item, cfg):
        # Reset for EVERY level: common random numbers, not guaranteed semantic identity.
        generator = self.torch.Generator(device=self.device).manual_seed(item["seed"])
        with self.torch.inference_mode():
            output = self.pipe(prompt=item["prompt"], generator=generator,
                               audio_length_in_s=cfg["duration_s"],
                               num_inference_steps=cfg["steps"],
                               guidance_scale=item["guidance_scale"],
                               negative_prompt=cfg.get("negative_prompt"),
                               num_waveforms_per_prompt=1).audios[0]
        return np.asarray(output, dtype=np.float32), self.sample_rate


def generate(config_path, output, backend="synthetic", device="cuda", offload=False):
    cfg = load_config(config_path)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    path = root/"manifest.json"
    request = {"config": cfg, "backend": backend, "device": device if backend != "synthetic" else "cpu",
               "offload": offload if backend != "synthetic" else False}
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Run directory belongs to a different configuration; use a new directory")
    else:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                             text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        versions = {}
        for package in ("numpy", "scipy", "torch", "diffusers", "transformers", "accelerate"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        manifest = {**request, "fingerprint": fingerprint, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "python": platform.python_version(), "packages": versions, "git_commit": commit,
                    "status": "incomplete", "records": [],
                    "evidence": "synthetic software calibration only" if backend == "synthetic" else "pretrained neural inference"}
        save_json(path, manifest)
    (root/"audio").mkdir(exist_ok=True)
    completed = {r["id"]: r for r in manifest["records"]}
    todo = []
    for item in design(cfg):
        if item["id"] in completed:
            wavpath = root/item["file"]
            if not wavpath.exists() or hashlib.sha256(wavpath.read_bytes()).hexdigest() != completed[item["id"]]["sha256"]:
                raise ValueError(f"Missing or changed saved audio: {item['id']}; use a new directory")
        else:
            todo.append(item)
    if not todo:
        manifest["status"] = "complete"
        save_json(path, manifest)
        print(f"Already complete: {len(completed)} clips")
        return manifest
    adapter = None
    if backend == "audioldm2":
        actual_cfg = dict(cfg)
        actual_cfg["revision"] = manifest.get("resolved_revision", cfg["revision"])
        adapter = AudioLDM2Generator(actual_cfg, device, offload)
        manifest["resolved_revision"] = adapter.revision
        manifest["hardware"] = adapter.torch.cuda.get_device_name(0) if device == "cuda" else platform.processor()
        save_json(path, manifest)
    elif backend != "synthetic":
        raise ValueError("Unknown backend")
    for index, item in enumerate(todo, 1):
        audio, sr = adapter(item, cfg) if adapter else synthetic_control(item, cfg["duration_s"])
        if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"Invalid generated waveform: {item['id']}")
        target = root/item["file"]
        temporary = target.with_suffix(".tmp")
        # Float WAV preserves amplitude: no per-file normalisation or silent clipping.
        wavfile.write(temporary, sr, audio.astype(np.float32))
        temporary.replace(target)
        manifest["records"].append({**item, "sample_rate": sr,
                                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
        save_json(path, manifest)
        print(f"[{index}/{len(todo)}] {item['id']}", flush=True)
    manifest["status"] = "complete"
    save_json(path, manifest)
    return manifest
