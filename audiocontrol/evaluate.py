"""Paired analysis, cluster bootstrap and a self-contained listening report."""
from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .experiment import METRICS, design, save_json
from .metrics import measure


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cluster_interval(rows, key, seed=2026, repetitions=2000):
    """Resample prompt pairs, retaining their seed replicates as one cluster.

    Descriptive interval for the selected prompt set, not a population guarantee.
    Undefined when fewer than two distinct prompt pairs have usable observations.
    """
    groups = defaultdict(list)
    for row in rows:
        if row[key] is not None:
            groups[row["pair_id"]].append(float(row[key]))
    means = np.array([np.mean(values) for values in groups.values()])
    if not len(means):
        return None, None, None, 0
    point = float(means.mean())
    if len(means) < 2:
        return point, None, None, len(means)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(means, size=(repetitions, len(means)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return point, float(low), float(high), len(means)


def pair_results(metrics):
    grouped = defaultdict(dict)
    for row in metrics:
        key = row["pair_id"], row["guidance_scale"], row["seed"]
        if row["level"] in grouped[key]:
            raise ValueError("Duplicate paired observation")
        grouped[key][row["level"]] = row
    results = []
    for (pid, guidance, seed), levels in grouped.items():
        if set(levels) != {"low", "high"}:
            raise ValueError(f"Incomplete pair: {pid}")
        low, high = levels["low"], levels["high"]
        metric, threshold = METRICS[low["attribute"]]
        usable = not low["silent"] and not high["silent"] and low[metric] is not None and high[metric] is not None
        delta = high[metric]-low[metric] if usable else None
        # Missing/silent pairs count as failures in the conservative total success rate.
        results.append({"pair_id": pid, "attribute": low["attribute"], "guidance_scale": guidance,
                        "seed": seed, "metric": metric, "threshold": threshold,
                        "low_value": low[metric], "high_value": high[metric], "delta": delta,
                        "usable": usable, "success": bool(usable and delta > threshold),
                        "low_clipping_fraction": low["clipping_fraction"],
                        "high_clipping_fraction": high["clipping_fraction"]})
    return results


def evaluate(output):
    root = Path(output)
    manifest = json.loads((root/"manifest.json").read_text())
    expected = {r["id"]: r for r in design(manifest["config"])}
    records = manifest["records"]
    if manifest["status"] != "complete" or len(records) != len(expected) or {r["id"] for r in records} != set(expected):
        raise ValueError("Run is incomplete; finish generation before evaluating")
    metrics = []
    for record in records:
        if any(record[k] != v for k, v in expected[record["id"]].items()):
            raise ValueError("Manifest record does not match experiment design")
        path = root/record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Audio integrity check failed: {path.name}")
        sr, audio = wavfile.read(path)
        if sr != record["sample_rate"]:
            raise ValueError("Sample-rate metadata mismatch")
        metrics.append({**record, **measure(audio, sr)})
    paired = pair_results(metrics)
    summary = []
    for attribute in METRICS:
        for guidance in manifest["config"]["guidance_scales"]:
            rows = [r for r in paired if r["attribute"] == attribute and r["guidance_scale"] == guidance]
            if not rows:
                continue
            rate, lo, hi, clusters = cluster_interval(rows, "success")
            delta, dlo, dhi, valid_clusters = cluster_interval(rows, "delta")
            if manifest["backend"] == "synthetic":
                # Replicated analytic signals are not independent experimental evidence.
                lo = hi = dlo = dhi = None
            summary.append({"attribute": attribute, "guidance_scale": guidance,
                            "n_pair_seeds": len(rows), "n_prompt_pairs": clusters,
                            "n_usable": sum(r["usable"] for r in rows),
                            "success_rate": rate, "success_ci_low": lo, "success_ci_high": hi,
                            "mean_delta": delta, "delta_ci_low": dlo, "delta_ci_high": dhi,
                            "valid_prompt_pairs": valid_clusters, "metric": METRICS[attribute][0]})
    save_json(root/"summary.json", {"backend": manifest["backend"], "evidence": manifest["evidence"],
                                    "n_clips": len(metrics), "groups": summary})
    write_csv(root/"metrics.csv", metrics)
    write_csv(root/"paired.csv", paired)
    write_csv(root/"summary.csv", summary)
    # Template is never filled automatically: human assessments must come from listeners.
    rating_path = root/"listening_ratings.csv"
    if not rating_path.exists():
        write_csv(rating_path, [{"clip_id": r["id"], "rater_id": "", "source_matches_0_1": "",
                                "attribute_matches_1_5": "", "quality_1_5": "", "notes": ""} for r in records])
    render_report(root, manifest, metrics, paired, summary)
    print(f"Evaluated {len(metrics)} clips. Report: {root/'report.html'}")
    return summary


def render_report(root, manifest, metrics, paired, summary):
    escape = html.escape
    synthetic = manifest["backend"] == "synthetic"
    title = "Synthetic calibration" if synthetic else "AudioLDM2 paired experiment"
    notice = ("Analytic tones and pulses only. These clips test the software; they are NOT model outputs, "
              "natural sound examples, or evidence of text understanding.") if synthetic else (
              "Exploratory acoustic proxies, not perceptual quality or semantic accuracy. "
              "Listen to all clips and record human ratings before drawing conclusions.")
    rows_html = []
    def fmt(v):
        return "n/a" if v is None else f"{v:.3f}"
    for row in summary:
        rows_html.append(f'<tr><td>{escape(row["attribute"])}</td><td>{row["guidance_scale"]}</td>'
                         f'<td>{row["n_usable"]}/{row["n_pair_seeds"]}</td>'
                         f'<td>{fmt(row["success_rate"])}</td><td>{fmt(row["mean_delta"])}</td>'
                         f'<td>{fmt(row["delta_ci_low"])} — {fmt(row["delta_ci_high"])}</td></tr>')
    index = {(r["pair_id"], r["guidance_scale"], r["seed"], r["level"]): r for r in metrics}
    cards = []
    for pair in paired:
        clips = []
        for level in ("low", "high"):
            record = index[pair["pair_id"], pair["guidance_scale"], pair["seed"], level]
            data = base64.b64encode((root/record["file"]).read_bytes()).decode()
            label = f'Analytic {record["attribute"]} control: {level}' if synthetic else record["prompt"]
            clips.append(f'<div><h3>{level.upper()}</h3><p>{escape(label)}</p>'
                         f'<audio controls preload="none" src="data:audio/wav;base64,{data}"></audio>'
                         f'<small>{escape(pair["metric"])}: {fmt(record[pair["metric"]])} · '
                         f'clipped samples: {record["clipping_fraction"]:.4%}</small></div>')
        cards.append(f'<article data-attribute="{escape(pair["attribute"])}"><header>'
                     f'<b>{escape(pair["pair_id"])}</b><span>Guidance {pair["guidance_scale"]} · seed {pair["seed"]}</span>'
                     f'</header><div class="clips">{"".join(clips)}</div><footer>'
                     f'High − low: {fmt(pair["delta"])} · required increase &gt; {pair["threshold"]} · '
                     f'{"PASS" if pair["success"] else "FAIL / UNUSABLE"}</footer></article>')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AudioControl Lab</title><style>
:root{color-scheme:light;font-family:system-ui,sans-serif;color:#182b3a;background:#eef3f6}
body{max-width:1080px;margin:0 auto;padding:32px 20px}h1{font-size:42px;letter-spacing:-1.5px;margin-bottom:8px}
.eyebrow{color:#006b70;letter-spacing:3px;font-weight:700;font-size:12px}.lead{font-size:19px;color:#496170}
.notice{border-left:5px solid #d69720;background:#fff5da;padding:18px;line-height:1.6;margin:24px 0}
table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #e0e7ed}
.table{overflow:auto}article{background:white;border:1px solid #dbe4e9;border-radius:14px;margin:18px 0;overflow:hidden}
header,footer{padding:15px 20px;background:#f8fafb;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
.clips{display:grid;grid-template-columns:1fr 1fr;gap:24px;padding:20px}.clips p{min-height:48px;line-height:1.5}
h3{font-size:12px;color:#006b70;letter-spacing:2px}audio{width:100%;margin:8px 0}small{display:block;color:#607480}
select{padding:10px;border-radius:8px;border:1px solid #bccbd4;background:white}details{margin:20px 0}pre{overflow:auto;font-size:12px}
@media(max-width:640px){.clips{grid-template-columns:1fr}h1{font-size:32px}body{padding:20px 12px}}
</style><p class="eyebrow">AUDIOCONTROL LAB / EXPERIMENT REPORT</p>'''
    document += f'<h1>{title}</h1><p class="lead">Paired prompts. Repeated seeds. Inspectable evidence.</p><div class="notice">{notice}</div>'
    document += f'<p>{len(metrics)} clips · {len(manifest["config"]["pairs"])} prompt pairs · '
    document += f'Backend: <b>{escape(manifest["backend"])}</b></p>'
    document += '<h2>Directional control checks</h2><p>Success requires a positive increase beyond the prespecified threshold. Missing measurements count as failures. Intervals resample prompt pairs, not clips; n/a for calibration or fewer than two usable pairs.</p>'
    document += '<div class="table"><table><tr><th>Attribute</th><th>Guidance</th><th>Usable pairs</th><th>Success rate</th><th>Mean Δ</th><th>95% Δ interval</th></tr>'+''.join(rows_html)+'</table></div>'
    document += '<h2>Listen and inspect</h2><p>Start at a comfortable playback volume. RMS is digital amplitude, not calibrated perceived loudness. Pitch requires tonal audio; event counting is a heuristic.</p><label>Attribute <select id="filter"><option value="all">All attributes</option><option>loudness</option><option>pitch</option><option>rate</option></select></label>'
    document += ''.join(cards)
    document += '<details><summary>Run provenance</summary><pre>'+escape(json.dumps({k:v for k,v in manifest.items() if k != "records"}, indent=2))+'</pre></details>'
    document += '<script>document.getElementById("filter").addEventListener("change",e=>{document.querySelectorAll("article").forEach(a=>a.hidden=e.target.value!=="all"&&a.dataset.attribute!==e.target.value)})</script></html>'
    (root/"report.html").write_text(document)
