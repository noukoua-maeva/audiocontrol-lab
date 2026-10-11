# AudioControl Lab

**Do text-to-audio models change the acoustic attribute we ask them to change?**

An experimental toolkit by **Maëva Noukoua** for paired evaluation of loudness,
pitch and event-rate control in generated audio. It combines a PyTorch / Diffusers
inference adapter, explicit acoustic measurements, repeated-seed experiments,
cluster bootstrap summaries and a portable listening report.

> **Release status — research prototype.** The offline analysis pipeline and its
> tests have been executed. Included audio examples are **analytic calibration
> signals**, not neural generations. The AudioLDM2 adapter and GPU notebook are
> implemented but **have not yet been executed against pretrained weights**.
> No model performance, training, fine-tuning or novel model result is claimed.

## Research question

Consider two prompts that differ in one requested property:

| Attribute | Low condition | High condition | Acoustic proxy |
| --- | --- | --- | --- |
| Loudness | A quiet recording of rain falling steadily | A loud recording of rain falling steadily | Raw RMS in dBFS |
| Pitch | A low-pitched continuous pure electronic tone | A high-pitched continuous pure electronic tone | Periodic F0 in Hz |
| Event rate | A slow sequence of evenly spaced knocks | A fast sequence of evenly spaced knocks | Envelope peaks per second |

The project tests **directional control**, not whether generated sound is generally
good. A model can pass an acoustic check while generating the wrong source. Human
listening is therefore part of the protocol, not an optional claim of success.

## What is implemented

- A hand-authored prompt suite: **12 pairs / 24 prompts**, three attributes.
- A full design with **3 seeds × 3 guidance scales × 24 prompts = 216 clips**.
- A six-clip GPU smoke configuration and six-signal offline calibration.
- Matched seeds across conditions; one sample per prompt with no best-of-N selection.
- Resumable runs with configuration guards, SHA-256 checksums, package versions,
  model revision resolution and saved raw float WAV files.
- Per-clip measurements, paired differences and threshold-based success checks.
- Confidence intervals resampled at the prompt-pair level; missing measurements
  remain visible and count as failures in total success rates.
- A self-contained HTML report with embedded audio and an attribute filter.
- A manual listening-ratings CSV template and a GPU notebook.
- Offline correctness tests and a GitHub Actions workflow.

## Quick start: no GPU or model downloads

Use Python 3.10+ (Python 3.11 is a suitable starting point).

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
python -m audiocontrol.cli generate --config configs/calibration.json --backend synthetic --out runs/calibration
python -m audiocontrol.cli evaluate --out runs/calibration
```

Open `runs/calibration/report.html`. It contains analytic tones and pulses and
explicitly labels them as **software calibration only**. The synthetic backend
uses the configured attribute/level, not a language model or prompt parser.

## Real generation: GPU path

Read [the protocol](docs/PROTOCOL.md) and the [model card](https://huggingface.co/cvssp/audioldm2)
first. The first run downloads pretrained model files. No paid service is required
by this repository, but GPU providers may charge; choose your own runtime.

```bash
python -m pip install -e '.[generation]'
python -m audiocontrol.cli plan --config configs/smoke.json
python -m audiocontrol.cli generate --config configs/smoke.json --backend audioldm2 --device cuda --offload --out runs/audioldm2-smoke
python -m audiocontrol.cli evaluate --out runs/audioldm2-smoke
```

Alternatively open [`notebooks/run_gpu.ipynb`](notebooks/run_gpu.ipynb) in a GPU
notebook environment. Its default is the six-clip smoke experiment. The larger
experiment is behind an explicit `RUN_FULL_BENCHMARK = False` switch.

After validating the smoke run:

```bash
python -m audiocontrol.cli plan --config configs/benchmark.json
python -m audiocontrol.cli generate --config configs/benchmark.json --backend audioldm2 --device cuda --offload --out runs/audioldm2-benchmark
python -m audiocontrol.cli evaluate --out runs/audioldm2-benchmark
```

The benchmark synthesizes 1,080 seconds of audio; **that is not a wall-clock
runtime estimate**. Actual compute time and memory depend on the GPU and library
versions. CPU inference is explicit (`--device cpu`) and may be very slow.

Run the same generation command to resume. A run directory refuses changed
configurations and altered/missing completed audio. Use a new directory when
changing the experiment. If a download or GPU error interrupts generation, its
manifest remains incomplete and cannot be scored as a finished experiment.

## Outputs

| File | Contents |
| --- | --- |
| `manifest.json` | Configuration, versions, resolved model revision, hashes, progress |
| `audio/*.wav` | Raw amplitude-preserving mono audio |
| `metrics.csv` | Clip-level measurements and provenance |
| `paired.csv` | High-minus-low differences and failures |
| `summary.json`, `summary.csv` | Per-attribute / guidance summaries |
| `report.html` | Portable report with all audio embedded |
| `listening_ratings.csv` | Unfilled human rating template; never overwritten |

The report embeds all waveforms and can be large for the full experiment.
Synthetic calibration evidence is in [`examples/calibration/`](examples/calibration/).
The initial execution record is in [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Why these design choices?

**Matched seeds.** Resetting the generator for each paired prompt reduces one
source of variation. It does not guarantee that the same scene or source identity
is preserved across prompts.

**Guidance sweep.** The AudioLDM2 adapter uses guidance values 2.0, 3.5 and 5.0.
This is a prespecified inference-parameter comparison, not training or a new method.
Comparisons should use the same prompt pairs and seeds at every setting.

**No individual normalisation.** Normalising every clip before analysis or playback
would destroy an amplitude-control comparison. Raw WAV values are preserved.

**Prompt-pair bootstrap.** Repeated seeds of one prompt are not new independent
language examples. We average within a prompt pair and resample prompt pairs.
Four pairs per attribute remain a small exploratory sample; intervals are fragile.

**Transparent proxies.** F0 and event-count failures are exposed. No automatic
quality score is substituted for a listener. RMS alone does not establish perceived
loudness, and a high F0 does not establish semantic correctness.

## Scope and next experiments

This release covers **text-conditioned generation and evaluation**, not editing an
existing recording. Before making model-performance claims:

1. Execute and audit the real six-clip smoke run.
2. Calibrate metrics on separate development audio, then freeze the protocol.
3. Run the complete neural design; retain every seed and failure.
4. Collect independently ordered, blinded listening judgments with multiple raters.
5. Expand held-out prompt families and compare another model or one justified method.

No state-of-the-art claim, application outcome or publication is implied.
See [the French walkthrough](docs/GUIDE_FR.md) for an explanation of the code and
what can honestly be said about this project in an interview.

## References and licenses

- Liu et al., [AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734).
- [Official Diffusers AudioLDM2 API](https://huggingface.co/docs/diffusers/api/pipelines/audioldm2).
- [AudioLDM2 model card and weight license](https://huggingface.co/cvssp/audioldm2).
- [Original AudioLDM2 repository](https://github.com/haoheliu/AudioLDM2).

Original project code and prompt definitions: MIT, see [LICENSE](LICENSE).
External models, packages and generated outputs retain their applicable terms;
the code license does not relicense model weights. No model weights or downloaded
third-party datasets are redistributed. Development was assisted by OpenAI Codex;
experimental claims must be supported by the saved run evidence.
