# Initial execution record

Executed on 2026-10-10 in a CPU-only Python environment without PyTorch,
Diffusers, Transformers or downloaded model weights.

## Executed

- 15 offline tests: PCM handling, amplitude scaling, F0 calibration at three
  sample rates, pulse counting, silence, invalid input, clipping, conservative
  failure accounting, bootstrap clustering, full pipeline, resume, configuration
  guards, interrupted finalization, incomplete-run rejection and checksum checks.
- Six analytic signals generated and evaluated using `configs/calibration.json`.
- Full benchmark plan checked: 216 clips, 1,080 requested seconds of audio.
- Python sources compiled; GPU notebook parsed as valid JSON (11 cells).

| Control | Expected measurement | Observed high-minus-low |
| --- | --- | --- |
| Amplitude 0.05 → 0.20 | `20 log10(4)` = 12.0412 dB | 12.0412 dB |
| Tone 220 → 660 Hz | 440 Hz | 439.9913 Hz |
| Pulse train, 5 s window | (19 − 5) / 5 = 2.8 events/s | 2.8 events/s |

Pulse trains use rates of 1 and 4 Hz, with fixed endpoint margins. Counting events
in a finite five-second window gives 5 and 19 detected events, so the observed
difference is 2.8 rather than 3.0 events/s. This boundary effect is explicit.

These results validate the chosen software checks on simple analytic signals.
They are not accuracy numbers for a neural model or natural environmental audio.
No confidence interval is reported for these deterministic calibration signals.

## Not executed / not established

- Installing the optional neural dependency stack and loading AudioLDM2 weights.
- GPU inference, memory requirements, wall-clock runtime, full neural benchmark.
- Natural-audio reliability of pitch and event-rate proxies.
- Human listening study or any generation-quality improvement.
- Training, fine-tuning, audio editing or comparison against other learned models.

The GPU notebook is a next execution step, not evidence that these checks passed.
Record its actual environment, errors and outputs when it is run.
