# Experimental protocol v0.1

## Objective and unit of analysis

Test whether changing a textual acoustic attribute produces the corresponding
directional change in generated audio. Each unit is a low/high prompt pair,
generated at the same seed and classifier-free guidance setting. A 12-pair
hand-authored suite covers four source families for each of three attributes.
This is a small exploratory benchmark, not a validated general-purpose dataset.

The full design is 12 pairs × 2 levels × 3 seeds × 3 guidance values = 216 clips,
each requested at 5 seconds and 50 denoising steps. Seeds: 17, 29, 43. Guidance:
2.0, 3.5, 5.0. The same negative prompt applies to every condition. These choices
are workload defaults, not optimized hyperparameters or claims of sufficient quality.

## Primary comparisons

Compute `delta = measurement(high) - measurement(low)` on unnormalised waveforms.

| Attribute | Measurement | Prespecified success criterion |
| --- | --- | --- |
| Loudness | Full-clip RMS, `20 log10(sqrt(mean(x²)))` dBFS | Δ > 1 dB |
| Pitch | Median periodic frame F0 | Δ > 50 Hz |
| Rate | Peaks in a 10 ms RMS envelope / actual duration | Δ > 0.25 Hz |

Thresholds are engineering choices, not validated perceptual boundaries. Any
change must be made on development data and recorded as a new protocol version
before evaluating held-out audio. The published suite currently has no distinct
development/test split, so tuning on it makes subsequent findings exploratory.

## Signal handling

- Integer PCM is converted using its encoding range before mono channel averaging.
- Floating-point amplitude is retained without peak/RMS normalisation or clipping.
- Nonfinite waveforms and clips shorter than 250 ms are rejected.
- Silence is defined as RMS < 1e-5. Silent pairs are unusable for all comparisons.
- Record the fraction of samples with absolute value >= 0.999 as a clipping flag.
  This also flags full-scale float output; it is not proof of an audible artifact.
- F0: 64 ms frames, 50% overlap, 80–1200 Hz search range, autocorrelation peak
  >=0.6 of lag-zero energy, at least 25% periodic energetic frames. Report missing
  F0 explicitly. Octave errors, noise and polyphony remain limitations.
- Rate: 10 ms RMS frames, 15% relative prominence, minimum peak spacing 150 ms.
  Ringing can create false peaks and closely spaced events can merge. The metric
  saturates for rapid events; inspect envelope/listening disagreements.

## Aggregation and uncertainty

The success-rate denominator includes every designed pair/seed observation;
silent or unmeasurable pairs count as failures. Mean delta is conditional on valid
measurements, and its coverage is reported separately. Average seed replicates
within each prompt pair, then weight prompt pairs equally. Bootstrap prompt-pair
means 2,000 times with analysis seed 2026 and report percentile 95% intervals.

With only four prompt pairs per attribute, these intervals are descriptive and
may underrepresent generalization uncertainty. Do not report formal significance
across the many attribute/guidance comparisons. Intervals are omitted for the
synthetic calibration and for fewer than two usable prompt pairs. A future
guidance-comparison test should resample the same prompt clusters jointly across
settings; this release reports setting-specific intervals only.

## Human listening

The included gallery is labeled and suitable for debugging, not a blinded study.
The CSV template is blank. For a research evaluation, recruit multiple consenting
raters, randomize presentation independently of condition, conceal model/guidance
identity, and maintain a separate allocation key. Assess:

1. Is the requested source actually audible? (0/1)
2. Is the requested attribute represented? (1–5)
3. How acceptable is the sound quality? (1–5)
4. For paired listening, does the high condition exceed the low condition while
   retaining the intended source? Record ambiguous cases.

The script does not aggregate manual ratings or manufacture judgments. Record
listening hardware and playback settings; disable automatic volume normalization.
Digital amplitude results are not calibrated sound-pressure measurements.

## Confounds and boundaries

Quiet/loud descriptors can alter source identity or distance; output normalization
inside a model can remove amplitude cues. F0 estimation can fail independently of
generation quality. Rate estimation does not prove source/event recognition.
Guidance affects multiple sound properties simultaneously. A single model, small
prompt suite and three seeds cannot establish robust general performance.

The shared seed controls initial randomness only; this is not audio editing,
causal disentanglement, or proof that all other acoustic attributes are preserved.
No CLAP similarity score is currently implemented: an embedding similarity score
alone would also be insufficient for these fine-grained comparisons.

## Reproducibility and provenance

The adapter resolves a Hugging Face model branch/tag to a commit SHA before
loading. Resumed runs reuse that SHA. The manifest includes the config, backend,
package versions, Python version, code commit when available, and audio checksums.
GPU runs record device name. Exact bitwise reproduction across hardware/software
is not guaranteed. Initial dependency constraints are not a fully validated lockfile.

Generation writes audio and progress atomically and refuses configuration reuse
or tampered completed files. Evaluation requires complete paired coverage and
verifies hashes. Do not run concurrent writers into the same run directory.
