"""Transparent acoustic proxies; these are NOT perceptual quality scores."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def mono_float(audio: np.ndarray) -> np.ndarray:
    """Convert PCM safely before channel averaging (supports unsigned 8-bit WAV)."""
    x = np.asarray(audio)
    if x.ndim not in (1, 2) or not x.size:
        raise ValueError("Expected a nonempty mono or sample-by-channel waveform")
    if x.dtype.kind == "u":
        midpoint = (np.iinfo(x.dtype).max + 1) / 2
        x = (x.astype(np.float64) - midpoint) / midpoint
    elif x.dtype.kind == "i":
        x = x.astype(np.float64) / (abs(np.iinfo(x.dtype).min))
    else:
        x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if not np.isfinite(x).all():
        raise ValueError("Waveform contains NaN or infinity")
    return x


def measure(audio: np.ndarray, sample_rate: int) -> dict:
    """Measure raw loudness, tonal F0 and separated event-rate proxies.

    F0 is estimated by autocorrelation on 64 ms frames and only reported when
    >=25% of energetic frames have a periodicity score >=0.6. The supported
    range is 80-1200 Hz. Event rate counts peaks in a 10 ms RMS envelope with
    at least 150 ms spacing. Both require listening-based validation.
    """
    if sample_rate < 4000:
        raise ValueError("Sample rate must be at least 4000 Hz")
    x = mono_float(audio)
    duration = len(x) / sample_rate
    if duration < 0.25:
        raise ValueError("At least 0.25 seconds of audio is required")
    rms = float(np.sqrt(np.mean(x*x)))
    result = {"duration_s": duration, "rms_dbfs": 20*float(np.log10(max(rms, 1e-12))),
              "peak": float(np.max(np.abs(x))),
              "clipping_fraction": float(np.mean(np.abs(x) >= 0.999)),
              "silent": rms < 1e-5}
    n = max(1, round(0.01 * sample_rate))
    framed = x[:len(x)//n*n].reshape(-1, n)
    envelope = np.sqrt(np.mean(framed**2, axis=1))
    peaks, _ = find_peaks(envelope, distance=15,
                         prominence=max(float(envelope.max())*0.15, 1e-5))
    result["event_rate_hz"] = len(peaks)/duration
    result["event_count"] = len(peaks)
    size = round(0.064*sample_rate)
    hop = size//2
    lo, hi = max(1, int(sample_rate/1200)), int(sample_rate/80)
    pitches = []
    active = 0
    for start in range(0, len(x)-size+1, hop):
        frame = x[start:start+size]
        if np.sqrt(np.mean(frame**2)) < max(rms*0.2, 1e-5):
            continue
        active += 1
        frame = frame - frame.mean()
        ac = np.correlate(frame, frame, mode="full")[size-1:]
        if ac[0] <= 1e-15:
            continue
        candidates, _ = find_peaks(ac[lo:hi+1])
        if not len(candidates):
            continue
        candidates = candidates + lo
        lag = int(candidates[np.argmax(ac[candidates])])
        if ac[lag]/ac[0] >= 0.6:
            # Parabolic interpolation improves pure-tone calibration.
            left, mid, right = ac[lag-1:lag+2]
            denominator = left - 2*mid + right
            offset = 0.5*(left-right)/denominator if denominator else 0.0
            pitches.append(sample_rate/(lag+float(offset)))
    fraction = len(pitches)/max(active, 1)
    result["voiced_fraction"] = fraction
    result["f0_hz"] = float(np.median(pitches)) if pitches and fraction >= 0.25 else None
    return result
