"""Waveform, spectrogram, and mode-timeline figures."""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram as spg

from backend.config import SAMPLE_RATE
from backend.processor import FrameLog

NAVY = "#0B1C2C"
GOLD = "#C5A46E"
CYAN = "#3EC6C9"
OFF = "#E8EEF2"

MODE_COLORS = {
    "SPEECH_PRESERVATION": "#3EC6C9",
    "STATIONARY": "#7CB87C",
    "DYNAMIC": "#E0B44A",
    "IMPULSE_PROTECTION": "#D45D5D",
    "RECOVERY": "#C5A46E",
}


def _style(fig, ax):
    fig.patch.set_facecolor(NAVY)
    if not isinstance(ax, (list, np.ndarray)):
        ax = [ax]
    else:
        ax = np.ravel(ax)
    for a in ax:
        a.set_facecolor("#122536")
        a.tick_params(colors=OFF)
        a.xaxis.label.set_color(OFF)
        a.yaxis.label.set_color(OFF)
        a.title.set_color(OFF)
        for spine in a.spines.values():
            spine.set_color("#2A4158")
        a.grid(True, color="#1E3348", linestyle="--", linewidth=0.5)


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def plot_waveforms(noisy: np.ndarray, enhanced: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    t = np.arange(len(noisy)) / sr
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True)
    _style(fig, ax)
    ax[0].plot(t, noisy, color=GOLD, lw=0.6)
    ax[0].set_title("BEFORE (input)")
    ax[0].set_ylabel("amp")
    ax[1].plot(t, enhanced, color=CYAN, lw=0.6)
    ax[1].set_title("AFTER (AEGIS-COM)")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("amp")
    fig.tight_layout()
    return fig_to_png(fig)


def plot_spectrograms(noisy: np.ndarray, enhanced: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
    _style(fig, ax)
    for a, x, title in (
        (ax[0], noisy, "BEFORE spectrogram"),
        (ax[1], enhanced, "AFTER spectrogram"),
    ):
        f, tt, sxx = spg(x, fs=sr, nperseg=512, noverlap=384, window="hann")
        db = 10 * np.log10(sxx + 1e-12)
        im = a.pcolormesh(tt, f, db, shading="gouraud", cmap="magma", vmin=-90, vmax=-20)
        a.set_ylabel("Hz")
        a.set_title(title)
        a.set_ylim(0, 8000)
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    return fig_to_png(fig)


def plot_timeline(logs: list[FrameLog], impulse_time_s: float | None = None) -> bytes:
    if not logs:
        fig, ax = plt.subplots(figsize=(11, 2.4))
        _style(fig, ax)
        return fig_to_png(fig)
    t = np.array([fr.t for fr in logs])
    modes = [fr.mode for fr in logs]
    fig, ax = plt.subplots(3, 1, figsize=(11, 6.2), sharex=True)
    _style(fig, ax)
    # mode bands
    ymap = {m: i for i, m in enumerate(MODE_COLORS)}
    for i in range(len(t) - 1):
        ax[0].axvspan(t[i], t[i + 1], color=MODE_COLORS.get(modes[i], "#888"), alpha=0.85)
    if impulse_time_s is not None:
        ax[0].axvline(impulse_time_s, color="white", ls="--", lw=1.0, label="impulse (scripted)")
    ax[0].set_yticks([])
    ax[0].set_title("ACTIVE PROCESSING MODE")
    ax[0].set_xlim(t[0], t[-1])
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=m.replace("_", " "))
        for m, c in MODE_COLORS.items()
    ]
    ax[0].legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.3, labelcolor=OFF)

    ax[1].plot(t, [fr.impulse_prob for fr in logs], color="#D45D5D", label="impulse P")
    ax[1].plot(t, [fr.speech_prob for fr in logs], color=CYAN, label="speech P")
    ax[1].plot(t, [fr.severity for fr in logs], color=GOLD, label="severity")
    ax[1].set_ylim(0, 1.05)
    ax[1].legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor=OFF)
    ax[1].set_title("Acoustic intelligence")

    ax[2].plot(t, [fr.snr_est_db for fr in logs], color=OFF, lw=0.9)
    ax[2].set_title("Estimated frame SNR (dB) — heuristic, not the SIH target")
    ax[2].set_xlabel("time (s)")
    fig.tight_layout()
    return fig_to_png(fig)
