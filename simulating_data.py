"""
NMR Peak Picking Pipeline — CNN vs Vision Transformer (fully 2-D)
==================================================================

Data format throughout the entire pipeline
-------------------------------------------
  Raw spectrum   ndarray  (H, W)       float32   H = ¹H points, W = ¹⁵N points
  Torch input    Tensor   (B, 1, H, W) float32   channel dim added in NMRDataset
  Label heatmap  ndarray  (H, W)       float32   Gaussian blob at each true peak
  Torch label    Tensor   (B, 1, H, W) float32

Simulated data is physically motivated HSQC-style spectra (see Section 1).

Usage
-----
  python nmr_pipeline.py                    # full run
  python nmr_pipeline.py --demo             # quick 2-min sanity check
  python nmr_pipeline.py --compare          # CNN vs ViT head-to-head
  python nmr_pipeline.py --experiment noise
  python nmr_pipeline.py --experiment overlap
  python nmr_pipeline.py --experiment datasize
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")    # swap to "TkAgg" for interactive windows
import matplotlib.pyplot as plt
from scipy.ndimage import label as ndlabel, maximum_filter, gaussian_filter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from einops import rearrange
    from einops.layers.torch import Rearrange
    HAS_EINOPS = True
except ImportError:
    HAS_EINOPS = False
    print("Warning: einops not installed — ViT unavailable.  pip install einops")


# ================================================================
# SECTION 1 — PHYSICALLY MOTIVATED 2-D NMR DATA GENERATOR
# ================================================================
#
# Background for students
# -----------------------
# A 2-D ¹H-¹⁵N HSQC experiment records one cross-peak per backbone
# amide NH in a protein.  The spectrum is digitised as a rectangular
# matrix whose axes are:
#
#   F2 (direct, horizontal)  — ¹H  chemical shift  ~  6–10 ppm
#   F1 (indirect, vertical)  — ¹⁵N chemical shift  ~ 105–135 ppm
#
# Peak shape in REAL NMR
# ----------------------
# • F2 (¹H axis)  : pure Lorentzian  L(x) = 1 / (1 + (x/lw)²)
#                   Lorentzians have wide "feet" unlike Gaussians.
# • F1 (¹⁵N axis) : mixed Lorentzian/Gaussian due to indirect
#                   detection and apodisation; approximated here as
#                   Voigt = α·Lorentzian + (1−α)·Gaussian
#
# Common artefacts in REAL spectra
# ---------------------------------
# • Gaussian white noise  — instrument electronics
# • t1 noise              — vertical ridges at every ¹H chemical shift
# • Baseline roll         — low-frequency sinusoidal baseline
# • Intensity variation   — peaks vary by ~10× depending on T2
# • Linewidth variation   — broader near the diagonal / in disordered regions
# • Overlap              — peaks can merge if chemical shifts are similar
#
# All of these are simulated below.

class HSQCGenerator:
    """
    Generates synthetic ¹H-¹⁵N HSQC-style 2-D NMR spectra.

    Parameters
    ----------
    n_points_h  : int    — number of digital points on the ¹H  (F2) axis
    n_points_n  : int    — number of digital points on the ¹⁵N (F1) axis
    sw_h_ppm    : float  — spectral width in ¹H  ppm (default 4 ppm: 6–10)
    sw_n_ppm    : float  — spectral width in ¹⁵N ppm (default 30 ppm: 105–135)
    num_samples : int    — number of spectra to generate
    """

    def __init__(
        self,
        n_points_h  = 256,   # ¹H  axis (rows)    matches ViT height
        n_points_n  = 32,    # ¹⁵N axis (cols)    matches ViT width
        sw_h_ppm    = 4.0,   # ¹H  spectral width  (ppm)
        sw_n_ppm    = 30.0,  # ¹⁵N spectral width  (ppm)
        num_samples = 500,
    ):
        self.H          = n_points_h
        self.W          = n_points_n
        self.sw_h       = sw_h_ppm
        self.sw_n       = sw_n_ppm
        self.num_samples = num_samples

        # ppm axes (for labelling; row 0 = 6 ppm, row H-1 = 10 ppm)
        self.ppm_h = np.linspace(6.0,  10.0,  self.H)   # ¹H  ppm values per row
        self.ppm_n = np.linspace(105.0, 135.0, self.W)   # ¹⁵N ppm values per col

        # Points-per-ppm conversion
        self.pts_per_ppm_h = self.H / self.sw_h   # ~64 pts/ppm for H=256, sw=4
        self.pts_per_ppm_n = self.W / self.sw_n   # ~ 1 pt/ppm  for W=32,  sw=30

        # Row/col coordinate grids (reused for every peak)
        self._rows = np.arange(self.H, dtype=np.float32)
        self._cols = np.arange(self.W, dtype=np.float32)
        self._RR, self._CC = np.meshgrid(self._rows, self._cols, indexing="ij")

    # ----------------------------------------------------------------
    # Low-level lineshape primitives
    # ----------------------------------------------------------------

    def _lorentzian_1d(self, centers_pts, lw_pts, axis_len):
        """
        Vectorised 1-D Lorentzian for multiple peaks on one axis.

        L(x; c, lw) = 1 / (1 + ((x − c) / (lw/2))²)

        Parameters
        ----------
        centers_pts : array (n_peaks,) — peak centres in points
        lw_pts      : array (n_peaks,) — full linewidth at half maximum (FWHM) in points
        axis_len    : int

        Returns
        -------
        profiles : array (n_peaks, axis_len)
        """
        x   = np.arange(axis_len, dtype=np.float64)             # (axis_len,)
        hwhm = (lw_pts / 2.0)[:, None]                           # (n, 1)
        c    = centers_pts[:, None]                               # (n, 1)
        return (1.0 / (1.0 + ((x - c) / hwhm) ** 2)).astype(np.float32)

    def _voigt_1d(self, centers_pts, lw_pts, axis_len, lor_fraction=0.6):
        """
        Pseudo-Voigt for the ¹⁵N (F1) axis.

        V = lor_fraction · Lorentzian + (1 − lor_fraction) · Gaussian
        The Gaussian uses the same FWHM as the Lorentzian.

        Parameters
        ----------
        lor_fraction : float in [0, 1] — how Lorentzian the F1 lineshape is
        """
        x    = np.arange(axis_len, dtype=np.float64)
        hwhm = (lw_pts / 2.0)[:, None]
        c    = centers_pts[:, None]
        lor  = 1.0 / (1.0 + ((x - c) / hwhm) ** 2)
        sig  = hwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # σ from FWHM
        gau  = np.exp(-0.5 * ((x - c) / sig) ** 2)
        return (lor_fraction * lor + (1.0 - lor_fraction) * gau).astype(np.float32)

    # ----------------------------------------------------------------
    # Build one 2-D peak as an outer product of 1-D lineshapes
    # ----------------------------------------------------------------

    def _peak_2d(self, cy, cx, amp, lw_h, lw_n, lor_fraction=0.6):
        """
        Single 2-D HSQC cross-peak = F2_lineshape ⊗ F1_lineshape.

        Parameters
        ----------
        cy, cx      : float  — centre in (row, col) points
        amp         : float  — peak amplitude
        lw_h        : float  — ¹H  FWHM in points
        lw_n        : float  — ¹⁵N FWHM in points

        Returns
        -------
        peak2d : ndarray (H, W)
        """
        prof_h = self._lorentzian_1d(
            np.array([cy]), np.array([lw_h]), self.H
        )[0]   # (H,)
        prof_n = self._voigt_1d(
            np.array([cx]), np.array([lw_n]), self.W, lor_fraction
        )[0]   # (W,)
        return amp * np.outer(prof_h, prof_n).astype(np.float32)   # (H, W)

    # ----------------------------------------------------------------
    # Generate one HSQC spectrum with realistic artefacts
    # ----------------------------------------------------------------

    def generate_spectrum(self, num_peaks=None, noise_level=0.02, seed=None):
        """
        Parameters
        ----------
        num_peaks   : int or None — if None, drawn from U[3, 15]
                      (a small protein section has ~3–15 visible peaks)
        noise_level : float — base Gaussian noise σ relative to max peak amplitude
        seed        : int or None — for reproducibility

        Returns
        -------
        spectrum   : ndarray (H, W) float32 — the simulated HSQC spectrum
        positions  : list of (row, col) tuples — true peak centres in points
        ppm_coords : list of (ppm_h, ppm_n) tuples — same peaks in ppm
        """
        rng = np.random.default_rng(seed)

        if num_peaks is None:
            num_peaks = int(rng.integers(3, 16))

        # ── 1. Choose random peak positions in ppm space ────────
        #    ¹H  range: 6.5–9.5 ppm  (avoids spectral edges)
        #    ¹⁵N range: 107–133 ppm
        ppm_h_centres = rng.uniform(6.5,  9.5,  num_peaks)
        ppm_n_centres = rng.uniform(107.0, 133.0, num_peaks)

        # Convert ppm → point indices
        # row 0 = ppm_h[0] = 6 ppm, so:  row = (ppm − 6) / sw_h × H
        cy_pts = (ppm_h_centres - 6.0)  / self.sw_h * self.H
        cx_pts = (ppm_n_centres - 105.0) / self.sw_n * self.W

        # ── 2. Draw peak-specific parameters ────────────────────
        #    Amplitude:  log-normal (realistic ~10× variation in T2)
        amplitudes   = rng.lognormal(mean=1.0, sigma=0.6, size=num_peaks)
        amplitudes  /= amplitudes.max()   # normalise so max = 1

        #    ¹H  linewidth:  3–10 pts  (broader = less resolved)
        lw_h_pts = rng.uniform(3.0, 10.0, num_peaks)

        #    ¹⁵N linewidth:  2–6 pts
        lw_n_pts = rng.uniform(2.0,  6.0, num_peaks)

        #    Lorentzian fraction for F1: varies per peak (0.4–0.8)
        lor_fracs = rng.uniform(0.4, 0.8, num_peaks)

        # ── 3. Sum all 2-D peaks ────────────────────────────────
        spectrum = np.zeros((self.H, self.W), dtype=np.float32)
        positions, ppm_coords = [], []

        for i in range(num_peaks):
            # Skip peaks whose centres fall outside the matrix
            if not (0 <= cy_pts[i] < self.H and 0 <= cx_pts[i] < self.W):
                continue
            peak2d    = self._peak_2d(cy_pts[i], cx_pts[i],
                                       amplitudes[i],
                                       lw_h_pts[i], lw_n_pts[i],
                                       lor_fracs[i])
            spectrum += peak2d
            positions.append((int(round(cy_pts[i])), int(round(cx_pts[i]))))
            ppm_coords.append((float(ppm_h_centres[i]), float(ppm_n_centres[i])))

        # ── 4. Realistic artefacts ───────────────────────────────

        # 4a. Gaussian white noise  (instrument electronics)
        noise = rng.normal(0, noise_level, (self.H, self.W)).astype(np.float32)
        spectrum += noise

        # 4b. t1 noise — vertical ridges at each ¹H chemical shift
        #     (artefact from imperfect t1 increment; appears as faint
        #     vertical stripes constant across the ¹⁵N axis)
        t1_amplitude  = noise_level * rng.uniform(0.5, 2.0)
        t1_profile_h  = rng.normal(0, t1_amplitude, self.H).astype(np.float32)
        # Smooth slightly along ¹H so they look like real ridge artefacts
        t1_profile_h  = gaussian_filter(t1_profile_h, sigma=1.5).astype(np.float32)
        t1_noise      = np.outer(t1_profile_h, np.ones(self.W, dtype=np.float32))
        spectrum     += t1_noise

        # 4c. Baseline roll — slow sinusoidal drift across ¹H axis
        #     (from DC offset and apodisation mismatch)
        n_waves      = rng.integers(1, 4)
        baseline_h   = np.zeros(self.H, dtype=np.float32)
        for _ in range(n_waves):
            freq  = rng.uniform(0.5, 2.0) / self.H
            phase = rng.uniform(0, 2 * np.pi)
            amp_b = noise_level * rng.uniform(0.3, 1.0)
            baseline_h += (amp_b * np.sin(
                2 * np.pi * freq * np.arange(self.H) + phase
            )).astype(np.float32)
        # Same slow roll on ¹⁵N axis (much smaller)
        baseline_n   = np.zeros(self.W, dtype=np.float32)
        freq_n  = rng.uniform(0.3, 1.0) / self.W
        amp_bn  = noise_level * rng.uniform(0.1, 0.4)
        baseline_n += (amp_bn * np.sin(
            2 * np.pi * freq_n * np.arange(self.W)
        )).astype(np.float32)
        spectrum += np.outer(baseline_h, np.ones(self.W, dtype=np.float32))
        spectrum += np.outer(np.ones(self.H, dtype=np.float32), baseline_n)

        # 4d. Mild phase error — slight asymmetry in peak feet
        #     (from imperfect phase correction; Hilbert approximation)
        phase_err = rng.uniform(-0.05, 0.05)   # small dispersion contribution
        if abs(phase_err) > 0.01:
            # Approximate dispersive component: derivative of spectrum rows
            dispersive = np.gradient(spectrum, axis=1).astype(np.float32)
            spectrum  += phase_err * dispersive

        # Keep spectrum non-negative (NMR magnitude spectra are positive)
        spectrum = np.clip(spectrum, 0, None)

        return spectrum, positions, ppm_coords

    # ----------------------------------------------------------------
    # Generate a full dataset
    # ----------------------------------------------------------------

    def generate_dataset(self, noise_level=0.02, num_peaks_range=(3, 12),
                         save_path=None):
        """
        Generate num_samples HSQC spectra and optionally save them.

        Parameters
        ----------
        noise_level      : float        — Gaussian noise σ
        num_peaks_range  : (int, int)   — uniform range for number of peaks per spectrum
        save_path        : str or None  — if given, save to this .npz file

        Returns
        -------
        spectra    : ndarray (N, H, W)
        peak_lists : list[list[(row, col)]]
        ppm_lists  : list[list[(ppm_h, ppm_n)]]
        """
        spectra, peak_lists, ppm_lists = [], [], []
        for i in range(self.num_samples):
            n = int(np.random.randint(num_peaks_range[0], num_peaks_range[1] + 1))
            sp, pk, ppm = self.generate_spectrum(num_peaks=n,
                                                  noise_level=noise_level,
                                                  seed=i)
            spectra.append(sp)
            peak_lists.append(pk)
            ppm_lists.append(ppm)
        spectra = np.array(spectra)
        if save_path is not None:
            HSQCGenerator.save_dataset(
                save_path, spectra, peak_lists, ppm_lists,
                noise_level=noise_level,
                num_peaks_range=num_peaks_range,
                ppm_h_axis=self.ppm_h,
                ppm_n_axis=self.ppm_n,
            )
        return spectra, peak_lists, ppm_lists

    # ----------------------------------------------------------------
    # Save / load dataset
    # ----------------------------------------------------------------

    @staticmethod
    def save_dataset(path, spectra, peak_lists, ppm_lists,
                     noise_level=None, num_peaks_range=None,
                     ppm_h_axis=None, ppm_n_axis=None):
        """
        Save a generated dataset to a single compressed .npz file.

        Everything needed to reproduce plots and train models is stored:

          spectra        — (N, H, W) float32 spectrum images
          peaks_row      — (N, max_peaks) int16, peak row indices;
                           padded with -1 where a spectrum has fewer peaks
          peaks_col      — (N, max_peaks) int16, peak col indices;
                           padded with -1 where a spectrum has fewer peaks
          peaks_count    — (N,) int16, how many real peaks each spectrum has
          ppm_h_peaks    — (N, max_peaks) float32, ¹H  ppm of each peak
          ppm_n_peaks    — (N, max_peaks) float32, ¹⁵N ppm of each peak
          ppm_h_axis     — (H,) float32, ¹H  ppm value for every row
          ppm_n_axis     — (W,) float32, ¹⁵N ppm value for every col

        Metadata stored as scalar arrays (readable without pickling):
          noise_level, peaks_min, peaks_max, n_points_h, n_points_w

        Parameters
        ----------
        path           : str  — file path; .npz extension added if missing
        spectra        : ndarray (N, H, W)
        peak_lists     : list[list[(row, col)]]
        ppm_lists      : list[list[(ppm_h, ppm_n)]]
        noise_level    : float or None
        num_peaks_range: (int, int) or None
        ppm_h_axis     : ndarray (H,) or None  — ¹H  axis in ppm
        ppm_n_axis     : ndarray (W,) or None  — ¹⁵N axis in ppm

        Returns
        -------
        path : str — the path the file was actually written to
        """
        if not path.endswith(".npz"):
            path = path + ".npz"

        N, H, W = spectra.shape
        max_pk   = max((len(pk) for pk in peak_lists), default=0)

        # Build padded peak arrays (pad value = -1 so 0 is a valid index)
        peaks_row   = np.full((N, max_pk), -1, dtype=np.int16)
        peaks_col   = np.full((N, max_pk), -1, dtype=np.int16)
        peaks_count = np.zeros(N, dtype=np.int16)
        ppm_h_pk    = np.full((N, max_pk), np.nan, dtype=np.float32)
        ppm_n_pk    = np.full((N, max_pk), np.nan, dtype=np.float32)

        for i, (pk_list, ppm_list) in enumerate(zip(peak_lists, ppm_lists)):
            n_pk = len(pk_list)
            peaks_count[i] = n_pk
            for j, ((ry, rx), (ph, pn)) in enumerate(zip(pk_list, ppm_list)):
                peaks_row[i, j] = ry
                peaks_col[i, j] = rx
                ppm_h_pk[i, j]  = ph
                ppm_n_pk[i, j]  = pn

        # Default ppm axes if not supplied
        if ppm_h_axis is None:
            ppm_h_axis = np.linspace(6.0, 10.0, H, dtype=np.float32)
        if ppm_n_axis is None:
            ppm_n_axis = np.linspace(105.0, 135.0, W, dtype=np.float32)

        np.savez_compressed(
            path,
            # Core data
            spectra      = spectra.astype(np.float32),
            peaks_row    = peaks_row,
            peaks_col    = peaks_col,
            peaks_count  = peaks_count,
            ppm_h_peaks  = ppm_h_pk,
            ppm_n_peaks  = ppm_n_pk,
            ppm_h_axis   = ppm_h_axis.astype(np.float32),
            ppm_n_axis   = ppm_n_axis.astype(np.float32),
            # Metadata (stored as 0-d arrays)
            noise_level  = np.float32(noise_level  if noise_level  is not None else np.nan),
            peaks_min    = np.int16(num_peaks_range[0] if num_peaks_range else -1),
            peaks_max    = np.int16(num_peaks_range[1] if num_peaks_range else -1),
            n_points_h   = np.int16(H),
            n_points_w   = np.int16(W),
        )
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  Dataset saved → {path}  ({N} spectra, {size_mb:.1f} MB)")
        return path

    @staticmethod
    def load_dataset(path):
        """
        Load a dataset previously saved with save_dataset().

        Parameters
        ----------
        path : str — path to the .npz file (with or without extension)

        Returns
        -------
        spectra    : ndarray (N, H, W) float32
        peak_lists : list[list[(row, col)]]      — integer point coordinates
        ppm_lists  : list[list[(ppm_h, ppm_n)]] — ppm coordinates
        meta       : dict with keys:
                       ppm_h_axis   — ndarray (H,)
                       ppm_n_axis   — ndarray (W,)
                       noise_level  — float
                       peaks_min    — int
                       peaks_max    — int
                       n_points_h   — int
                       n_points_w   — int

        Example
        -------
        spectra, peak_lists, ppm_lists, meta = HSQCGenerator.load_dataset("hsqc_train.npz")
        ppm_h_axis = meta["ppm_h_axis"]   # for axis labels in plots
        """
        if not path.endswith(".npz"):
            path = path + ".npz"

        data = np.load(path, allow_pickle=False)

        spectra     = data["spectra"]          # (N, H, W)
        peaks_row   = data["peaks_row"]        # (N, max_pk)
        peaks_col   = data["peaks_col"]
        peaks_count = data["peaks_count"]      # (N,)
        ppm_h_pk    = data["ppm_h_peaks"]
        ppm_n_pk    = data["ppm_n_peaks"]

        N = len(spectra)
        peak_lists, ppm_lists = [], []
        for i in range(N):
            n_pk = int(peaks_count[i])
            pk   = [(int(peaks_row[i, j]), int(peaks_col[i, j]))
                    for j in range(n_pk)]
            ppm  = [(float(ppm_h_pk[i, j]), float(ppm_n_pk[i, j]))
                    for j in range(n_pk)]
            peak_lists.append(pk)
            ppm_lists.append(ppm)

        meta = {
            "ppm_h_axis"  : data["ppm_h_axis"],
            "ppm_n_axis"  : data["ppm_n_axis"],
            "noise_level" : float(data["noise_level"]),
            "peaks_min"   : int(data["peaks_min"]),
            "peaks_max"   : int(data["peaks_max"]),
            "n_points_h"  : int(data["n_points_h"]),
            "n_points_w"  : int(data["n_points_w"]),
        }

        print(f"  Dataset loaded ← {path}  "
              f"({N} spectra, {spectra.shape[1]}×{spectra.shape[2]} pts, "
              f"noise σ={meta['noise_level']:.3f})")
        return spectra, peak_lists, ppm_lists, meta
