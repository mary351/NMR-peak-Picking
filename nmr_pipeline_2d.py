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

from classical_picker_model_2d import ClassicalPeakPicker2D
from cnn_picker_model import CNN_PeakDetector2D
from vit_picker_model import ViT_PP
from simulating_data import HSQCGenerator 
from load_real_nmr import RealDataLoader, load_real_dataset

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

# class HSQCGenerator:
#     """
#     Generates synthetic ¹H-¹⁵N HSQC-style 2-D NMR spectra.

#     Parameters
#     ----------
#     n_points_h  : int    — number of digital points on the ¹H  (F2) axis
#     n_points_n  : int    — number of digital points on the ¹⁵N (F1) axis
#     sw_h_ppm    : float  — spectral width in ¹H  ppm (default 4 ppm: 6–10)
#     sw_n_ppm    : float  — spectral width in ¹⁵N ppm (default 30 ppm: 105–135)
#     num_samples : int    — number of spectra to generate
#     """

#     def __init__(
#         self,
#         n_points_h  = 256,   # ¹H  axis (rows)    matches ViT height
#         n_points_n  = 32,    # ¹⁵N axis (cols)    matches ViT width
#         sw_h_ppm    = 4.0,   # ¹H  spectral width  (ppm)
#         sw_n_ppm    = 30.0,  # ¹⁵N spectral width  (ppm)
#         num_samples = 500,
#     ):
#         self.H          = n_points_h
#         self.W          = n_points_n
#         self.sw_h       = sw_h_ppm
#         self.sw_n       = sw_n_ppm
#         self.num_samples = num_samples

#         # ppm axes (for labelling; row 0 = 6 ppm, row H-1 = 10 ppm)
#         self.ppm_h = np.linspace(6.0,  10.0,  self.H)   # ¹H  ppm values per row
#         self.ppm_n = np.linspace(105.0, 135.0, self.W)   # ¹⁵N ppm values per col

#         # Points-per-ppm conversion
#         self.pts_per_ppm_h = self.H / self.sw_h   # ~64 pts/ppm for H=256, sw=4
#         self.pts_per_ppm_n = self.W / self.sw_n   # ~ 1 pt/ppm  for W=32,  sw=30

#         # Row/col coordinate grids (reused for every peak)
#         self._rows = np.arange(self.H, dtype=np.float32)
#         self._cols = np.arange(self.W, dtype=np.float32)
#         self._RR, self._CC = np.meshgrid(self._rows, self._cols, indexing="ij")

#     # ----------------------------------------------------------------
#     # Low-level lineshape primitives
#     # ----------------------------------------------------------------

#     def _lorentzian_1d(self, centers_pts, lw_pts, axis_len):
#         """
#         Vectorised 1-D Lorentzian for multiple peaks on one axis.

#         L(x; c, lw) = 1 / (1 + ((x − c) / (lw/2))²)

#         Parameters
#         ----------
#         centers_pts : array (n_peaks,) — peak centres in points
#         lw_pts      : array (n_peaks,) — full linewidth at half maximum (FWHM) in points
#         axis_len    : int

#         Returns
#         -------
#         profiles : array (n_peaks, axis_len)
#         """
#         x   = np.arange(axis_len, dtype=np.float64)             # (axis_len,)
#         hwhm = (lw_pts / 2.0)[:, None]                           # (n, 1)
#         c    = centers_pts[:, None]                               # (n, 1)
#         return (1.0 / (1.0 + ((x - c) / hwhm) ** 2)).astype(np.float32)

#     def _voigt_1d(self, centers_pts, lw_pts, axis_len, lor_fraction=0.6):
#         """
#         Pseudo-Voigt for the ¹⁵N (F1) axis.

#         V = lor_fraction · Lorentzian + (1 − lor_fraction) · Gaussian
#         The Gaussian uses the same FWHM as the Lorentzian.

#         Parameters
#         ----------
#         lor_fraction : float in [0, 1] — how Lorentzian the F1 lineshape is
#         """
#         x    = np.arange(axis_len, dtype=np.float64)
#         hwhm = (lw_pts / 2.0)[:, None]
#         c    = centers_pts[:, None]
#         lor  = 1.0 / (1.0 + ((x - c) / hwhm) ** 2)
#         sig  = hwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # σ from FWHM
#         gau  = np.exp(-0.5 * ((x - c) / sig) ** 2)
#         return (lor_fraction * lor + (1.0 - lor_fraction) * gau).astype(np.float32)

#     # ----------------------------------------------------------------
#     # Build one 2-D peak as an outer product of 1-D lineshapes
#     # ----------------------------------------------------------------

#     def _peak_2d(self, cy, cx, amp, lw_h, lw_n, lor_fraction=0.6):
#         """
#         Single 2-D HSQC cross-peak = F2_lineshape ⊗ F1_lineshape.

#         Parameters
#         ----------
#         cy, cx      : float  — centre in (row, col) points
#         amp         : float  — peak amplitude
#         lw_h        : float  — ¹H  FWHM in points
#         lw_n        : float  — ¹⁵N FWHM in points

#         Returns
#         -------
#         peak2d : ndarray (H, W)
#         """
#         prof_h = self._lorentzian_1d(
#             np.array([cy]), np.array([lw_h]), self.H
#         )[0]   # (H,)
#         prof_n = self._voigt_1d(
#             np.array([cx]), np.array([lw_n]), self.W, lor_fraction
#         )[0]   # (W,)
#         return amp * np.outer(prof_h, prof_n).astype(np.float32)   # (H, W)

#     # ----------------------------------------------------------------
#     # Generate one HSQC spectrum with realistic artefacts
#     # ----------------------------------------------------------------

#     def generate_spectrum(self, num_peaks=None, noise_level=0.02, seed=None):
#         """
#         Parameters
#         ----------
#         num_peaks   : int or None — if None, drawn from U[3, 15]
#                       (a small protein section has ~3–15 visible peaks)
#         noise_level : float — base Gaussian noise σ relative to max peak amplitude
#         seed        : int or None — for reproducibility

#         Returns
#         -------
#         spectrum   : ndarray (H, W) float32 — the simulated HSQC spectrum
#         positions  : list of (row, col) tuples — true peak centres in points
#         ppm_coords : list of (ppm_h, ppm_n) tuples — same peaks in ppm
#         """
#         rng = np.random.default_rng(seed)

#         if num_peaks is None:
#             num_peaks = int(rng.integers(3, 16))

#         # ── 1. Choose random peak positions in ppm space ────────
#         #    ¹H  range: 6.5–9.5 ppm  (avoids spectral edges)
#         #    ¹⁵N range: 107–133 ppm
#         ppm_h_centres = rng.uniform(6.5,  9.5,  num_peaks)
#         ppm_n_centres = rng.uniform(107.0, 133.0, num_peaks)

#         # Convert ppm → point indices
#         # row 0 = ppm_h[0] = 6 ppm, so:  row = (ppm − 6) / sw_h × H
#         cy_pts = (ppm_h_centres - 6.0)  / self.sw_h * self.H
#         cx_pts = (ppm_n_centres - 105.0) / self.sw_n * self.W

#         # ── 2. Draw peak-specific parameters ────────────────────
#         #    Amplitude:  log-normal (realistic ~10× variation in T2)
#         amplitudes   = rng.lognormal(mean=1.0, sigma=0.6, size=num_peaks)
#         amplitudes  /= amplitudes.max()   # normalise so max = 1

#         #    ¹H  linewidth:  3–10 pts  (broader = less resolved)
#         lw_h_pts = rng.uniform(3.0, 10.0, num_peaks)

#         #    ¹⁵N linewidth:  2–6 pts
#         lw_n_pts = rng.uniform(2.0,  6.0, num_peaks)

#         #    Lorentzian fraction for F1: varies per peak (0.4–0.8)
#         lor_fracs = rng.uniform(0.4, 0.8, num_peaks)

#         # ── 3. Sum all 2-D peaks ────────────────────────────────
#         spectrum = np.zeros((self.H, self.W), dtype=np.float32)
#         positions, ppm_coords = [], []

#         for i in range(num_peaks):
#             # Skip peaks whose centres fall outside the matrix
#             if not (0 <= cy_pts[i] < self.H and 0 <= cx_pts[i] < self.W):
#                 continue
#             peak2d    = self._peak_2d(cy_pts[i], cx_pts[i],
#                                        amplitudes[i],
#                                        lw_h_pts[i], lw_n_pts[i],
#                                        lor_fracs[i])
#             spectrum += peak2d
#             positions.append((int(round(cy_pts[i])), int(round(cx_pts[i]))))
#             ppm_coords.append((float(ppm_h_centres[i]), float(ppm_n_centres[i])))

#         # ── 4. Realistic artefacts ───────────────────────────────

#         # 4a. Gaussian white noise  (instrument electronics)
#         noise = rng.normal(0, noise_level, (self.H, self.W)).astype(np.float32)
#         spectrum += noise

#         # 4b. t1 noise — vertical ridges at each ¹H chemical shift
#         #     (artefact from imperfect t1 increment; appears as faint
#         #     vertical stripes constant across the ¹⁵N axis)
#         t1_amplitude  = noise_level * rng.uniform(0.5, 2.0)
#         t1_profile_h  = rng.normal(0, t1_amplitude, self.H).astype(np.float32)
#         # Smooth slightly along ¹H so they look like real ridge artefacts
#         t1_profile_h  = gaussian_filter(t1_profile_h, sigma=1.5).astype(np.float32)
#         t1_noise      = np.outer(t1_profile_h, np.ones(self.W, dtype=np.float32))
#         spectrum     += t1_noise

#         # 4c. Baseline roll — slow sinusoidal drift across ¹H axis
#         #     (from DC offset and apodisation mismatch)
#         n_waves      = rng.integers(1, 4)
#         baseline_h   = np.zeros(self.H, dtype=np.float32)
#         for _ in range(n_waves):
#             freq  = rng.uniform(0.5, 2.0) / self.H
#             phase = rng.uniform(0, 2 * np.pi)
#             amp_b = noise_level * rng.uniform(0.3, 1.0)
#             baseline_h += (amp_b * np.sin(
#                 2 * np.pi * freq * np.arange(self.H) + phase
#             )).astype(np.float32)
#         # Same slow roll on ¹⁵N axis (much smaller)
#         baseline_n   = np.zeros(self.W, dtype=np.float32)
#         freq_n  = rng.uniform(0.3, 1.0) / self.W
#         amp_bn  = noise_level * rng.uniform(0.1, 0.4)
#         baseline_n += (amp_bn * np.sin(
#             2 * np.pi * freq_n * np.arange(self.W)
#         )).astype(np.float32)
#         spectrum += np.outer(baseline_h, np.ones(self.W, dtype=np.float32))
#         spectrum += np.outer(np.ones(self.H, dtype=np.float32), baseline_n)

#         # 4d. Mild phase error — slight asymmetry in peak feet
#         #     (from imperfect phase correction; Hilbert approximation)
#         phase_err = rng.uniform(-0.05, 0.05)   # small dispersion contribution
#         if abs(phase_err) > 0.01:
#             # Approximate dispersive component: derivative of spectrum rows
#             dispersive = np.gradient(spectrum, axis=1).astype(np.float32)
#             spectrum  += phase_err * dispersive

#         # Keep spectrum non-negative (NMR magnitude spectra are positive)
#         spectrum = np.clip(spectrum, 0, None)

#         return spectrum, positions, ppm_coords

#     # ----------------------------------------------------------------
#     # Generate a full dataset
#     # ----------------------------------------------------------------

#     def generate_dataset(self, noise_level=0.02, num_peaks_range=(3, 12),
#                          save_path=None):
#         """
#         Generate num_samples HSQC spectra and optionally save them.

#         Parameters
#         ----------
#         noise_level      : float        — Gaussian noise σ
#         num_peaks_range  : (int, int)   — uniform range for number of peaks per spectrum
#         save_path        : str or None  — if given, save to this .npz file

#         Returns
#         -------
#         spectra    : ndarray (N, H, W)
#         peak_lists : list[list[(row, col)]]
#         ppm_lists  : list[list[(ppm_h, ppm_n)]]
#         """
#         spectra, peak_lists, ppm_lists = [], [], []
#         for i in range(self.num_samples):
#             n = int(np.random.randint(num_peaks_range[0], num_peaks_range[1] + 1))
#             sp, pk, ppm = self.generate_spectrum(num_peaks=n,
#                                                   noise_level=noise_level,
#                                                   seed=i)
#             spectra.append(sp)
#             peak_lists.append(pk)
#             ppm_lists.append(ppm)
#         spectra = np.array(spectra)
#         if save_path is not None:
#             HSQCGenerator.save_dataset(
#                 save_path, spectra, peak_lists, ppm_lists,
#                 noise_level=noise_level,
#                 num_peaks_range=num_peaks_range,
#                 ppm_h_axis=self.ppm_h,
#                 ppm_n_axis=self.ppm_n,
#             )
#         return spectra, peak_lists, ppm_lists

#     # ----------------------------------------------------------------
#     # Save / load dataset
#     # ----------------------------------------------------------------

#     @staticmethod
#     def save_dataset(path, spectra, peak_lists, ppm_lists,
#                      noise_level=None, num_peaks_range=None,
#                      ppm_h_axis=None, ppm_n_axis=None):
#         """
#         Save a generated dataset to a single compressed .npz file.

#         Everything needed to reproduce plots and train models is stored:

#           spectra        — (N, H, W) float32 spectrum images
#           peaks_row      — (N, max_peaks) int16, peak row indices;
#                            padded with -1 where a spectrum has fewer peaks
#           peaks_col      — (N, max_peaks) int16, peak col indices;
#                            padded with -1 where a spectrum has fewer peaks
#           peaks_count    — (N,) int16, how many real peaks each spectrum has
#           ppm_h_peaks    — (N, max_peaks) float32, ¹H  ppm of each peak
#           ppm_n_peaks    — (N, max_peaks) float32, ¹⁵N ppm of each peak
#           ppm_h_axis     — (H,) float32, ¹H  ppm value for every row
#           ppm_n_axis     — (W,) float32, ¹⁵N ppm value for every col

#         Metadata stored as scalar arrays (readable without pickling):
#           noise_level, peaks_min, peaks_max, n_points_h, n_points_w

#         Parameters
#         ----------
#         path           : str  — file path; .npz extension added if missing
#         spectra        : ndarray (N, H, W)
#         peak_lists     : list[list[(row, col)]]
#         ppm_lists      : list[list[(ppm_h, ppm_n)]]
#         noise_level    : float or None
#         num_peaks_range: (int, int) or None
#         ppm_h_axis     : ndarray (H,) or None  — ¹H  axis in ppm
#         ppm_n_axis     : ndarray (W,) or None  — ¹⁵N axis in ppm

#         Returns
#         -------
#         path : str — the path the file was actually written to
#         """
#         if not path.endswith(".npz"):
#             path = path + ".npz"

#         N, H, W = spectra.shape
#         max_pk   = max((len(pk) for pk in peak_lists), default=0)

#         # Build padded peak arrays (pad value = -1 so 0 is a valid index)
#         peaks_row   = np.full((N, max_pk), -1, dtype=np.int16)
#         peaks_col   = np.full((N, max_pk), -1, dtype=np.int16)
#         peaks_count = np.zeros(N, dtype=np.int16)
#         ppm_h_pk    = np.full((N, max_pk), np.nan, dtype=np.float32)
#         ppm_n_pk    = np.full((N, max_pk), np.nan, dtype=np.float32)

#         for i, (pk_list, ppm_list) in enumerate(zip(peak_lists, ppm_lists)):
#             n_pk = len(pk_list)
#             peaks_count[i] = n_pk
#             for j, ((ry, rx), (ph, pn)) in enumerate(zip(pk_list, ppm_list)):
#                 peaks_row[i, j] = ry
#                 peaks_col[i, j] = rx
#                 ppm_h_pk[i, j]  = ph
#                 ppm_n_pk[i, j]  = pn

#         # Default ppm axes if not supplied
#         if ppm_h_axis is None:
#             ppm_h_axis = np.linspace(6.0, 10.0, H, dtype=np.float32)
#         if ppm_n_axis is None:
#             ppm_n_axis = np.linspace(105.0, 135.0, W, dtype=np.float32)

#         np.savez_compressed(
#             path,
#             # Core data
#             spectra      = spectra.astype(np.float32),
#             peaks_row    = peaks_row,
#             peaks_col    = peaks_col,
#             peaks_count  = peaks_count,
#             ppm_h_peaks  = ppm_h_pk,
#             ppm_n_peaks  = ppm_n_pk,
#             ppm_h_axis   = ppm_h_axis.astype(np.float32),
#             ppm_n_axis   = ppm_n_axis.astype(np.float32),
#             # Metadata (stored as 0-d arrays)
#             noise_level  = np.float32(noise_level  if noise_level  is not None else np.nan),
#             peaks_min    = np.int16(num_peaks_range[0] if num_peaks_range else -1),
#             peaks_max    = np.int16(num_peaks_range[1] if num_peaks_range else -1),
#             n_points_h   = np.int16(H),
#             n_points_w   = np.int16(W),
#         )
#         size_mb = os.path.getsize(path) / 1024 / 1024
#         print(f"  Dataset saved → {path}  ({N} spectra, {size_mb:.1f} MB)")
#         return path

#     @staticmethod
#     def load_dataset(path):
#         """
#         Load a dataset previously saved with save_dataset().

#         Parameters
#         ----------
#         path : str — path to the .npz file (with or without extension)

#         Returns
#         -------
#         spectra    : ndarray (N, H, W) float32
#         peak_lists : list[list[(row, col)]]      — integer point coordinates
#         ppm_lists  : list[list[(ppm_h, ppm_n)]] — ppm coordinates
#         meta       : dict with keys:
#                        ppm_h_axis   — ndarray (H,)
#                        ppm_n_axis   — ndarray (W,)
#                        noise_level  — float
#                        peaks_min    — int
#                        peaks_max    — int
#                        n_points_h   — int
#                        n_points_w   — int

#         Example
#         -------
#         spectra, peak_lists, ppm_lists, meta = HSQCGenerator.load_dataset("hsqc_train.npz")
#         ppm_h_axis = meta["ppm_h_axis"]   # for axis labels in plots
#         """
#         if not path.endswith(".npz"):
#             path = path + ".npz"

#         data = np.load(path, allow_pickle=False)

#         spectra     = data["spectra"]          # (N, H, W)
#         peaks_row   = data["peaks_row"]        # (N, max_pk)
#         peaks_col   = data["peaks_col"]
#         peaks_count = data["peaks_count"]      # (N,)
#         ppm_h_pk    = data["ppm_h_peaks"]
#         ppm_n_pk    = data["ppm_n_peaks"]

#         N = len(spectra)
#         peak_lists, ppm_lists = [], []
#         for i in range(N):
#             n_pk = int(peaks_count[i])
#             pk   = [(int(peaks_row[i, j]), int(peaks_col[i, j]))
#                     for j in range(n_pk)]
#             ppm  = [(float(ppm_h_pk[i, j]), float(ppm_n_pk[i, j]))
#                     for j in range(n_pk)]
#             peak_lists.append(pk)
#             ppm_lists.append(ppm)

#         meta = {
#             "ppm_h_axis"  : data["ppm_h_axis"],
#             "ppm_n_axis"  : data["ppm_n_axis"],
#             "noise_level" : float(data["noise_level"]),
#             "peaks_min"   : int(data["peaks_min"]),
#             "peaks_max"   : int(data["peaks_max"]),
#             "n_points_h"  : int(data["n_points_h"]),
#             "n_points_w"  : int(data["n_points_w"]),
#         }

#         print(f"  Dataset loaded ← {path}  "
#               f"({N} spectra, {spectra.shape[1]}×{spectra.shape[2]} pts, "
#               f"noise σ={meta['noise_level']:.3f})")
#         return spectra, peak_lists, ppm_lists, meta


# ================================================================
# SECTION 2 — CLASSICAL 2-D PEAK PICKING
# ================================================================

# class ClassicalPeakPicker2D:
#     """
#     Three classical peak-picking strategies for 2-D NMR spectra.
#     All return a list of (row, col) integer tuples (point coordinates).
#     """

#     @staticmethod
#     def _noise_floor(spectrum_2d, factor):
#         """
#         Estimate noise σ from spectrum corners (peak-free regions),
#         return factor × σ as the detection threshold.
#         """
#         H, W = spectrum_2d.shape
#         mh   = max(1, H // 10)   # 10% margin
#         mw   = max(1, W // 10)
#         corners = np.concatenate([
#             spectrum_2d[:mh, :mw].ravel(),
#             spectrum_2d[:mh, -mw:].ravel(),
#             spectrum_2d[-mh:, :mw].ravel(),
#             spectrum_2d[-mh:, -mw:].ravel(),
#         ])
#         return factor * max(float(np.std(corners)), 1e-6)

#     @staticmethod
#     def threshold(spectrum_2d, factor=4.0):
#         """
#         Simple intensity threshold + connected-component blob analysis.

#         Steps:
#           1. Compute noise-floor threshold from spectral corners
#           2. Binarise: pixels > threshold are "signal"
#           3. Label connected blobs with scipy.ndimage
#           4. Return the intensity-weighted centroid of each blob

#         Works well when peaks are well separated and noise is low.
#         Fails when noise spikes are above the threshold (false positives)
#         or when two nearby peaks merge into one blob (missed split).
#         """
#         thresh  = ClassicalPeakPicker2D._noise_floor(spectrum_2d, factor)
#         binary  = spectrum_2d > thresh
#         labeled, n = ndlabel(binary)
#         peaks = []
#         for bid in range(1, n + 1):
#             mask     = labeled == bid
#             blob_val = spectrum_2d * mask
#             # Intensity-weighted centroid (more accurate than argmax)
#             total = blob_val.sum()
#             if total > 0:
#                 rows = np.arange(spectrum_2d.shape[0])[:, None]
#                 cols = np.arange(spectrum_2d.shape[1])[None, :]
#                 cy   = int(round((blob_val * rows).sum() / total))
#                 cx   = int(round((blob_val * cols).sum() / total))
#                 peaks.append((cy, cx))
#         return peaks

#     @staticmethod
#     def local_maxima(spectrum_2d, factor=3.0, footprint_h=15, footprint_w=5):
#         """
#         Local-maxima method: a point is a peak if it is the strict
#         maximum in a (footprint_h × footprint_w) neighbourhood AND
#         its intensity exceeds the noise floor.

#         The asymmetric footprint reflects the typical NMR lineshape:
#         peaks are broader in ¹H (rows) than ¹⁵N (cols).

#         Better than threshold at separating nearby peaks, but sensitive
#         to the footprint size: too small → noise peaks; too large → merges.
#         """
#         thresh    = ClassicalPeakPicker2D._noise_floor(spectrum_2d, factor)
#         footprint = np.ones((footprint_h, footprint_w), dtype=bool)
#         local_max = maximum_filter(spectrum_2d, footprint=footprint) == spectrum_2d
#         hits      = np.argwhere(local_max & (spectrum_2d > thresh))
#         return [tuple(p) for p in hits]

#     @staticmethod
#     def matched_filter(spectrum_2d, lw_h=8.0, lw_n=3.0, factor=3.5):
#         """
#         2-D matched filter: cross-correlate the spectrum with a template
#         Lorentzian that matches the expected peak shape, then find local
#         maxima of the response map.

#         Because the template is matched to typical NMR linewidths, this
#         is theoretically optimal (maximum SNR) for detecting peaks of
#         that shape.  It degrades when peaks have unusual linewidths.
#         """
#         H, W = spectrum_2d.shape

#         # Build a normalised Lorentzian×Voigt template
#         rows = np.arange(H, dtype=np.float64)
#         cols = np.arange(W, dtype=np.float64)
#         cy, cx = H / 2.0, W / 2.0
#         lz_h = 1.0 / (1.0 + ((rows - cy) / (lw_h / 2.0)) ** 2)
#         lz_n = 1.0 / (1.0 + ((cols - cx) / (lw_n / 2.0)) ** 2)
#         template = np.outer(lz_h, lz_n).astype(np.float32)
#         template /= template.sum()

#         # Cross-correlation via FFT (wrapped in scipy gaussian_filter proxy)
#         # For speed we use gaussian_filter with matching σ as an approximation
#         sigma_h = lw_h / (2.0 * np.sqrt(2.0 * np.log(2.0)))
#         sigma_n = lw_n / (2.0 * np.sqrt(2.0 * np.log(2.0)))
#         response = gaussian_filter(spectrum_2d.astype(np.float64),
#                                    sigma=(sigma_h, sigma_n)).astype(np.float32)

#         thresh    = ClassicalPeakPicker2D._noise_floor(response, factor)
#         footprint = np.ones((max(3, int(lw_h)), max(3, int(lw_n))), dtype=bool)
#         local_max = maximum_filter(response, footprint=footprint) == response
#         hits      = np.argwhere(local_max & (response > thresh))
#         return [tuple(p) for p in hits]




# ================================================================
# SECTION 4 — VISION TRANSFORMER
# ================================================================

# def positional_embd_sin_cos(h, w, dim, temp=10_000, dtype=torch.float32):
#     """Fixed 2-D sinusoidal positional embedding for an (h × w) patch grid."""
#     y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
#     assert dim % 4 == 0, "dim must be divisible by 4"
#     omega = torch.arange(dim // 4) / (dim // 4 - 1)
#     omega = 1.0 / (temp ** omega)
#     y = y.flatten()[:, None] * omega[None, :]
#     x = x.flatten()[:, None] * omega[None, :]
#     pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
#     return pe.type(dtype)


# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim),
#         )
#     def forward(self, x):
#         return self.net(x)


# class Attention(nn.Module):
#     """
#     Multi-head self-attention.
#     BUG 1 FIXED: super().__init__() was missing → all sub-modules were plain
#                  Python attributes; the optimizer saw zero parameters.
#     BUG 2 FIXED: inner_dim = dim * heads → must be dim_head * heads.
#     """
#     def __init__(self, dim, heads=4, dim_head=32):
#         super().__init__()                      # FIX 1
#         inner_dim    = dim_head * heads         # FIX 2
#         self.heads   = heads
#         self.scale   = dim_head ** -0.5
#         self.norm    = nn.LayerNorm(dim)
#         self.attend  = nn.Softmax(dim=-1)
#         self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias=False)
#         self.to_out  = nn.Linear(inner_dim, dim,      bias=False)

#     def forward(self, x):
#         x       = self.norm(x)
#         q, k, v = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
#                       (q, k, v))
#         attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
#         out  = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
#         return self.to_out(out)


# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim):
#         super().__init__()
#         self.norm   = nn.LayerNorm(dim)
#         self.layers = nn.ModuleList([
#             nn.ModuleList([Attention(dim, heads=heads, dim_head=dim_head),
#                            FeedForward(dim, mlp_dim)])
#             for _ in range(depth)
#         ])
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x)   + x
#         return self.norm(x)


# class ViT_PP(nn.Module):
#     """
#     Vision Transformer for 2-D NMR peak picking.

#     Input : Tensor (B, 1, H, W)
#     Output: Tensor (B, 1, H, W) — per-pixel peak probability in [0, 1]

#     How it works
#     ------------
#     1. Divide the (H, W) spectrum into non-overlapping (ph × pw) patches.
#     2. Flatten each patch → linear projection → sequence of tokens.
#     3. Add 2-D sinusoidal positional embedding so the model knows where
#        each patch came from
#     4. Feed through Transformer encoder → one feature vector per token.
#     5. Project each token → 1 scalar score.
#     6. Reshape token scores to patch grid (gh × gw), then bilinearly
#        upsample back to (H, W) → per-pixel probability heatmap.

#     """

#     def __init__(self, *, spectra_size, patch_size, dim, depth, heads,
#                  mlp_dim, channels=1, dim_head=64):
#         super().__init__()
#         if not HAS_EINOPS:
#             raise ImportError("pip install einops")

#         sh, sw = spectra_size
#         ph, pw = patch_size
#         assert sh % ph == 0 and sw % pw == 0, \
#             "spectra_size must be exactly divisible by patch_size"

#         self.ph, self.pw = ph, pw
#         self.gh = sh // ph   # number of patches along ¹H axis
#         self.gw = sw // pw   # number of patches along ¹⁵N axis
#         patch_dim = channels * ph * pw

#         self.to_patch_embedding = nn.Sequential(
#             Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=ph, p2=pw),
#             nn.LayerNorm(patch_dim),
#             nn.Linear(patch_dim, dim),
#             nn.LayerNorm(dim),
#         )

#         self.register_buffer(
#             "pos_embedding",
#             positional_embd_sin_cos(h=self.gh, w=self.gw, dim=dim),
#         )

#         self.transformer = Transformer(dim=dim, depth=depth, heads=heads,
#                                        dim_head=dim_head, mlp_dim=mlp_dim)
#         self.patch_head  = nn.Linear(dim, 1)
#         self.upsample    = nn.Upsample(scale_factor=(ph, pw),
#                                        mode="bilinear", align_corners=False)
#         self.sigmoid     = nn.Sigmoid()

#     def forward(self, spectra):
#         # FIX 4: output shape is (B, 1, H, W), one probability per pixel
#         x = self.to_patch_embedding(spectra)          # (B, gh*gw, dim)
#         x = x + self.pos_embedding.to(dtype=x.dtype) # add positional info
#         x = self.transformer(x)                       # (B, gh*gw, dim)

#         scores = self.patch_head(x)                   # (B, gh*gw, 1)
#         scores = scores.permute(0, 2, 1)              # (B, 1, gh*gw)
#         scores = scores.reshape(-1, 1, self.gh, self.gw)  # (B, 1, gh, gw)
#         heatmap = self.upsample(scores)               # (B, 1, H, W)
#         return self.sigmoid(heatmap)


# ================================================================
# SECTION 5 — 2-D HEATMAP LABELS, DATASET, TRAINING
# ================================================================

def make_label_heatmap(peak_positions, height, width, sigma_y=8.0, sigma_x=3.0):
    """
    Build a (H, W) float32 Gaussian heatmap from peak (row, col) positions.

    Why Gaussian labels instead of binary point labels?
    ---------------------------------------------------
    A single-pixel label at each peak provides almost no gradient
    signal during training (99.9% of pixels would be 0).  A Gaussian
    blob spreads the target over ~σ² pixels, providing a smooth
    gradient field that trains much more stably.

    sigma_y / sigma_x are chosen to match the expected peak lineshape.
    """
    label = np.zeros((height, width), dtype=np.float32)
    y     = np.arange(height)
    x     = np.arange(width)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    for (cy, cx) in peak_positions:
        blob  = np.exp(
            -((yy - cy) ** 2) / (2 * sigma_y ** 2)
            -((xx - cx) ** 2) / (2 * sigma_x ** 2)
        )
        label = np.maximum(label, blob)
    return label


def _f1_2d(true_peaks, pred_peaks, tol_y=10, tol_x=4):
    """Peak-level F1 with separate row/col tolerances."""
    if not true_peaks and not pred_peaks:
        return 1.0
    if not true_peaks or not pred_peaks:
        return 0.0
    matched = set()
    tp = 0
    for ty, tx in true_peaks:
        for j, (py, px) in enumerate(pred_peaks):
            if abs(ty-py) <= tol_y and abs(tx-px) <= tol_x and j not in matched:
                tp += 1; matched.add(j); break
    fp = len(pred_peaks) - len(matched)
    fn = len(true_peaks)  - tp
    p  = tp / (tp + fp) if tp + fp > 0 else 0.0
    r  = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


class NMRDataset(Dataset):
    """
    PyTorch Dataset for 2-D NMR spectra.

    Input  : (1, H, W) Tensor  — single-channel spectrum image
    Target : (1, H, W) Tensor  — Gaussian heatmap label

    Both CNN and ViT use exactly this format; no 1-D projections anywhere.
    """

    def __init__(self, spectra_2d, peak_lists, sigma_y=8.0, sigma_x=3.0):
        H, W = spectra_2d[0].shape
        self.X, self.y = [], []
        for sp, peaks in zip(spectra_2d, peak_lists):
            label = make_label_heatmap(peaks, H, W, sigma_y, sigma_x)
            self.X.append(torch.FloatTensor(sp).unsqueeze(0))      # (1, H, W)
            self.y.append(torch.FloatTensor(label).unsqueeze(0))   # (1, H, W)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Training primitives ──────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)                    # (B, 1, H, W)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate_f1(model, loader, device, threshold=0.5,
                tolerance_y=10, tolerance_x=4):
    """
    Mean peak-level F1 over the dataset.

    For each spectrum:
      1. Threshold the predicted (H, W) heatmap → binary mask
      2. Label connected blobs → one candidate peak per blob
         (take the blob's intensity-weighted centroid)
      3. Match predicted peaks to true peaks within (tol_y, tol_x) pts
      4. Compute F1 from TP / FP / FN counts
    """
    model.eval()
    all_f1 = []
    for X, y in loader:
        preds  = model(X.to(device)).cpu().numpy()    # (B, 1, H, W)
        labels = y.numpy()                            # (B, 1, H, W)
        for pred_map, label_map in zip(preds[:, 0], labels[:, 0]):
            p_peaks = heatmap_to_peaks_2d(pred_map,  threshold=threshold)
            t_peaks = heatmap_to_peaks_2d(label_map, threshold=0.5)
            all_f1.append(_f1_2d(t_peaks, p_peaks, tolerance_y, tolerance_x))
    return float(np.mean(all_f1)) if all_f1 else 0.0


def heatmap_to_peaks_2d(heatmap, threshold=0.5):
    """Threshold a (H, W) probability map → list of (row, col) peak centres."""
    binary        = heatmap > threshold
    labeled, n    = ndlabel(binary)
    peaks         = []
    rows = np.arange(heatmap.shape[0])[:, None]
    cols = np.arange(heatmap.shape[1])[None, :]
    for bid in range(1, n + 1):
        mask  = (labeled == bid).astype(np.float32)
        vals  = heatmap * mask
        total = vals.sum()
        if total > 0:
            cy = int(round((vals * rows).sum() / total))
            cx = int(round((vals * cols).sum() / total))
            peaks.append((cy, cx))
    return peaks


def _f1_2d(true_peaks, pred_peaks, tol_y=10, tol_x=4):
    """Peak-level F1 with separate row/col tolerances."""
    if not true_peaks and not pred_peaks:
        return 1.0
    if not true_peaks or not pred_peaks:
        return 0.0
    matched = set()
    tp = 0
    for ty, tx in true_peaks:
        for j, (py, px) in enumerate(pred_peaks):
            if abs(ty-py) <= tol_y and abs(tx-px) <= tol_x and j not in matched:
                tp += 1; matched.add(j); break
    fp = len(pred_peaks) - len(matched)
    fn = len(true_peaks)  - tp
    p  = tp / (tp + fp) if tp + fp > 0 else 0.0
    r  = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def train_model(model, train_loader, val_loader, epochs, lr, device, label=""):
    """Training loop; returns (train_losses, val_f1s)."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn   = nn.BCELoss()
    tr_losses, val_f1s = [], []

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'─'*58}")
    print(f"  Training {label}  ({n_params:,} parameters)")
    print(f"{'─'*58}")

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        f1   = evaluate_f1(model, val_loader, device)
        tr_losses.append(loss); val_f1s.append(f1)
        if epoch % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={loss:.4f}  val_F1={f1:.3f}")

        if epoch % 5 == 0: #save every 5 epochs 
            torch.save({"epoch" : epoch, 
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict" : optimizer.state_dict(),
                    "loss" : loss, 
                    "f1" : f1,}, 
                    os.path.join(os.getcwd(), "saving_trained_models", f"{label}_epoch_{epoch}_checkpoint.pth"))


    return tr_losses, val_f1s


# ================================================================
# SECTION 6 — VISUALISATION (2-D, ppm-labelled axes)
# ================================================================

def plot_spectrum_2d(spectrum, true_peaks, pred_peaks=None, heatmap=None,
                     ppm_h_axis=None, ppm_n_axis=None, title="", path=None):
    """
    Three-panel figure:
      Panel 1 — raw 2-D NMR spectrum (imshow, viridis)
      Panel 2 — spectrum with true peaks (○) and predicted peaks (×) overlaid
      Panel 3 — model probability heatmap (hot colourmap)  [optional]

    Axes are labelled in ppm if ppm arrays are provided, otherwise in points.
    """
    n_panels = 3 if heatmap is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))

    H, W = spectrum.shape
    extent = None
    if ppm_h_axis is not None and ppm_n_axis is not None:
        # imshow extent: [left, right, bottom, top] in ppm
        extent = [ppm_n_axis[0], ppm_n_axis[-1],
                  ppm_h_axis[0], ppm_h_axis[-1]]
        xlabel = "¹⁵N chemical shift (ppm)"
        ylabel = "¹H chemical shift (ppm)"
    else:
        xlabel = "¹⁵N axis (points)"
        ylabel = "¹H axis (points)"

    def _pts_to_ppm(ry, rx):
        """Convert (row, col) in points → (ppm_h, ppm_n) for scatter."""
        if ppm_h_axis is None:
            return rx, ry   # x=col, y=row for imshow
        ph = ppm_h_axis[min(ry, H-1)]
        pn = ppm_n_axis[min(rx, W-1)]
        return pn, ph   # x=ppm_n, y=ppm_h

    # Panel 1: raw spectrum
    ax = axes[0]
    im = ax.imshow(spectrum, origin="lower", aspect="auto",
                   cmap="viridis", extent=extent, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Intensity")
    ax.set_title("2-D HSQC Spectrum")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    # Panel 2: annotations
    ax = axes[1]
    ax.imshow(spectrum, origin="lower", aspect="auto",
              cmap="viridis", extent=extent, interpolation="nearest")
    for i, (ry, rx) in enumerate(true_peaks):
        px, py = _pts_to_ppm(ry, rx)
        ax.plot(px, py, "o", ms=11, mfc="none", mec="#2ECC71", mew=2,
                label="True peak" if i == 0 else "")
    if pred_peaks:
        for i, (ry, rx) in enumerate(pred_peaks):
            px, py = _pts_to_ppm(ry, rx)
            ax.plot(px, py, "x", ms=10, mec="#E74C3C", mew=2.5,
                    label="Predicted" if i == 0 else "")
    handles = [plt.Line2D([0],[0], ls="none", marker="o", ms=9,
                           mfc="none", mec="#2ECC71", mew=2, label="True peak")]
    if pred_peaks:
        handles.append(plt.Line2D([0],[0], ls="none", marker="x", ms=9,
                                   mec="#E74C3C", mew=2.5, label="Predicted"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(title or "Peak Annotations")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    # Panel 3: heatmap
    if heatmap is not None:
        ax = axes[2]
        im2 = ax.imshow(heatmap, origin="lower", aspect="auto",
                        cmap="hot", vmin=0, vmax=1,
                        extent=extent, interpolation="nearest")
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="P(peak)")
        ax.set_title("Model Confidence Heatmap")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  → Saved {path}")
    return fig


def plot_classical_comparison(spectrum, true_peaks,
                               ppm_h_axis=None, ppm_n_axis=None, path=None):
    """Show ground truth + all three classical methods in a 4-panel figure."""
    picker  = ClassicalPeakPicker2D()
    methods = {
        "Threshold":      picker.threshold(spectrum),
        "Local Maxima":   picker.local_maxima(spectrum),
        "Matched Filter": picker.matched_filter(spectrum),
    }

    H, W = spectrum.shape
    extent = None
    if ppm_h_axis is not None:
        extent = [ppm_n_axis[0], ppm_n_axis[-1],
                  ppm_h_axis[0], ppm_h_axis[-1]]

    def _p(ry, rx):
        if ppm_h_axis is None:
            return rx, ry
        return ppm_n_axis[min(rx, W-1)], ppm_h_axis[min(ry, H-1)]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    titles = ["Ground Truth"] + list(methods.keys())
    pred_sets = [true_peaks] + list(methods.values())

    for ax, title_str, preds in zip(axes, titles, pred_sets):
        ax.imshow(spectrum, origin="lower", aspect="auto",
                  cmap="viridis", extent=extent, interpolation="nearest")
        # true peaks
        for ry, rx in true_peaks:
            ax.plot(*_p(ry, rx), "o", ms=10, mfc="none", mec="#2ECC71", mew=2)
        if title_str != "Ground Truth":
            for ry, rx in preds:
                ax.plot(*_p(ry, rx), "x", ms=9, mec="#E74C3C", mew=2.5)
            f1 = _f1_2d(true_peaks, preds)
            ax.set_title(f"{title_str}\nF1 = {f1:.2f}")
        else:
            ax.set_title(title_str)
        ax.set_xlabel("¹⁵N (ppm)" if extent else "col")
    axes[0].set_ylabel("¹H (ppm)" if extent else "row")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  → Saved {path}")
    return fig


def plot_learning_curves(cnn_losses, cnn_f1s,
                          vit_losses=None, vit_f1s=None,
                          path="learning_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(cnn_losses, label="U-Net CNN", color="#2176AE")
    if vit_losses: ax1.plot(vit_losses, label="ViT", color="#E63946")
    ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(cnn_f1s, label="U-Net CNN", color="#2176AE")
    if vit_f1s: ax2.plot(vit_f1s, label="ViT", color="#E63946")
    ax2.set_title("Validation F1"); ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score"); ax2.set_ylim(0, 1)
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"  → Saved {path}")


def plot_comparison_bar(results, path="comparison_summary.png"):
    metrics = ["Precision", "Recall", "F1"]
    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, [results["cnn"][m] for m in metrics], w,
                label="U-Net CNN", color="#2176AE", alpha=0.85)
    b2 = ax.bar(x + w/2, [results["vit"][m] for m in metrics], w,
                label="ViT", color="#E63946", alpha=0.85)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("U-Net CNN vs ViT: 2-D NMR Peak Picking")
    ax.legend(); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"  → Saved {path}")


# ================================================================
# SECTION 7 — HELPER FUNCTIONS
# ================================================================

def _make_loaders(data, peaks, batch_size=8, val_frac=0.2):
    n     = len(data)
    split = int(n * (1 - val_frac))
    tr_ds = NMRDataset(data[:split], peaks[:split])
    va_ds = NMRDataset(data[split:], peaks[split:])
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    return tr_ld, va_ld, split


def _fresh_cnn(device):
    return CNN_PeakDetector2D(in_channels=1, base_filters=16).to(device)


def _fresh_vit(device):
    if not HAS_EINOPS:
        return None
    return ViT_PP(
        spectra_size=(256, 32), patch_size=(16, 8),
        dim=64, depth=3, heads=4, dim_head=16, mlp_dim=128, channels=1,
    ).to(device)


def _collect_metrics(model, val_data, val_peaks, device, tol_y=10, tol_x=4):
    """Aggregate P/R/F1 over a numpy validation set."""
    tps = fps = fns = 0
    model.eval()
    with torch.no_grad():
        for sp, pk in zip(val_data, val_peaks):
            inp    = torch.FloatTensor(sp).unsqueeze(0).unsqueeze(0).to(device)
            hm     = model(inp).cpu().numpy()[0, 0]
            p_pk   = heatmap_to_peaks_2d(hm)
            matched = set()
            for ty, tx in pk:
                for j, (py, px) in enumerate(p_pk):
                    if abs(ty-py) <= tol_y and abs(tx-px) <= tol_x and j not in matched:
                        tps += 1; matched.add(j); break
                else:
                    fns += 1
            fps += len(p_pk) - len(matched)
    p  = tps / (tps + fps) if tps + fps > 0 else 0.0
    r  = tps / (tps + fns) if tps + fns > 0 else 0.0
    f1 = 2*p*r / (p+r) if p+r > 0 else 0.0
    return {"Precision": p, "Recall": r, "F1": f1}


# ================================================================
# SECTION 8 — EXPERIMENTS
# ================================================================

def _resolve_dataset(preloaded, n_samples, noise_level, label=""):
    """
    Return (data, peaks, ppm_h_axis, ppm_n_axis), either from a preloaded
    tuple or by generating fresh spectra.
    """
    if preloaded is not None:
        spectra, peak_lists, ppm_lists, meta = preloaded
        print(f"  Using preloaded dataset: {len(spectra)} spectra{label}")
        return spectra, peak_lists, meta["ppm_h_axis"], meta["ppm_n_axis"]
    gen = HSQCGenerator(num_samples=n_samples)
    data, peaks, _ = gen.generate_dataset(noise_level=noise_level)
    return data, peaks, gen.ppm_h, gen.ppm_n


def experiment_noise(device, epochs=15, preloaded=None):
    """Train once on moderate noise; evaluate at multiple noise levels."""
    print("\n" + "="*58)
    print("  Experiment: Noise Robustness")
    print("="*58)

    data, peaks, _, _ = _resolve_dataset(preloaded, n_samples=400,
                                          noise_level=0.02, label=" (training)")
    tr_ld, va_ld, _ = _make_loaders(data, peaks)

    cnn = _fresh_cnn(device); vit = _fresh_vit(device)
    train_model(cnn, tr_ld, va_ld, epochs, lr=1e-3, device=device, label="CNN")
    if vit: train_model(vit, tr_ld, va_ld, epochs, lr=5e-4, device=device, label="ViT")

    noise_levels = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    cnn_f1s, vit_f1s = [], []
    print(f"\n  {'Noise σ':>8}  {'CNN F1':>8}  {'ViT F1':>8}")

    for nl in noise_levels:
        te_d, te_p, _ = HSQCGenerator(num_samples=150).generate_dataset(noise_level=nl)
        te_ld = DataLoader(NMRDataset(te_d, te_p), batch_size=8)
        c = evaluate_f1(cnn, te_ld, device)
        v = evaluate_f1(vit, te_ld, device) if vit else 0.0
        cnn_f1s.append(c); vit_f1s.append(v)
        print(f"  {nl:>8.3f}  {c:>8.3f}  {v:>8.3f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(noise_levels, cnn_f1s, "o-", label="U-Net CNN", color="#2176AE")
    if vit: ax.plot(noise_levels, vit_f1s, "s-", label="ViT", color="#E63946")
    ax.set_xlabel("Noise level σ (relative to peak amplitude)")
    ax.set_ylabel("F1 Score"); ax.set_title("Noise Robustness: 2-D CNN vs ViT")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("experiment_noise.png", dpi=120)
    print("  → Saved experiment_noise.png")


def experiment_overlap(device, epochs=15, preloaded=None):
    """Test on spectra with peaks forced to be closer and closer together."""
    print("\n" + "="*58)
    print("  Experiment: Peak Overlap (minimum ¹H separation)")
    print("="*58)

    data, peaks, _, _ = _resolve_dataset(preloaded, n_samples=400,
                                          noise_level=0.02, label=" (training)")
    tr_ld, va_ld, _ = _make_loaders(data, peaks)

    cnn = _fresh_cnn(device); vit = _fresh_vit(device)
    train_model(cnn, tr_ld, va_ld, epochs, lr=1e-3, device=device, label="CNN")
    if vit: train_model(vit, tr_ld, va_ld, epochs, lr=5e-4, device=device, label="ViT")

    min_seps_pts = [5, 10, 20, 40, 80]   # minimum row separation in points
    cnn_f1s, vit_f1s = [], []
    print(f"\n  {'Sep(pts)':>9}  {'Sep(ppm)':>9}  {'CNN F1':>8}  {'ViT F1':>8}")

    sub_gen = HSQCGenerator(num_samples=1)  # re-use helper methods
    for sep in min_seps_pts:
        spectra, all_peaks = [], []
        sw_h = sub_gen.sw_h; H = sub_gen.H
        sep_ppm = sep / (H / sw_h)
        for _ in range(150):
            n_pk = 3
            positions = []
            attempts  = 0
            while len(positions) < n_pk and attempts < 2000:
                cy = np.random.randint(20, H - 20)
                cx = np.random.randint(4, sub_gen.W - 4)
                if all(abs(cy - p[0]) >= sep for p in positions):
                    positions.append((cy, cx))
                attempts += 1
            sp = np.zeros((H, sub_gen.W), dtype=np.float32)
            for cy, cx in positions:
                amp = np.random.uniform(0.5, 1.0)
                lw_h = np.random.uniform(5, 12)
                lw_n = np.random.uniform(2,  5)
                sp += sub_gen._peak_2d(cy, cx, amp, lw_h, lw_n)
            sp += np.random.normal(0, 0.02, sp.shape).astype(np.float32)
            sp  = np.clip(sp, 0, None)
            spectra.append(sp); all_peaks.append(positions)

        te_ld = DataLoader(NMRDataset(spectra, all_peaks), batch_size=8)
        c = evaluate_f1(cnn, te_ld, device)
        v = evaluate_f1(vit, te_ld, device) if vit else 0.0
        cnn_f1s.append(c); vit_f1s.append(v)
        print(f"  {sep:>9d}  {sep_ppm:>9.3f}  {c:>8.3f}  {v:>8.3f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(min_seps_pts, cnn_f1s, "o-", label="U-Net CNN", color="#2176AE")
    if vit: ax.plot(min_seps_pts, vit_f1s, "s-", label="ViT", color="#E63946")
    ax.set_xlabel("Minimum ¹H separation between peaks (points)")
    ax.set_ylabel("F1 Score"); ax.set_title("Overlap Robustness: 2-D CNN vs ViT")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("experiment_overlap.png", dpi=120)
    print("  → Saved experiment_overlap.png")


def experiment_datasize(device, epochs=20, preloaded=None):
    """Learning curve: how much HSQC training data does each model need?"""
    print("\n" + "="*58)
    print("  Experiment: Training Data Size (Learning Curve)")
    print("="*58)

    all_data, all_peaks, _, _ = _resolve_dataset(preloaded, n_samples=700,
                                                   noise_level=0.02)

    va_data, va_peaks = all_data[600:], all_peaks[600:]
    va_ld = DataLoader(NMRDataset(va_data, va_peaks), batch_size=8)

    sizes = [30, 75, 150, 300, 500]
    cnn_f1s, vit_f1s = [], []
    print(f"\n  {'N train':>8}  {'CNN F1':>8}  {'ViT F1':>8}")

    for sz in sizes:
        tr_ld = DataLoader(NMRDataset(all_data[:sz], all_peaks[:sz]),
                           batch_size=min(8, sz), shuffle=True, drop_last=True)
        cnn = _fresh_cnn(device); vit = _fresh_vit(device)
        train_model(cnn, tr_ld, va_ld, epochs, lr=1e-3,
                    device=device, label=f"CNN n={sz}")
        c = evaluate_f1(cnn, va_ld, device)
        v = 0.0
        if vit:
            train_model(vit, tr_ld, va_ld, epochs, lr=5e-4,
                        device=device, label=f"ViT n={sz}")
            v = evaluate_f1(vit, va_ld, device)
        cnn_f1s.append(c); vit_f1s.append(v)
        print(f"  {sz:>8d}  {c:>8.3f}  {v:>8.3f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sizes, cnn_f1s, "o-", label="U-Net CNN", color="#2176AE")
    if vit: ax.plot(sizes, vit_f1s, "s-", label="ViT", color="#E63946")
    ax.set_xlabel("Training set size (number of HSQC spectra)")
    ax.set_ylabel("F1 Score"); ax.set_title("Learning Curve: 2-D CNN vs ViT")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("experiment_datasize.png", dpi=120)
    print("  → Saved experiment_datasize.png")


# ================================================================
# SECTION 9 — FULL CNN vs ViT COMPARISON
# ================================================================

def run_comparison(device, epochs=25, num_samples=500, preloaded=None):
    print("\n" + "="*58)
    print("  Full Comparison: U-Net CNN vs ViT on HSQC Data")
    print("="*58)

    data, peaks, ppm_h_ax, ppm_n_ax = _resolve_dataset(
        preloaded, n_samples=num_samples, noise_level=0.02)
    tr_ld, va_ld, split = _make_loaders(data, peaks)

    cnn = _fresh_cnn(device); vit = _fresh_vit(device)
    cnn_losses, cnn_f1s = train_model(cnn, tr_ld, va_ld, epochs, lr=1e-3,
                                       device=device, label="U-Net CNN")
    vit_losses, vit_f1s = [], []
    if vit:
        vit_losses, vit_f1s = train_model(vit, tr_ld, va_ld, epochs, lr=5e-4,
                                           device=device, label="ViT")
    plot_learning_curves(cnn_losses, cnn_f1s,
                          vit_losses or None, vit_f1s or None)

    va_data  = data[split:]
    va_peaks = peaks[split:]

    # Sample prediction plots
    for model, name in [(cnn, "cnn"), (vit, "vit")] if vit else [(cnn, "cnn")]:
        if model is None: continue
        model.eval()
        for i in range(min(3, len(va_data))):
            sp, pk = va_data[i], va_peaks[i]
            with torch.no_grad():
                inp = torch.FloatTensor(sp).unsqueeze(0).unsqueeze(0).to(device)
                hm  = model(inp).cpu().numpy()[0, 0]
            pred_pk = heatmap_to_peaks_2d(hm)
            plot_spectrum_2d(sp, pk, pred_pk, hm,
                             ppm_h_axis=ppm_h_ax, ppm_n_axis=ppm_n_ax,
                             title=f"{name.upper()} Sample {i+1}  "
                                   f"F1={_f1_2d(pk, pred_pk):.2f}",
                             path=f"{name}_sample_{i+1}.png")

    # Classical methods on first validation spectrum
    plot_classical_comparison(va_data[0], va_peaks[0],
                               ppm_h_axis=ppm_h_ax, ppm_n_axis=ppm_n_ax,
                               path="classical_comparison.png")

    results = {
        "cnn": _collect_metrics(cnn, va_data, va_peaks, device),
        "vit": _collect_metrics(vit, va_data, va_peaks, device) if vit
               else {"Precision": 0.0, "Recall": 0.0, "F1": 0.0},
    }
    plot_comparison_bar(results)

    print("\n  ┌──────────────┬───────────┬───────────┬───────────┐")
    print(  "  │    Model     │ Precision │  Recall   │    F1     │")
    print(  "  ├──────────────┼───────────┼───────────┼───────────┤")
    for name, m in results.items():
        print(f"  │ {name.upper():<12} │  {m['Precision']:.3f}    │"
              f"  {m['Recall']:.3f}    │  {m['F1']:.3f}    │")
    print(  "  └──────────────┴───────────┴───────────┴───────────┘")


# ================================================================
# SECTION 10 — QUICK DEMO
# ================================================================

def run_demo(device, preloaded=None):
    """2-minute sanity-check: generate (or load) HSQC spectra, train briefly, plot."""
    print("\n[DEMO — HSQC spectra, 5 epochs, fully 2-D]\n")

    output_cnn = os.path.join(os.getcwd(), "output_path_cnn")
    os.makedirs(output_cnn, exist_ok = True)
    
    output_vit = os.path.join(os.getcwd(), "output_path_vit")
    os.makedirs(output_vit, exist_ok = True)

    data, peaks, ppm_h_ax, ppm_n_ax = _resolve_dataset(
        preloaded, n_samples=80, noise_level=0.02)
    tr_ld, va_ld, split = _make_loaders(data, peaks, batch_size=8)

    cnn = _fresh_cnn(device)
    vit = _fresh_vit(device)
   
    train_model(cnn, tr_ld, va_ld, epochs=20, lr=1e-3,
                device=device, label="U-Net CNN (demo)")

    train_model(vit, tr_ld, va_ld, epochs=20, lr=1e-3, device=device, label="ViT (demo)")

    cnn.eval()
    for i in range(min(2, len(data) - split)):
        sp, pk = data[split + i], peaks[split + i]
        with torch.no_grad():
            inp = torch.FloatTensor(sp).unsqueeze(0).unsqueeze(0).to(device)
            hm  = cnn(inp).cpu().numpy()[0, 0]
        pred_pk = heatmap_to_peaks_2d(hm)
        plot_spectrum_2d(sp, pk, pred_pk, hm,
                         ppm_h_axis=ppm_h_ax, ppm_n_axis=ppm_n_ax,
                         title=f"Demo Sample {i+1}  F1={_f1_2d(pk, pred_pk):.2f}",
                         path=os.path.join(output_cnn, f"demo_sample_{i+1}.png"))

    plot_classical_comparison(data[split], peaks[split],
                               ppm_h_axis=ppm_h_ax, ppm_n_axis=ppm_n_ax,
                               path=os.path.join(output_cnn, "demo_classical.png"))
    
    #------------------------ViT-----------------------------------
    #-----------------------------------------------------------
    # Show 2 validation spectra: raw, predicted, heatmap
    vit.eval()
    for i in range(min(2, len(data) - split)):
        sp_vit, pk_vit = data[split + i], peaks[split + i]
        with torch.no_grad():
            inp_vit = torch.FloatTensor(sp_vit).unsqueeze(0).unsqueeze(0).to(device)
            hm_vit  = vit(inp_vit).cpu().numpy()[0, 0]
        pred_pk_vit = heatmap_to_peaks_2d(hm_vit)
        plot_spectrum_2d(sp_vit, pk_vit, pred_pk_vit, hm_vit,
                         ppm_h_axis=ppm_h_ax, ppm_n_axis=ppm_n_ax,
                         title=f"Demo Sample {i+1}  "
                               f"F1={_f1_2d(pk_vit, pred_pk_vit):.2f}",
                         path=os.path.join(output_vit,f"demo_sample_vit_{i+1}.png"))

    # Compare three classical methods
    plot_classical_comparison(data[split], peaks[split],
                               ppm_h_axis=ppm_h_ax, ppm_n_axis=ppm_h_ax,
                               path=os.path.join(output_vit, "demo_classical.png"))
    
    print("\nDemo complete!  See demo_sample_*.png and demo_classical.png")







# def run_demo(device):
#     """2-minute sanity-check: generate a few HSQC spectra, train briefly, plot."""
#     print("\n[DEMO — 80 HSQC spectra, 5 epochs, fully 2-D]\n")

#     gen  = HSQCGenerator(num_samples=80)
#     data, peaks, ppm_lists = gen.generate_dataset(noise_level=0.02)
#     tr_ld, va_ld, split = _make_loaders(data, peaks, batch_size=8)

#     cnn = _fresh_cnn(device)
#     vit = _fresh_vit(device)

#     train_model(cnn, tr_ld, va_ld, epochs=20, lr=1e-3,
#                 device=device, label="U-Net CNN (demo)")
    

#     train_model(vit, tr_ld, va_ld, epochs=20, lr=1e-3,
#                 device=device, label="ViT CNN (demo)")

#     # Show 2 validation spectra: raw, predicted, heatmap
#     cnn.eval()
#     for i in range(min(2, len(data) - split)):
#         sp, pk = data[split + i], peaks[split + i]
#         with torch.no_grad():
#             inp = torch.FloatTensor(sp).unsqueeze(0).unsqueeze(0).to(device)
#             hm  = cnn(inp).cpu().numpy()[0, 0]
#         pred_pk = heatmap_to_peaks_2d(hm)
#         plot_spectrum_2d(sp, pk, pred_pk, hm,
#                          ppm_h_axis=gen.ppm_h, ppm_n_axis=gen.ppm_n,
#                          title=f"Demo Sample {i+1}  "
#                                f"F1={_f1_2d(pk, pred_pk):.2f}",
#                          path=f"demo_sample_cnn_{i+1}.png")

#     # Compare three classical methods
#     plot_classical_comparison(data[split], peaks[split],
#                                ppm_h_axis=gen.ppm_h, ppm_n_axis=gen.ppm_n,
#                                path="demo_classical.png")
#     print("\nDemo complete!  See demo_sample_*.png and demo_classical.png")


#     #------------------------ViT-----------------------------------
#     #-----------------------------------------------------------
#     # Show 2 validation spectra: raw, predicted, heatmap
#     vit.eval()
#     for i in range(min(2, len(data) - split)):
#         sp_vit, pk_vit = data[split + i], peaks[split + i]
#         with torch.no_grad():
#             inp_vit = torch.FloatTensor(sp_vit).unsqueeze(0).unsqueeze(0).to(device)
#             hm_vit  = vit(inp_vit).cpu().numpy()[0, 0]
#         pred_pk_vit = heatmap_to_peaks_2d(hm_vit)
#         plot_spectrum_2d(sp_vit, pk_vit, pred_pk_vit, hm_vit,
#                          ppm_h_axis=gen.ppm_h, ppm_n_axis=gen.ppm_n,
#                          title=f"Demo Sample {i+1}  "
#                                f"F1={_f1_2d(pk_vit, pred_pk_vit):.2f}",
#                          path=f"demo_sample_vit_{i+1}.png")

#     # Compare three classical methods
#     plot_classical_comparison(data[split], peaks[split],
#                                ppm_h_axis=gen.ppm_h, ppm_n_axis=gen.ppm_n,
#                                path="demo_classical.png")
#     print("\nDemo complete!  See demo_sample_*.png and demo_classical.png")



# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="2-D HSQC NMR Peak Picking: CNN vs ViT")
    parser.add_argument("--demo",       action="store_true",
                        help="Quick 2-min demo with 80 spectra")
    parser.add_argument("--compare",    action="store_true",
                        help="Full CNN vs ViT head-to-head comparison")
    parser.add_argument("--experiment", choices=["noise", "overlap", "datasize"],
                        help="Run a specific experiment")

    # ── Dataset save / load ────────────────────────────────────────
    parser.add_argument(
        "--save-dataset", metavar="PATH",
        help=(
            "Generate a dataset and save it to PATH.npz, then exit.  "
            "Example: python nmr_pipeline.py --save-dataset hsqc_train"
        ),
    )

    # parser.add_argument("--output_path_cnn", metavar="PATH", 
    #                     help = ("Path to save the CNN model generated outputs"))
    
    # parser.add_argument("--output_path_cnn", metavar="PATH", 
    #                     help = ("Path to save the ViT model generated outputs"))


    parser.add_argument(
        "--load-dataset", metavar="PATH",
        help=(
            "Load a previously saved dataset from PATH.npz instead of "
            "generating new spectra. Works with --compare and --experiment.  "
            "Example: python nmr_pipeline.py --compare --load-dataset hsqc_train"
        ),
    )
    parser.add_argument("--n-samples",   type=int, default=500,
                        help="Number of spectra to generate (default: 500)")
    parser.add_argument("--noise-level", type=float, default=0.02,
                        help="Gaussian noise σ during generation (default: 0.02)")

    parser.add_argument("--real_data", metavar="PATH" , 
                        help = "Path to the .ft2 data of spectrum, instead of simulating the data. Usage --real_data asyn.ft2")
    
    parser.add_argument("--peak_tab", metavar="PATH" , 
                        help = "Path to the .tab tabular data of peaks, instead of simulating the data. Usage --peak_tab asyn_peacks.tab")
    
    parser.add_argument("--inspect", action="store_true", help = "Inspect the tabluar and the .ft2 data, print summaries and plots. Usage --inspect asyn_peacks.tab")
    
    parser.add_argument("--patch_w", type = int, default=32, help = "Tile width in points, set to 32 by default.")

    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── --save-dataset: generate + save, then exit ─────────────────
    if args.save_dataset:
        gen = HSQCGenerator(num_samples=args.n_samples)
        print(f"Generating {args.n_samples} HSQC spectra "
              f"(noise σ={args.noise_level}) …")
        gen.generate_dataset(
            noise_level=args.noise_level,
            save_path=args.save_dataset,
        )
        return

    # ── optional pre-loaded dataset (passed into run_* functions) ──
    preloaded = None
    if args.load_dataset:
        spectra, peak_lists, ppm_lists, meta = \
            HSQCGenerator.load_dataset(args.load_dataset)
        preloaded = (spectra, peak_lists, ppm_lists, meta)

    if args.demo:

        run_demo(device, preloaded=preloaded)
    elif args.experiment == "noise":
        experiment_noise(device, preloaded=preloaded)
    elif args.experiment == "overlap":
        experiment_overlap(device, preloaded=preloaded)
    elif args.experiment == "datasize":
        experiment_datasize(device, preloaded=preloaded)
    elif args.compare:
        run_comparison(device, num_samples=args.n_samples, preloaded=preloaded)
    else:
        run_demo(device, preloaded=preloaded)
        run_comparison(device, epochs=20, num_samples=args.n_samples,
                       preloaded=preloaded)
        experiment_noise(device, epochs=15)


if __name__ == "__main__":
    main()
    # data = np.load("nmr_dataset.npz", allow_pickle=True)

    # spectra = data["spectra"]
    # peak_lists = data["peak_lists"].tolist()
    # ppm_lists = data["ppm_lists"].tolist()