"""
export_student_images.py

Exports a student-ready dataset from generated or loaded HSQC spectra:

  student_data/
  ├── spectra/
  │   ├── spectrum_0001.png      ← clean 2D spectrum image (no annotations)
  │   ├── spectrum_0002.png
  │   └── ...
  ├── ground_truth/
  │   ├── spectrum_0001_gt.png   ← same image with true peaks marked
  │   └── ...
  ├── peak_lists/
  │   ├── spectrum_0001_peaks.csv  ← peak positions in ppm + points
  │   └── ...
  └── summary.csv                ← one row per spectrum (n_peaks, noise info)

Usage (standalone):
    python export_student_images.py --load-dataset hsqc_train.npz --out student_data/
    python export_student_images.py --generate --n-samples 50    --out student_data/

Usage (called from nmr_pipeline.py):
    from export_student_images import export_student_dataset
    export_student_dataset(spectra, peak_lists, ppm_h_axis, ppm_n_axis,
                           out_dir="student_data")
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Core export function ───────────────────────────────────────────────────────

def export_student_dataset(
    spectra,          # ndarray (N, H, W)
    peak_lists,       # list[list[(row, col)]]
    ppm_h_axis,       # ndarray (H,)  — ¹H  ppm per row
    ppm_n_axis,       # ndarray (W,)  — ¹⁵N ppm per col
    out_dir="student_data",
    colormap="viridis",
    dpi=150,
    zero_indexed=False,   # True → filenames start at 0, False → start at 1
    verbose=True,
):
    """
    Save one clean PNG + one annotated PNG + one CSV per spectrum.

    Parameters
    ----------
    spectra       : (N, H, W) float32 array
    peak_lists    : list of lists of (row, col) integer tuples
    ppm_h_axis    : (H,) float32 — ¹H  chemical shift axis
    ppm_n_axis    : (W,) float32 — ¹⁵N chemical shift axis
    out_dir       : root output directory
    colormap      : matplotlib colormap name for the spectrum
    dpi           : image resolution
    zero_indexed  : if True, filenames start at spectrum_0000
    verbose       : print progress
    """
    N, H, W = spectra.shape

    # Subdirectory layout
    dir_spectra = os.path.join(out_dir, "spectra")
    dir_gt      = os.path.join(out_dir, "ground_truth")
    dir_peaks   = os.path.join(out_dir, "peak_lists")
    for d in [dir_spectra, dir_gt, dir_peaks]:
        os.makedirs(d, exist_ok=True)

    # imshow extent: [left, right, bottom, top] in ppm
    # Note: ¹H increases downward in NMR convention (origin="upper")
    extent = [
        ppm_n_axis[0],  ppm_n_axis[-1],   # ¹⁵N: left → right
        ppm_h_axis[-1], ppm_h_axis[0],    # ¹H:  bottom = high ppm (origin upper)
    ]

    summary_rows = []

    for i in range(N):
        idx    = i if zero_indexed else i + 1
        stem   = f"spectrum_{idx:04d}"
        sp     = spectra[i]           # (H, W)
        peaks  = peak_lists[i]        # [(row, col), ...]

        # ── 1. Clean spectrum image (no annotations) ──────────────
        fig, ax = _make_spectrum_ax(sp, extent, ppm_h_axis, ppm_n_axis,
                                    colormap=colormap)
        clean_path = os.path.join(dir_spectra, f"{stem}.png")
        fig.savefig(clean_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # ── 2. Ground truth image (peaks overlaid) ────────────────
        fig, ax = _make_spectrum_ax(sp, extent, ppm_h_axis, ppm_n_axis,
                                    colormap=colormap)
        _overlay_peaks(ax, peaks, ppm_h_axis, ppm_n_axis, H, W)
        ax.set_title(f"Ground truth — {len(peaks)} peak(s)", fontsize=10, pad=6)
        gt_path = os.path.join(dir_gt, f"{stem}_gt.png")
        fig.savefig(gt_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        # ── 3. Peak list CSV ──────────────────────────────────────
        rows = []
        for k, (ry, rx) in enumerate(peaks):
            ph = float(ppm_h_axis[min(ry, H - 1)])
            pn = float(ppm_n_axis[min(rx, W - 1)])
            rows.append({
                "peak_id":       k + 1,
                "row_pts":       ry,
                "col_pts":       rx,
                "1H_ppm":        round(ph, 4),
                "15N_ppm":       round(pn, 4),
            })
        peak_df = pd.DataFrame(rows)
        csv_path = os.path.join(dir_peaks, f"{stem}_peaks.csv")
        peak_df.to_csv(csv_path, index=False)

        # ── 4. Summary row ────────────────────────────────────────
        summary_rows.append({
            "spectrum_id":   stem,
            "n_peaks":       len(peaks),
            "spectrum_file": os.path.relpath(clean_path, out_dir),
            "gt_file":       os.path.relpath(gt_path,    out_dir),
            "peaks_file":    os.path.relpath(csv_path,   out_dir),
        })

        if verbose and (i == 0 or (i + 1) % max(1, N // 10) == 0 or i == N - 1):
            print(f"  [{i+1:>{len(str(N))}}/{N}]  {stem}  ({len(peaks)} peaks)")

    # ── 5. Master summary CSV ─────────────────────────────────────
    summary_df  = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    if verbose:
        print(f"\n  Export complete → {out_dir}/")
        print(f"    {N} clean spectra      → spectra/")
        print(f"    {N} ground truth imgs  → ground_truth/")
        print(f"    {N} peak list CSVs     → peak_lists/")
        print(f"    summary               → summary.csv")

    return summary_df


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _make_spectrum_ax(sp, extent, ppm_h_axis, ppm_n_axis, colormap="viridis"):
    """
    Single-panel clean spectrum figure.
    Uses NMR convention: ¹H axis increases downward (origin='upper'),
    ¹⁵N axis increases left-to-right.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(
        sp,
        origin="upper",          # row 0 = low ¹H ppm (top of image)
        aspect="auto",
        cmap=colormap,
        extent=extent,
        interpolation="nearest",
    )

    ax.set_xlabel("¹⁵N chemical shift (ppm)", fontsize=10)
    ax.set_ylabel("¹H chemical shift (ppm)",  fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.tick_params(labelsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Intensity", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig, ax


def _overlay_peaks(ax, peaks, ppm_h_axis, ppm_n_axis, H, W):
    """Overlay true peak positions as circles on an existing spectrum axis."""
    for k, (ry, rx) in enumerate(peaks):
        ph = float(ppm_h_axis[min(ry, H - 1)])
        pn = float(ppm_n_axis[min(rx, W - 1)])
        ax.plot(
            pn, ph,
            "o",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor="#2ECC71",
            markeredgewidth=2.0,
            zorder=5,
        )
        ax.text(
            pn + 0.15, ph,           # slight offset so label doesn't overlap circle
            str(k + 1),
            fontsize=7,
            color="#2ECC71",
            va="center",
            zorder=6,
        )


# ── Optional contour overlay ───────────────────────────────────────────────────

def export_with_contours(
    spectra, peak_lists, ppm_h_axis, ppm_n_axis,
    out_dir="student_data_contour",
    n_levels=8, dpi=150, verbose=True,
):
    """
    Like export_student_dataset but renders spectra as contour plots
    (closer to how NMR software like CCPN or Sparky displays spectra).

    Useful if you want students to work with a more realistic representation.
    """
    N, H, W = spectra.shape
    dir_spectra = os.path.join(out_dir, "spectra_contour")
    dir_gt      = os.path.join(out_dir, "ground_truth_contour")
    os.makedirs(dir_spectra, exist_ok=True)
    os.makedirs(dir_gt,      exist_ok=True)

    ppm_h_grid, ppm_n_grid = np.meshgrid(ppm_h_axis, ppm_n_axis, indexing="ij")

    for i in range(N):
        idx  = i + 1
        stem = f"spectrum_{idx:04d}"
        sp   = spectra[i]
        pks  = peak_lists[i]

        # Compute contour levels: log-spaced from noise floor to max
        noise_floor = float(np.std(sp[:H//10, :W//10])) * 3.0
        sp_max      = float(sp.max())
        if sp_max > noise_floor:
            levels = np.logspace(np.log10(noise_floor), np.log10(sp_max), n_levels)
        else:
            levels = np.linspace(noise_floor, sp_max + 1e-6, n_levels)

        for annotate, suffix, save_dir in [
            (False, "",     dir_spectra),
            (True,  "_gt",  dir_gt),
        ]:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.contour(ppm_n_grid, ppm_h_grid, sp,
                       levels=levels, cmap="Blues", linewidths=0.8)
            ax.set_xlabel("¹⁵N chemical shift (ppm)", fontsize=10)
            ax.set_ylabel("¹H chemical shift (ppm)",  fontsize=10)
            ax.invert_xaxis()   # NMR convention: ¹⁵N decreases left→right
            ax.invert_yaxis()   # NMR convention: ¹H  decreases top→bottom
            ax.tick_params(labelsize=8)

            if annotate:
                _overlay_peaks(ax, pks, ppm_h_axis, ppm_n_axis, H, W)
                ax.set_title(f"Ground truth — {len(pks)} peak(s)",
                             fontsize=10, pad=6)

            fig.tight_layout()
            path = os.path.join(save_dir, f"{stem}{suffix}.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        if verbose and (i == 0 or (i+1) % max(1, N//10) == 0 or i == N-1):
            print(f"  [{i+1:>{len(str(N))}}/{N}]  {stem} (contour)")

    if verbose:
        print(f"\n  Contour export complete → {out_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export student-ready 2D HSQC images from a dataset")
    parser.add_argument("--load-dataset", metavar="PATH",
                        help="Path to .npz file saved by HSQCGenerator.save_dataset()")
    parser.add_argument("--generate",     action="store_true",
                        help="Generate a fresh dataset instead of loading")
    parser.add_argument("--n-samples",    type=int,   default=50)
    parser.add_argument("--noise-level",  type=float, default=0.02)
    parser.add_argument("--out",          default="student_data",
                        help="Output directory")
    parser.add_argument("--contour",      action="store_true",
                        help="Also export contour-style images")
    parser.add_argument("--dpi",          type=int,   default=150)
    args = parser.parse_args()

    # Import HSQCGenerator only when running as CLI (avoids circular imports)
    from simulating_data import HSQCGenerator

    if args.load_dataset:
        spectra, peak_lists, _, meta = HSQCGenerator.load_dataset(args.load_dataset)
        ppm_h_axis = meta["ppm_h_axis"]
        ppm_n_axis = meta["ppm_n_axis"]
    elif args.generate:
        gen = HSQCGenerator(num_samples=args.n_samples)
        spectra, peak_lists, _ = gen.generate_dataset(noise_level=args.noise_level)
        ppm_h_axis, ppm_n_axis = gen.ppm_h, gen.ppm_n
    else:
        parser.error("Provide --load-dataset PATH or --generate")

    export_student_dataset(
        spectra, peak_lists, ppm_h_axis, ppm_n_axis,
        out_dir=args.out,
        dpi=args.dpi,
    )

    if args.contour:
        export_with_contours(
            spectra, peak_lists, ppm_h_axis, ppm_n_axis,
            out_dir=args.out,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()