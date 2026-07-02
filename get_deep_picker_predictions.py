"""
get_deep_picker_predictions.py
-------------------------------
Run DEEP Picker on all 500 spectra and save the predicted peaks to a CSV.

Output CSV columns:
    spectrum_id   – 1-based index
    x_ppm         – ¹H  chemical shift (ppm)
    y_ppm         – ¹⁵N chemical shift (ppm)
    row           – ¹H  pixel index (0-based, matches numpy array rows)
    col           – ¹⁵N pixel index (0-based, matches numpy array cols)
    height        – peak intensity reported by DEEP Picker
    confidence    – DEEP Picker confidence score

Usage:
    python get_deep_picker_predictions.py
"""

import subprocess, csv
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
FT2_DIR     = ROOT / "train_ft2_F"
DEEP_BINARY = ROOT / "deep/build/deep_picker"
TAB_DIR     = ROOT / "deep_picker_results"
CSV_OUT     = ROOT / "deep_picker_predictions.csv"
NPZ_PATH    = ROOT / "hsqc_train.npz"

TAB_DIR.mkdir(exist_ok=True)


# ── Step 1: run DEEP Picker on every spectrum ──────────────────────────────────

def run_deep_picker(spectrum_id: int) -> Path:
    """Call the binary for one spectrum; return the path to the .tab output."""
    ft2 = FT2_DIR / f"spectrum_{spectrum_id:04d}.ft2"
    tab = TAB_DIR / f"spectrum_{spectrum_id:04d}_peaks.tab"

    if tab.exists():          # skip if already done
        return tab

    cmd = [
        str(DEEP_BINARY),
        "-in",       str(ft2),
        "-out",      str(tab),
        "-model",    "2",
        "-auto_ppp", "no",    # indirect dim too small (32 pts) for auto width
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] spectrum {spectrum_id:04d} failed: {result.stderr[-200:]}")
    return tab


# ── Step 2: parse one .tab file ────────────────────────────────────────────────

def parse_tab(tab_path: Path, ppm_h_axis: np.ndarray, ppm_n_axis: np.ndarray) -> list:
    """
    Read a DEEP Picker .tab file.
    Returns a list of dicts with keys: x_ppm, y_ppm, row, col, height, confidence.

    .tab format (relevant columns):
        INDEX  X_AXIS  Y_AXIS  X_PPM  Y_PPM  XW  YW  X1  X3  Y1  Y3  HEIGHT  ASS  CONFIDENCE  POINTER
    """
    if not tab_path.exists():
        return []

    peaks = []
    with open(tab_path) as fh:
        for line in fh:
            line = line.strip()
            # skip header lines
            if not line or line.startswith(("DATA", "VARS", "FORMAT")):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                x_ppm      = float(parts[3])   # ¹H  ppm
                y_ppm      = float(parts[4])   # ¹⁵N ppm
                height     = float(parts[11])  # peak intensity
                confidence = float(parts[13])  # confidence score
            except (ValueError, IndexError):
                continue

            # NMRPipe stores high ppm at index 0 (decreasing), but our simulation
            # stores low ppm at index 0 (increasing), so deep_picker's reported
            # ppm values are mirrored. Un-flip before mapping to pixel index.
            x_ppm_actual = ppm_h_axis[0] + ppm_h_axis[-1] - x_ppm   # 6+10 - reported
            y_ppm_actual = ppm_n_axis[0] + ppm_n_axis[-1] - y_ppm   # 105+135 - reported
            row = int(np.argmin(np.abs(ppm_h_axis - x_ppm_actual)))  # ¹H  → row
            col = int(np.argmin(np.abs(ppm_n_axis - y_ppm_actual)))  # ¹⁵N → col

            peaks.append({
                "x_ppm":      x_ppm,
                "y_ppm":      y_ppm,
                "row":        row,
                "col":        col,
                "height":     height,
                "confidence": confidence,
            })
    return peaks


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Load ppm axes so we can convert ppm → pixel index
    d          = np.load(NPZ_PATH, allow_pickle=True)
    ppm_h_axis = d["ppm_h_axis"]   # (256,)  ¹H  axis, 6–10 ppm
    ppm_n_axis = d["ppm_n_axis"]   # (32,)   ¹⁵N axis, 105–135 ppm
    n_spectra  = len(d["spectra"]) # 500

    all_rows = []   # accumulate every peak across all spectra

    for i in range(1, n_spectra + 1):
        if i % 50 == 0 or i == 1:
            print(f"  Processing spectrum {i}/{n_spectra} …")

        tab   = run_deep_picker(i)
        peaks = parse_tab(tab, ppm_h_axis, ppm_n_axis)

        for p in peaks:
            all_rows.append({
                "spectrum_id": i,
                "x_ppm":       p["x_ppm"],
                "y_ppm":       p["y_ppm"],
                "row":         p["row"],
                "col":         p["col"],
                "height":      p["height"],
                "confidence":  p["confidence"],
            })

    # Write CSV
    fieldnames = ["spectrum_id", "x_ppm", "y_ppm", "row", "col", "height", "confidence"]
    with open(CSV_OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    total_peaks = len(all_rows)
    avg_peaks   = total_peaks / n_spectra
    print(f"\nDone. {total_peaks} peaks across {n_spectra} spectra "
          f"(avg {avg_peaks:.1f} per spectrum)")
    print(f"Saved → {CSV_OUT}")


if __name__ == "__main__":
    main()
