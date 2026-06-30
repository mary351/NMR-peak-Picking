import numpy as np
import nmrglue as ng
import os 
import matplotlib.pyplot as plt

class RealDataLoader:
    """
    Reads a single NMRPipe .ft2 spectrum and its companion .tab peak list,
    tiles the spectrum into overlapping patches, and returns them in the
    same (H, W) format used throughout the rest of the pipeline.
 
    Typical use
    -----------
    loader = RealDataLoader("asyn_a.ft2", "asyn_a_peaks.tab")
 
    # Inspect the full spectrum
    spectrum, ppm_h, ppm_n, peaks_ppm, peaks_pts = loader.load_full_spectrum()
    loader.print_summary()
    loader.plot_full_spectrum("asyn_a_overview.png")
 
    # Tile into patches for training / evaluation
    patches, patch_peaks, patch_origins = loader.tile(patch_h=256, patch_w=32,
                                                       stride_h=128, stride_w=16)
    """
 
    def __init__(self, ft2_path, tab_path=None):
        """
        Parameters
        ----------
        ft2_path : str   — path to the NMRPipe .ft2 spectrum file
        tab_path : str or None — path to the companion .tab peak list;
                                 if None, no ground-truth labels are loaded
        """
        self.ft2_path = ft2_path
        self.tab_path = tab_path
        self._spectrum  = None   # (H, W) float32  — cached after first load
        self._ppm_h     = None   # (H,)  ¹H  ppm axis
        self._ppm_n     = None   # (W,)  ¹⁵N ppm axis
        self._peaks_ppm = None   # list of (ppm_h, ppm_n)
        self._peaks_pts = None   # list of (row, col) in the full matrix
 
    # ----------------------------------------------------------------
    # .ft2 reader
    # ----------------------------------------------------------------
 
    def _read_ft2(self):
        """
        Read the NMRPipe binary spectrum using nmrglue.
 
        NMRPipe stores the indirect dimension (¹⁵N / F1) as rows and the
        direct dimension (¹H / F2) as columns: shape = (F1_pts, F2_pts).
        We transpose to (F2_pts, F1_pts) = (H, W) so that H indexes ¹H
        (rows = slow axis) and W indexes ¹⁵N (cols = fast axis), matching
        our pipeline convention and the ARTINA default (256×32).
        """
        dic, raw = ng.pipe.read(self.ft2_path)
 
        # raw.shape == (n_indirect, n_direct) == (¹⁵N pts, ¹H pts)
        # Transpose → (¹H pts, ¹⁵N pts) = (H, W)
        data = raw.T.astype(np.float32)      # (H, W)
 
        # Build unit-conversion objects to get ppm axes.
        # dim=0 in the original (pre-transpose) array is the indirect (¹⁵N) axis.
        # dim=1 in the original array is the direct  (¹H)  axis.
        uc_f1 = ng.pipe.make_uc(dic, raw, dim=0)   # ¹⁵N (indirect)
        uc_f2 = ng.pipe.make_uc(dic, raw, dim=1)   # ¹H  (direct)
 
        H, W = data.shape
        # ppm_h: ¹H  ppm for each row  (H values)
        ppm_h = np.array([uc_f2.ppm(i) for i in range(H)], dtype=np.float32)
        # ppm_n: ¹⁵N ppm for each col  (W values)
        ppm_n = np.array([uc_f1.ppm(i) for i in range(W)], dtype=np.float32)
 
        return data, ppm_h, ppm_n, dic
 
    # ----------------------------------------------------------------
    # .tab reader
    # ----------------------------------------------------------------
 
    @staticmethod
    def read_tab(tab_path):
        """
        Parse an NMRPipe .tab peak list file.
 
        The file looks like:
            REMARK ...
            VARS   INDEX X_AXIS Y_AXIS DX DY X_PPM Y_PPM X_HZ Y_HZ ...
            FORMAT %5d   %9.3f  ...
            DATA   FIRST_POINT 1
            <blank line>
              1   432.123  87.456  ...  8.432  115.23  ...
 
        Returns
        -------
        peaks : list of dicts, one per peak.
                Keys correspond to the VARS header columns, e.g.:
                  'INDEX', 'X_AXIS', 'Y_AXIS', 'X_PPM', 'Y_PPM',
                  'X_HZ', 'Y_HZ', 'XW', 'YW', 'XW_HZ', 'YW_HZ',
                  'X1', 'X3', 'Y1', 'Y3', 'HEIGHT', 'DHEIGHT',
                  'VOL', 'PCHI2', 'TYPE', 'ASS', 'CLUSTID', 'MEMCNT'
        col_names : list of str — the VARS column names in order
        """
        peaks     = []
        col_names = []
 
        with open(tab_path, "r") as fh:
            for line in fh:
                line = line.rstrip("\n")
 
                # Skip blank lines and REMARK / comment lines
                if not line.strip() or line.strip().startswith("#"):
                    continue
 
                # VARS line defines the column names
                if line.startswith("VARS"):
                    col_names = line.split()[1:]   # drop the "VARS" keyword
                    continue
 
                # FORMAT and DATA lines are metadata — skip
                if line.startswith("FORMAT") or line.startswith("DATA"):
                    continue
 
                # Everything else is a data row
                if not col_names:
                    continue   # haven't seen VARS yet — skip
 
                tokens = line.split()
                if len(tokens) < len(col_names):
                    continue   # malformed line
 
                row = {}
                for name, tok in zip(col_names, tokens):
                    # Try int, then float, then keep as string
                    try:
                        row[name] = int(tok)
                    except ValueError:
                        try:
                            row[name] = float(tok)
                        except ValueError:
                            row[name] = tok
                peaks.append(row)
 
        return peaks, col_names
 
    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
 
    def load_full_spectrum(self):
        """
        Load (and cache) the spectrum and peak list.
 
        Returns
        -------
        spectrum   : ndarray (H, W) float32
        ppm_h      : ndarray (H,)   ¹H  ppm per row
        ppm_n      : ndarray (W,)   ¹⁵N ppm per col
        peaks_ppm  : list of (ppm_h, ppm_n) tuples — peak positions in ppm
                     (empty list if no tab_path given)
        peaks_pts  : list of (row, col) tuples — peak positions in matrix points
                     (empty list if no tab_path given)
        """
        if self._spectrum is None:
            sp, ph, pn, dic = self._read_ft2()
            # Clip negative values (common in NMR after baseline correction)
            sp = np.clip(sp, 0, None)
            # Normalise to [0, ~1] by 99th percentile to avoid outlier distortion
            p99 = np.percentile(sp, 99)
            if p99 > 0:
                sp = sp / p99
            self._spectrum = sp
            self._ppm_h    = ph
            self._ppm_n    = pn
            self._dic      = dic
 
        if self._peaks_ppm is None:
            if self.tab_path and os.path.isfile(self.tab_path):
                raw_peaks, _ = self.read_tab(self.tab_path)
                H, W = self._spectrum.shape
 
                peaks_ppm, peaks_pts = [], []
                for pk in raw_peaks:
                    # X_PPM = ¹H ppm, Y_PPM = ¹⁵N ppm
                    ph_val = float(pk.get("X_PPM", pk.get("X_ppm", 0)))
                    pn_val = float(pk.get("Y_PPM", pk.get("Y_ppm", 0)))
 
                    # Convert ppm → nearest row/col using precomputed ppm axes
                    row = int(np.argmin(np.abs(self._ppm_h - ph_val)))
                    col = int(np.argmin(np.abs(self._ppm_n - pn_val)))
 
                    # Sanity check: skip peaks outside the matrix
                    if 0 <= row < H and 0 <= col < W:
                        peaks_ppm.append((ph_val, pn_val))
                        peaks_pts.append((row, col))
 
                self._peaks_ppm = peaks_ppm
                self._peaks_pts = peaks_pts
            else:
                self._peaks_ppm = []
                self._peaks_pts = []
 
        return (self._spectrum, self._ppm_h, self._ppm_n,
                self._peaks_ppm, self._peaks_pts)
 
    def print_summary(self):
        """Print a human-readable summary of the loaded spectrum and peak list."""
        sp, ph, pn, pk_ppm, pk_pts = self.load_full_spectrum()
        H, W = sp.shape
        print(f"\n{'═'*60}")
        print(f"  File     : {self.ft2_path}")
        print(f"  Shape    : {H} × {W}  (¹H pts × ¹⁵N pts)")
        print(f"  ¹H  ppm  : {ph.min():.2f} – {ph.max():.2f} ppm")
        print(f"  ¹⁵N ppm  : {pn.min():.2f} – {pn.max():.2f} ppm")
        print(f"  Intensity: min={sp.min():.4f}  max={sp.max():.4f}  "
              f"mean={sp.mean():.4f}")
        if self.tab_path:
            print(f"  Peak list: {self.tab_path}")
            print(f"  Peaks    : {len(pk_pts)} peaks")
            if pk_ppm:
                ph_vals = [p[0] for p in pk_ppm]
                pn_vals = [p[1] for p in pk_ppm]
                print(f"             ¹H  range: {min(ph_vals):.2f} – "
                      f"{max(ph_vals):.2f} ppm")
                print(f"             ¹⁵N range: {min(pn_vals):.2f} – "
                      f"{max(pn_vals):.2f} ppm")
        print(f"{'═'*60}\n")
 
    def plot_full_spectrum(self, path=None, n_contours=12,
                           contour_base_factor=0.04):
        """
        Plot the full 2-D spectrum with peaks overlaid (contour plot style,
        as NMR spectroscopists are used to seeing).
 
        Parameters
        ----------
        path                : str or None — save path; None = return figure only
        n_contours          : int  — number of positive contour levels
        contour_base_factor : float — first contour at this fraction of the max
        """
        sp, ph, pn, pk_ppm, pk_pts = self.load_full_spectrum()
 
        # Contour levels: geometric series starting at base_factor × max
        base  = float(sp.max()) * contour_base_factor
        levels = [base * (1.3 ** i) for i in range(n_contours)]
 
        fig, ax = plt.subplots(figsize=(10, 8))
 
        # ppm extents for imshow: [left, right, bottom, top]
        # ¹H ppm increases right to left in NMR convention
        extent = [pn[0], pn[-1], ph[0], ph[-1]]
 
        # Contour plot (classic NMR display)
        ax.contour(pn, ph, sp, levels=levels, colors=["#2176AE"],
                   linewidths=0.6, alpha=0.8)
 
        # Peak positions
        if pk_ppm:
            h_vals = [p[0] for p in pk_ppm]
            n_vals = [p[1] for p in pk_ppm]
            ax.scatter(n_vals, h_vals, s=40, marker="x",
                       c="#E63946", linewidths=1.5,
                       zorder=5, label=f"Peaks (n={len(pk_ppm)})")
            ax.legend(fontsize=9)
 
        ax.set_xlabel("¹⁵N chemical shift (ppm)", fontsize=11)
        ax.set_ylabel("¹H chemical shift (ppm)",  fontsize=11)
        ax.set_title(f"2-D ¹H–¹⁵N HSQC:  {os.path.basename(self.ft2_path)}",
                     fontsize=12)
        # NMR convention: both axes increase right-to-left and top-to-bottom
        ax.invert_xaxis()
        ax.invert_yaxis()
        fig.tight_layout()
 
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  → Saved {path}")
        return fig
 
    def tile(self, patch_h=256, patch_w=32,
             stride_h=None, stride_w=None):
        """
        Tile the full spectrum into overlapping (patch_h × patch_w) patches.
 
        This is necessary because the real spectra are much larger than 256×32
        (asyn_a.ft2 is ~2048×512) while the models expect fixed-size inputs.
        Overlapping tiles ensure peaks near patch boundaries are seen in full
        context from at least one tile.
 
        Parameters
        ----------
        patch_h  : int — patch height in ¹H  points  (default 256)
        patch_w  : int — patch width  in ¹⁵N points  (default 32)
        stride_h : int — stride along ¹H  axis (default patch_h // 2)
        stride_w : int — stride along ¹⁵N axis (default patch_w // 2)
 
        Returns
        -------
        patches       : ndarray (N_tiles, patch_h, patch_w) float32
        patch_peaks   : list[list[(row, col)]] — peaks in each tile's local coords
        patch_origins : list[(r0, c0)] — top-left corner of each tile in full matrix
        patch_ppm_h   : list[ndarray (patch_h,)] — ¹H  ppm axes for each tile
        patch_ppm_n   : list[ndarray (patch_w,)] — ¹⁵N ppm axes for each tile
        """
        sp, ph, pn, _, pk_pts = self.load_full_spectrum()
        H, W = sp.shape
 
        stride_h = stride_h or patch_h // 2
        stride_w = stride_w or patch_w // 2
 
        patches, patch_peaks, patch_origins = [], [], []
        patch_ppm_h_list, patch_ppm_n_list  = [], []
 
        row_starts = list(range(0, max(1, H - patch_h + 1), stride_h))
        col_starts = list(range(0, max(1, W - patch_w + 1), stride_w))
 
        # Ensure we always have a tile that covers the bottom-right corner
        if H > patch_h and row_starts[-1] + patch_h < H:
            row_starts.append(H - patch_h)
        if W > patch_w and col_starts[-1] + patch_w < W:
            col_starts.append(W - patch_w)
 
        for r0 in row_starts:
            r1 = min(r0 + patch_h, H)
            for c0 in col_starts:
                c1 = min(c0 + patch_w, W)
 
                # Extract tile (zero-pad if near boundary)
                tile = np.zeros((patch_h, patch_w), dtype=np.float32)
                tile[:r1-r0, :c1-c0] = sp[r0:r1, c0:c1]
 
                # Find peaks inside this tile; convert to tile-local coords
                local_peaks = []
                for (ry, rx) in pk_pts:
                    if r0 <= ry < r1 and c0 <= rx < c1:
                        local_peaks.append((ry - r0, rx - c0))
 
                patches.append(tile)
                patch_peaks.append(local_peaks)
                patch_origins.append((r0, c0))
                patch_ppm_h_list.append(ph[r0:r1])
                patch_ppm_n_list.append(pn[c0:c1])
 
        print(f"  Tiled {os.path.basename(self.ft2_path)}: "
              f"{H}×{W} → {len(patches)} patches of {patch_h}×{patch_w} "
              f"(stride {stride_h}×{stride_w})")
 
        return (np.array(patches, dtype=np.float32),
                patch_peaks, patch_origins,
                patch_ppm_h_list, patch_ppm_n_list)
 
    def stitch_predictions(self, patch_heatmaps, patch_origins,
                            full_h, full_w, patch_h, patch_w):
        """
        Reassemble tiled heatmap predictions into a full-spectrum heatmap
        by averaging overlapping regions (overlap-add with count normalisation).
 
        Parameters
        ----------
        patch_heatmaps : list of ndarray (patch_h, patch_w) — model outputs
        patch_origins  : list of (r0, c0) — top-left corners from tile()
        full_h, full_w : int — dimensions of the original full spectrum
        patch_h, patch_w : int — tile dimensions used during tiling
 
        Returns
        -------
        full_heatmap : ndarray (full_h, full_w) float32
        """
        accum = np.zeros((full_h, full_w), dtype=np.float64)
        count = np.zeros((full_h, full_w), dtype=np.float64)
 
        for hm, (r0, c0) in zip(patch_heatmaps, patch_origins):
            r1 = min(r0 + patch_h, full_h)
            c1 = min(c0 + patch_w, full_w)
            accum[r0:r1, c0:c1] += hm[:r1-r0, :c1-c0]
            count[r0:r1, c0:c1] += 1.0
 
        count = np.maximum(count, 1.0)   # avoid division by zero
        return (accum / count).astype(np.float32)
 
 
def load_real_dataset(ft2_paths, tab_paths=None,
                      patch_h=256, patch_w=32,
                      stride_h=128, stride_w=16,
                      min_peaks_per_patch=1):
    """
    Load one or more real .ft2 spectra, tile them, and return a flat list
    of patches in the same format as HSQCGenerator.generate_dataset().
 
    Parameters
    ----------
    ft2_paths  : list of str — .ft2 file paths
    tab_paths  : list of str or None — .tab file paths (parallel to ft2_paths)
    patch_h    : int — tile height  (should match model's expected H)
    patch_w    : int — tile width   (should match model's expected W)
    stride_h   : int — stride along ¹H  axis
    stride_w   : int — stride along ¹⁵N axis
    min_peaks_per_patch : int — discard tiles with fewer than this many peaks
                                (avoids training on almost entirely empty tiles)
 
    Returns
    -------
    spectra    : ndarray (N_tiles, patch_h, patch_w)
    peak_lists : list[list[(row, col)]]
    ppm_lists  : list[list[(ppm_h, ppm_n)]]   — approximate for tile centres
    loaders    : list[RealDataLoader]          — one per ft2 file, for stitching
    """
    if tab_paths is None:
        tab_paths = [None] * len(ft2_paths)
 
    all_spectra, all_peaks, all_ppms = [], [], []
    all_loaders = []
 
    for ft2_path, tab_path in zip(ft2_paths, tab_paths):
        loader = RealDataLoader(ft2_path, tab_path)
        loader.print_summary()
 
        patches, p_peaks, p_origins, ph_list, pn_list = loader.tile(
            patch_h=patch_h, patch_w=patch_w,
            stride_h=stride_h, stride_w=stride_w,
        )
 
        # Build approximate ppm coords from tile ppm axes + local peak positions
        for tile, pks, ph_ax, pn_ax in zip(patches, p_peaks, ph_list, pn_list):
            if tab_path and len(pks) < min_peaks_per_patch:
                continue   # skip nearly-empty tiles during supervised training
            ppm_pairs = []
            for (ry, rx) in pks:
                ph_v = float(ph_ax[min(ry, len(ph_ax)-1)]) if len(ph_ax) > 0 else 0.0
                pn_v = float(pn_ax[min(rx, len(pn_ax)-1)]) if len(pn_ax) > 0 else 0.0
                ppm_pairs.append((ph_v, pn_v))
            all_spectra.append(tile)
            all_peaks.append(pks)
            all_ppms.append(ppm_pairs)
 
        all_loaders.append(loader)
 
    print(f"\n  Total tiles (with ≥{min_peaks_per_patch} peak): "
          f"{len(all_spectra)} from {len(ft2_paths)} spectrum/spectra")
 
    return np.array(all_spectra, dtype=np.float32), all_peaks, all_ppms, all_loaders