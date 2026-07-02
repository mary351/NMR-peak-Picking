# import nmrglue as ng
# import numpy as np
# from pathlib import Path
# from simulating_data import HSQCGenerator

# def export_test_set_to_ft2(
#     spectra,        # (N, H, W) float32
#     ppm_h_axis,     # (H,)
#     ppm_n_axis,     # (W,)
#     out_dir="test_ft2",
#     obs_h=600.0,    # simulated spectrometer frequency MHz (¹H)
#     obs_n=60.9,     # simulated spectrometer frequency MHz (¹⁵N)
# ):
#     """
#     Convert numpy test spectra to NMRPipe .ft2 format for DeepPicker.
#     Call this once; give the student the resulting folder.
#     """
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     sw_h = float(abs(ppm_h_axis[-1] - ppm_h_axis[0])) * obs_h   # Hz
#     sw_n = float(abs(ppm_n_axis[-1] - ppm_n_axis[0])) * obs_n   # Hz
#     car_h = float(ppm_h_axis.mean()) * obs_h                      # Hz
#     car_n = float(ppm_n_axis.mean()) * obs_n                      # Hz

#     for i, sp in enumerate(spectra):
#         dic = ng.pipe.create_empty_dic()

#         # Dimension sizes
#         dic['FDDIMCOUNT']  = 2
#         dic['FDF2TDSIZE']  = sp.shape[1]   # ¹H  (direct)
#         dic['FDF1TDSIZE']  = sp.shape[0]   # ¹⁵N (indirect)
#         dic['FDF2APOD']    = sp.shape[1]
#         dic['FDF1APOD']    = sp.shape[0]

#         # Spectral parameters ¹H (F2, direct dimension)
#         dic['FDF2SW']      = sw_h
#         dic['FDF2OBS']     = obs_h
#         dic['FDF2CAR']     = car_h / obs_h   # carrier in ppm
#         dic['FDF2LABEL']   = 'HN'

#         # Spectral parameters ¹⁵N (F1, indirect dimension)
#         dic['FDF1SW']      = sw_n
#         dic['FDF1OBS']     = obs_n
#         dic['FDF1CAR']     = car_n / obs_n   # carrier in ppm
#         dic['FDF1LABEL']   = 'N'

#         # Flags expected by NMRPipe readers
#         dic['FDF2QUADFLAG'] = 1   # real data
#         dic['FDF1QUADFLAG'] = 1
#         dic['FDSPECNUM']    = sp.shape[0]
#         dic['FDSLICECOUNT'] = sp.shape[0]
#         dic['FDSIZE']       = sp.shape[1]

#         path = out_dir / f"spectrum_{i+1:04d}.ft2"
#         ng.pipe.write(str(path), dic, sp.astype(np.float32))

#     print(f"Exported {len(spectra)} spectra → {out_dir}/")
#     return out_dir


# # Load your saved .npz
# spectra, peak_lists, ppm_lists, meta = HSQCGenerator.load_dataset("hsqc_train.npz")

# # Convert to .ft2
# export_test_set_to_ft2(
#     spectra    = spectra,
#     ppm_h_axis = meta["ppm_h_axis"],
#     ppm_n_axis = meta["ppm_n_axis"],
#     out_dir    = "train_ft2",
# )

"""
export_to_ft2_fixed.py

Fixed .ft2 export using nmrglue's pipe.create_empty_dic() with all required
NMRPipe header fields that DEEP Picker needs to read dimensions correctly.

The segfault "ndata_frq*ydim is 0" means DEEP Picker read ydim=0 from the
header — the previous version was missing FDSPECNUM / FDSLICECOUNT which is
what NMRPipe uses to store the indirect dimension size.
"""

import numpy as np
import nmrglue as ng
from pathlib import Path


def numpy_to_ft2(
    spectrum,       # (H, W) float32  — H = ¹H rows, W = ¹⁵N cols
    ppm_h_axis,     # (H,) float32    — ¹H  ppm axis
    ppm_n_axis,     # (W,) float32    — ¹⁵N ppm axis
    out_path,
    obs_h=600.0,    # spectrometer frequency ¹H  (MHz)
    obs_n=60.9,    # spectrometer frequency ¹⁵N (MHz)  — 600 MHz * 0.101268
):
    """
    Write one (H, W) numpy spectrum to a valid NMRPipe .ft2 file.

    Key fields DEEP Picker requires:
        FDSPECNUM   = number of rows (¹⁵N points, indirect dim)
        FDF2TDSIZE  = number of cols (¹H  points, direct   dim)
        FDF1TDSIZE  = number of rows (¹⁵N points, indirect dim)
        FDF2SW, FDF1SW  = spectral widths in Hz
        FDF2OBS, FDF1OBS = observe frequencies in MHz
    """
    H, W = spectrum.shape
    # Your convention:  rows (H=256) = ¹H,  cols (W=32) = ¹⁵N
    # NMRPipe convention: F2 (direct, fast) = ¹H, F1 (indirect, slow) = ¹⁵N
    # So: F2 size = H = 256 (¹H),  F1 size = W = 32 (¹⁵N)
    # The spectrum must be transposed for NMRPipe: shape (W, H) = (32, 256)
    # because NMRPipe stores rows as ¹⁵N slices, each row being ¹H points

    sw_h_ppm  = float(abs(ppm_h_axis[-1] - ppm_h_axis[0]))
    sw_n_ppm  = float(abs(ppm_n_axis[-1] - ppm_n_axis[0]))
    sw_h_hz   = sw_h_ppm * obs_h
    sw_n_hz   = sw_n_ppm * obs_n

    car_h_ppm = float(ppm_h_axis.mean())
    car_n_ppm = float(ppm_n_axis.mean())

    dic = ng.pipe.create_empty_dic()

    # ── Core dimension sizes ───────────────────────────────────────────────────
    # NMRPipe 2D: FDSIZE = F2 points (¹H = H = 256)
    #             FDSPECNUM = number of F1 slices (¹⁵N = W = 32)
    dic['FDSIZE']       = float(H)   # ¹H  points (direct,   F2)
    dic['FDSPECNUM']    = float(W)   # ¹⁵N slices (indirect, F1)
    dic['FDSLICECOUNT'] = float(W)
    dic['FDDIMCOUNT']   = 2.0

    # ── Direct dimension F2 (¹H, 256 points) ──────────────────────────────────
    dic['FDF2TDSIZE']   = float(H)
    dic['FDF2APOD']     = float(H)
    dic['FDF2FTSIZE']   = float(H)
    dic['FDF2SW']       = sw_h_hz
    dic['FDF2OBS']      = obs_h
    dic['FDF2CAR']      = car_h_ppm * obs_h
    dic['FDF2ORIG']     = (car_h_ppm - sw_h_ppm / 2.0) * obs_h
    dic['FDF2QUADFLAG'] = 1.0
    dic['FDF2LABEL']    = 'HN'
    dic['FDF2UNITS']    = 1.0

    # ── Indirect dimension F1 (¹⁵N, 32 points) ────────────────────────────────
    dic['FDF1TDSIZE']   = float(W)
    dic['FDF1APOD']     = float(W)
    dic['FDF1FTSIZE']   = float(W)
    dic['FDF1SW']       = sw_n_hz
    dic['FDF1OBS']      = obs_n
    dic['FDF1CAR']      = car_n_ppm * obs_n
    dic['FDF1ORIG']     = (car_n_ppm - sw_n_ppm / 2.0) * obs_n
    dic['FDF1QUADFLAG'] = 1.0
    dic['FDF1LABEL']    = 'N'
    dic['FDF1UNITS']    = 1.0

    # ── Flags ──────────────────────────────────────────────────────────────────
    dic['FDF2FTFLAG']   = 1.0   # direct dim is frequency domain (post-FFT)
    dic['FDF1FTFLAG']   = 1.0   # indirect dim is frequency domain (post-FFT)
    dic['FD2DPHASE']    = 0.0
    dic['FDTRANSPOSED'] = 0.0

    # NMRPipe 2D layout: (n_slices, n_direct) = (W, H) = (32, 256)
    # Your array is (H, W) = (256, 32) → transpose before writing
    ng.pipe.write(str(out_path), dic, spectrum.T.astype(np.float32), overwrite=True)
    return out_path


def verify_ft2(ft2_path):
    """
    Read back a .ft2 file and print key header fields.
    Use this to confirm DEEP Picker will see the right dimensions.
    """
    dic, data = ng.pipe.read(str(ft2_path))
    print(f"\n── Verifying {ft2_path} ──")
    print(f"  data.shape     : {data.shape}")
    print(f"  FDSPECNUM      : {dic['FDSPECNUM']}")
    print(f"  FDF2TDSIZE (W) : {dic['FDF2TDSIZE']}")
    print(f"  FDF1TDSIZE (H) : {dic['FDF1TDSIZE']}")
    print(f"  FDF2SW  (¹H Hz): {dic['FDF2SW']:.1f}")
    print(f"  FDF1SW  (¹⁵N Hz): {dic['FDF1SW']:.1f}")
    print(f"  FDF2OBS (MHz)  : {dic['FDF2OBS']:.1f}")
    print(f"  FDF1OBS (MHz)  : {dic['FDF1OBS']:.1f}")
    print("data.shape[1]",  data.shape[1])
    print("data shape", data.shape)
    if data.shape[0] > 0 and data.shape[1] > 0:
        print(f"  data range     : [{data.min():.4f}, {data.max():.4f}]")
        print("  ✓ Header looks correct for DEEP Picker")
    else:
        print("  ✗ WARNING: zero dimension detected — header still wrong")
    return dic, data


def export_test_set_to_ft2(
    spectra,        # (N, H, W) float32
    ppm_h_axis,     # (H,)
    ppm_n_axis,     # (W,)
    out_dir="test_ft2",
    obs_h=600.0,
    obs_n=60.77,
    verify=True,    # verify the first file after writing
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N = len(spectra)
    print(f"Exporting {N} spectra to {out_dir}/ ...")

    for i, sp in enumerate(spectra):
        out_path = out_dir / f"spectrum_{i+1:04d}.ft2"
        numpy_to_ft2(sp, ppm_h_axis, ppm_n_axis, out_path, obs_h, obs_n)
        if (i+1) % max(1, N//10) == 0 or i == 0 or i == N-1:
            print(f"  [{i+1:>{len(str(N))}}/{N}]  {out_path.name}")

    # Verify the first file so you can catch header issues before running
    # DEEP Picker on all spectra
    if verify:
        verify_ft2(out_dir / "spectrum_0001.ft2")

    print(f"\nDone. Run DEEP Picker with:")
    print(f"  ./deep_picker -in {out_dir}/spectrum_0001.ft2 "
          f"-out {out_dir}/spectrum_0001_peaks.tab -model 2")
    return out_dir


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from simulating_data import HSQCGenerator

    print("Loading test spectra ...")
    spectra, peak_lists, _, meta = HSQCGenerator.load_dataset("/Users/meri_abg/Desktop/High_School_Student_Visit/hsqc_train.npz")

    export_test_set_to_ft2(
        spectra    = spectra,          # test on first 5 only
        ppm_h_axis = meta["ppm_h_axis"],
        ppm_n_axis = meta["ppm_n_axis"],
        out_dir    = "train_ft2_F",
        verify     = True,
    )