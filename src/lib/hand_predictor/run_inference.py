"""
run_inference.py
----------------
Standalone runner for the fast_eval Parkinsonism hand predictor.
Run from this directory (src/lib/hand_predictor/).

Usage
-----
    python run_inference.py "C:/Users/imadelkhechen/Pictures/Camera Roll/video.mp4" --hand Left

Arguments
---------
    video       Path to your video file (mp4, MOV, avi, etc.)
    --hand      Left or Right (default: Left)
    --output    Output folder (default: <video folder>/output/)

Output
------
    UPDRS severity score 0-3
    Kinematic parameters: frequency, intensity, FI value
    Annotated video, STFT plot, merge plot
    Two JSON files: UPDRS prediction + kinematic params

Requirements (conda activate mediapipe)
    python=3.8, pytorch=1.13, mediapipe>=0.8, ffmpeg in PATH
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# ── Add this directory to sys.path so relative imports work ──────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# ── Patch 1: Replace os.system("cp ...") with shutil.copy2 (Windows fix) ────
_orig_os_system = os.system

def _win_safe_os_system(cmd: str) -> int:
    if cmd.strip().startswith("cp "):
        parts = cmd.strip().split()
        if len(parts) >= 3:
            try:
                shutil.copy2(parts[1], parts[2])
                return 0
            except Exception as e:
                print(f"  [copy fallback error] {e}")
                return 1
    return _orig_os_system(cmd)

os.system = _win_safe_os_system

# ── Patch 2: Numpy-safe JSON encoder ─────────────────────────────────────────
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

# ── Patch 3: DataLoader num_workers=0 on Windows ─────────────────────────────
import torch.utils.data as _torch_data
_OrigDataLoader = _torch_data.DataLoader

class _WinSafeDataLoader(_OrigDataLoader):
    def __init__(self, *args, **kwargs):
        if sys.platform == "win32" and kwargs.get("num_workers", 0) > 0:
            kwargs["num_workers"] = 0
        super().__init__(*args, **kwargs)

_torch_data.DataLoader = _WinSafeDataLoader

# ── Now safe to import pipeline ───────────────────────────────────────────────
import pandas as pd
from utils.hand.api import (
    ffmpeg4format,
    hand_parameters,
    model_pred_severity,
    mp_kpts_generator,
    mp_kpts_preprocessing,
)
from utils.hand.keypoints import mergePlot_PeakInteRaw, stft_plot
from utils.seed import set_seed


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fwd(path: str) -> str:
    """Convert Windows backslash path to forward slashes (ffmpeg-python compat)."""
    return str(path).replace("\\", "/")


def _check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


# ── Main inference function ───────────────────────────────────────────────────

def run_inference(video_path: str, hand_lr: str, output_dir: str) -> tuple[dict, dict]:
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    filename   = video_path.stem
    ext        = video_path.suffix.lstrip(".")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wkdir      = str(SCRIPT_DIR)

    print(f"\n{'='*62}")
    print(f"  Fast Eval Parkinsonism — Hand Predictor")
    print(f"  Video  : {video_path}")
    print(f"  Hand   : {hand_lr}  |  Task: Finger Tapping (UPDRS 3.4)")
    print(f"  Output : {output_dir}")
    print(f"{'='*62}\n")

    if not _check_ffmpeg():
        print("  WARNING: ffmpeg not found in PATH.")
        print("  Download from https://ffmpeg.org/download.html and add to PATH.\n")

    set_seed(42)

    # ── Step 1: Convert to mp4 ────────────────────────────────────────────────
    print("[1/6] Converting video to mp4 (60fps, 720x1280)...")
    video_mp4 = _fwd(output_dir / f"{filename}.mp4")
    try:
        ffmpeg4format(video_path=_fwd(video_path), output_path=video_mp4)
        print(f"      -> {video_mp4}")
    except Exception as e:
        print(f"      WARNING: ffmpeg failed ({e})")
        print(f"      Copying original file as mp4 fallback...")
        shutil.copy2(str(video_path), video_mp4.replace("/", os.sep))
        if not video_path.suffix.lower() == ".mp4":
            print("      NOTE: skipping re-encode; quality may differ.")

    # ── Step 2: MediaPipe keypoint extraction ─────────────────────────────────
    # Use the original video file — cv2 ignores rotation metadata and gives the
    # stored landscape frame orientation, on which the float16 hand landmarker
    # achieves ~87% detection vs ~8% on the ffmpeg-reencoded portrait frames.
    print("[2/6] Extracting hand keypoints (MediaPipe)...")
    mp_kpts_generator(
        video_path=_fwd(video_path),
        output_root_path=_fwd(output_dir),
        hand_query=hand_lr,
        export_video=True,
        logging=False,
    )
    print(f"      -> keypoints CSV written")

    # Re-encode annotated video to H264
    annot_mp4  = video_mp4.replace(".mp4", "_annot.mp4")
    annot_tmp  = video_mp4.replace(".mp4", "_annot_h264.mp4")
    try:
        ffmpeg4format(video_path=annot_mp4, output_path=annot_tmp)
        os.replace(annot_tmp.replace("/", os.sep), annot_mp4.replace("/", os.sep))
    except Exception:
        pass  # annotated video stays as-is

    # ── Step 3: Preprocess keypoints ──────────────────────────────────────────
    print("[3/6] Preprocessing keypoints (reaxis, normalise by thumb)...")
    csv_in = csv_out = None
    error_frame_ratio = 1.0
    for suffix in ["", ".thre0"]:
        candidate_in  = _fwd(output_dir / f"{filename}_mp_hand_kpt{suffix}.csv")
        candidate_out = _fwd(output_dir / f"{filename}_mp_hand_kpt_processed{suffix}.csv")
        try:
            error_frame_ratio = mp_kpts_preprocessing(candidate_in, candidate_out, logging=False)
            csv_in, csv_out = candidate_in, candidate_out
            break
        except Exception as e:
            if suffix == "":
                continue  # try thre0 fallback
            raise RuntimeError(f"Keypoint preprocessing failed: {e}") from e

    if csv_out is None:
        raise RuntimeError("No processable keypoint CSV found. Check that MediaPipe detected a hand.")

    print(f"      -> error frame ratio: {error_frame_ratio:.1%}", end="")
    if error_frame_ratio > 0.5:
        print("  (HIGH — prediction may be unreliable)")
    else:
        print()

    # ── Step 4: UPDRS severity prediction ─────────────────────────────────────
    print("[4/6] Running severity models (3 × HandConvNet_o binary classifiers)...")
    csv_basename = Path(csv_out).name
    map_path = _fwd(output_dir / f"{filename}_map.csv")
    pd.DataFrame([[csv_basename, 0, error_frame_ratio]]).to_csv(
        map_path.replace("/", os.sep), index=False, header=False
    )

    df_predict = model_pred_severity(
        wkdir_path=wkdir,
        test_data_path=_fwd(output_dir),
        test_map_path=map_path,
        hand_LR=hand_lr,
        hand_pos=1,
        random_rotat_3d=True,
        seed=42,
    )
    df_predict.drop(["label"], axis=1, inplace=True)
    updrs_json_path = str(output_dir / f"{filename}_UPDRS_prediction.json")
    df_predict.iloc[0].to_json(updrs_json_path, indent=2)
    os.remove(map_path.replace("/", os.sep))
    print(f"      -> prediction written")

    # ── Step 5: Kinematic parameters ──────────────────────────────────────────
    print("[5/6] Computing kinematic parameters (STFT analysis)...")
    data_input = pd.read_csv(csv_out.replace("/", os.sep))
    results = hand_parameters(data_input=data_input)
    handparams_json_path = str(output_dir / f"{filename}_handparams.json")
    with open(handparams_json_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"      -> kinematic params written")

    # ── Step 6: Plots ─────────────────────────────────────────────────────────
    print("[6/6] Generating STFT and merge plots...")
    stft_png  = str(output_dir / f"{filename}_stft.png")
    merge_png = str(output_dir / f"{filename}_merge.png")
    try:
        stft_plot(
            np.array(results["distance-thumb-ratio"]),
            png_filepath=stft_png,
        )
        mergePlot_PeakInteRaw(
            np.array(results["stft"]["time"]),
            np.array(results["distance-thumb-ratio"]),
            max_freq=np.array(results["stft"]["freq"]),
            max_intensity=np.array(results["stft"]["intensity"]),
            inte_ylim_max=0.5,
            png_filepath=merge_png,
        )
        print(f"      -> plots saved")
    except Exception as e:
        print(f"      WARNING: plot generation failed: {e}")

    # ── Report ────────────────────────────────────────────────────────────────
    with open(updrs_json_path) as f:
        updrs = json.load(f)

    overall = updrs.get("predict_overall")
    severity_map = {
        0: "Normal  — no finger tapping impairment",
        1: "Slight  — mild slowing or hesitation",
        2: "Mild    — moderate impairment",
        3: "Moderate/Severe — marked difficulty or inability",
    }

    print(f"\n{'='*62}")
    print("  RESULTS")
    print(f"{'='*62}")

    print(f"\n  UPDRS Score Breakdown:")
    print(f"    FG_1  (0 vs 1+)  : {updrs.get('predict_Left_FG_1', updrs.get('predict_Right_FG_1', 'N/A'))}")
    print(f"    FG_2  (1- vs 2+) : {updrs.get('predict_Left_FG_2', updrs.get('predict_Right_FG_2', 'N/A'))}")
    print(f"    FG_3  (2- vs 3+) : {updrs.get('predict_Left_FG_3', updrs.get('predict_Right_FG_3', 'N/A'))}")
    print(f"\n  >> Overall UPDRS 3.4 Score : {int(overall) if overall is not None else 'N/A'}")
    if overall is not None:
        print(f"     {severity_map.get(int(overall), '')}")

    print(f"\n  Kinematic Parameters:")
    print(f"    Frequency   mean   : {results['freq-mean']:.3f} Hz")
    print(f"    Frequency   median : {results['freq-median']:.3f} Hz")
    print(f"    Frequency   std    : {results['freq-std']:.3f} Hz")
    print(f"    Intensity   mean   : {results['intensity-mean']:.4f}")
    print(f"    Intensity   median : {results['intensity-median']:.4f}")
    print(f"    FI value    mean   : {results['inte-freq-mean']:.4f}  (freq × intensity)")
    print(f"    FI value    median : {results['inte-freq-median']:.4f}")
    if results.get("peaks-mean") is not None:
        print(f"    Peak amplitude     : {results['peaks-mean']:.4f}  (mean)")
        print(f"    Peak count         : {len(results['peaks']['time'])}")
    print(f"    Error frame ratio  : {error_frame_ratio:.1%}")

    print(f"\n  Output files:")
    print(f"    UPDRS prediction   : {updrs_json_path}")
    print(f"    Kinematic params   : {handparams_json_path}")
    print(f"    Annotated video    : {annot_mp4.replace('/', os.sep)}")
    print(f"    Merge plot         : {merge_png}")
    print(f"    STFT plot          : {stft_png}")
    print(f"{'='*62}\n")

    return updrs, results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parkinsonism Finger Tapping Inference (UPDRS 3.4)"
    )
    parser.add_argument(
        "video",
        help='Path to video file. Example: "C:/Users/.../Pictures/Camera Roll/tap.mp4"',
    )
    parser.add_argument(
        "--hand",
        choices=["Left", "Right"],
        default="Left",
        help="Which hand is visible in the video (default: Left)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output folder (default: <video_folder>/output/)",
    )
    args = parser.parse_args()

    out = args.output
    if out is None:
        out = str(Path(args.video).resolve().parent / "output")

    run_inference(args.video, args.hand, out)
