"""
api.py — Flask REST API for the Parkinsonism hand predictor.

Run with the fp39 Python:
    C:/Users/imadelkhechen/miniconda3/envs/fp39/python.exe api.py

Endpoints:
    GET  /health            -> {"status": "ok"}
    POST /analyze           -> multipart/form-data, field "video"
    POST /analyze_keypoints -> JSON body with pre-extracted keypoints
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
_HAND_PREDICTOR_DIR = _HERE / "src" / "lib" / "hand_predictor"
_WKDIR = str(_HAND_PREDICTOR_DIR).replace("\\", "/")

if str(_HAND_PREDICTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_HAND_PREDICTOR_DIR))

import torch.utils.data as _torch_data

_OrigDataLoader = _torch_data.DataLoader


class _WinSafeDataLoader(_OrigDataLoader):
    def __init__(self, *args, **kwargs):
        if sys.platform == "win32" and kwargs.get("num_workers", 0) > 0:
            kwargs["num_workers"] = 0
        super().__init__(*args, **kwargs)


_torch_data.DataLoader = _WinSafeDataLoader

import json
import os
import re
import subprocess
import tempfile

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from scipy.interpolate import interp1d
from werkzeug.utils import secure_filename

from utils.hand.api import hand_parameters, model_pred_severity, mp_kpts_preprocessing
from utils.seed import set_seed

_FP39_PYTHON = Path(r"C:\Users\imadelkhechen\miniconda3\envs\fp39\python.exe")
_SCRIPT = _HERE / "src" / "lib" / "hand_predictor" / "run_inference.py"

_SEVERITY = {0: "Normal", 1: "Slight", 2: "Mild", 3: "Moderate/Severe"}

_KPT_COLS = (
    ["timestamp"]
    + [f"x_{i}" for i in range(21)]
    + [f"y_{i}" for i in range(21)]
    + [f"z_{i}" for i in range(21)]
)

app = Flask(__name__)


def _fwd(p) -> str:
    return str(p).replace("\\", "/")


def _upsample(df: pd.DataFrame, target: int = 200) -> pd.DataFrame:
    if len(df) >= target:
        return df
    original_len = len(df)
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, target)
    new_data = {}
    for col in df.columns:
        f = interp1d(x_old, df[col].values, kind='linear')
        new_data[col] = f(x_new)
    print(f"Upsampled from {original_len} to {target} rows")
    return pd.DataFrame(new_data)


def _safe(val, decimals: int = 2) -> float:
    try:
        v = float(val)
        if v != v or abs(v) == float("inf"):   # NaN / Inf check
            return 0.0
        return round(v, decimals)
    except (TypeError, ValueError):
        return 0.0


def _build_response(updrs_df: pd.DataFrame, results: dict, error_frame_ratio: float) -> dict:
    overall = int(updrs_df["predict_overall"].iloc[0])
    peaks_count = len((results.get("peaks") or {}).get("time") or [])
    return {
        "updrs_score": overall,
        "severity": _SEVERITY.get(overall, "Unknown"),
        "frequency": _safe(results.get("freq-mean", 0), 2),
        "intensity": _safe(results.get("intensity-mean", 0), 3),
        "fi_value": _safe(results.get("inte-freq-mean", 0), 2),
        "peaks": peaks_count,
        "detection_rate": _safe((1.0 - error_frame_ratio) * 100, 1),
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/analyze")
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file in request (field name: 'video')"}), 400

    video_file = request.files["video"]
    hand = request.form.get("hand", "Left")
    if hand not in ("Left", "Right"):
        return jsonify({"error": "Field 'hand' must be 'Left' or 'Right'"}), 400

    filename = secure_filename(video_file.filename or "upload.mp4") or "upload.mp4"

    with tempfile.TemporaryDirectory(prefix="parkinson_video_") as tmpdir:
        tmp = Path(tmpdir)
        video_path = tmp / filename
        video_file.save(str(video_path))
        output_dir = tmp / "output"

        proc = subprocess.run(
            [str(_FP39_PYTHON), str(_SCRIPT), str(video_path),
             "--hand", hand, "--output", str(output_dir)],
            capture_output=True, text=True, timeout=300,
        )

        if proc.returncode != 0:
            return jsonify({"error": "Inference failed", "stderr": proc.stderr[-2000:]}), 500

        error_frame_ratio = 0.0
        m = re.search(r"error frame ratio:\s*([\d.]+)%", proc.stdout)
        if m:
            error_frame_ratio = float(m.group(1)) / 100.0

        stem = video_path.stem
        updrs_path = output_dir / f"{stem}_UPDRS_prediction.json"
        handparams_path = output_dir / f"{stem}_handparams.json"

        try:
            with open(updrs_path) as f:
                updrs_raw = json.load(f)
            with open(handparams_path) as f:
                handparams = json.load(f)
        except FileNotFoundError as exc:
            return jsonify({"error": f"Output file not found: {exc.filename}",
                            "stdout": proc.stdout[-2000:]}), 500

    overall = int(updrs_raw.get("predict_overall", 0))
    peaks_count = len((handparams.get("peaks") or {}).get("time") or [])

    return jsonify({
        "updrs_score": overall,
        "severity": _SEVERITY.get(overall, "Unknown"),
        "frequency": round(float(handparams.get("freq-mean", 0)), 2),
        "intensity": round(float(handparams.get("intensity-mean", 0)), 3),
        "fi_value": round(float(handparams.get("inte-freq-mean", 0)), 2),
        "peaks": peaks_count,
        "detection_rate": round((1.0 - error_frame_ratio) * 100, 1),
    })


@app.post("/analyze_keypoints")
def analyze_keypoints():
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "JSON body required"}), 400

    hand = body.get("hand", "Left")
    frames = body.get("frames")

    if hand not in ("Left", "Right"):
        return jsonify({"error": "'hand' must be 'Left' or 'Right'"}), 400
    if not frames or not isinstance(frames, list):
        return jsonify({"error": "'frames' must be a non-empty array"}), 400

    rows = []
    for i, frame in enumerate(frames):
        row = {col: float(frame.get(col, 0.0)) for col in _KPT_COLS}
        row["timestamp"] = i
        rows.append(row)

    df_in = pd.DataFrame(rows, columns=_KPT_COLS)

    with tempfile.TemporaryDirectory(prefix="parkinson_kpt_") as tmpdir:
        tmp = Path(tmpdir)
        csv_in  = tmp / "hand_kpt.csv"
        csv_out = tmp / "hand_kpt_processed.csv"
        map_csv = tmp / "map.csv"

        df_in.to_csv(str(csv_in), index=False)

        # Step 3: preprocessing
        try:
            error_frame_ratio = mp_kpts_preprocessing(
                _fwd(csv_in), _fwd(csv_out), logging=False
            )
        except Exception as exc:
            return jsonify({"error": f"Preprocessing failed: {exc}"}), 500

        # Step 4: UPDRS severity prediction
        pd.DataFrame([[csv_out.name, 0, error_frame_ratio]]).to_csv(
            str(map_csv), index=False, header=False
        )
        set_seed(42)

        try:
            df_predict = model_pred_severity(
                wkdir_path=_WKDIR,
                test_data_path=_fwd(tmp),
                test_map_path=_fwd(map_csv),
                hand_LR=hand,
                hand_pos=1,
                random_rotat_3d=True,
                seed=42,
            )
        except Exception as exc:
            return jsonify({"error": f"UPDRS prediction failed: {exc}"}), 500

        df_predict.drop(["label"], axis=1, inplace=True)

        # Step 5: kinematic parameters
        data_processed = pd.read_csv(str(csv_out))
        data_processed = _upsample(data_processed, target=200)

        try:
            results = hand_parameters(data_input=data_processed)
        except Exception as exc:
            return jsonify({"error": f"Kinematic analysis failed: {exc}"}), 500

        return jsonify(_build_response(df_predict, results, error_frame_ratio))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)