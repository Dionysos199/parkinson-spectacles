"""
demo.py - FastEval Parkinsonism Scorer
---------------------------------------
IDLE       : live webcam + MediaPipe hand skeleton, SPACE to pick video
PROCESSING : tkinter file dialog → run_inference.py → show progress
RESULTS    : UPDRS score, FG breakdown, kinematics, error rate

Run:
    C:\\Users\\imadelkhechen\\miniconda3\\envs\\fp39\\python.exe demo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Paths ─────────────────────────────────────────────────────────────────────

HP_DIR      = Path(r"C:\Users\imadelkhechen\fast_eval_Parkinsonism\src\lib\hand_predictor")
MODEL_PATH  = HP_DIR / "utils" / "saved_models" / "hand_landmarker.task"
RUN_SCRIPT  = HP_DIR / "run_inference.py"
FP39_PYTHON = Path(r"C:\Users\imadelkhechen\miniconda3\envs\fp39\python.exe")
OUTPUT_ROOT = Path(r"C:\Users\imadelkhechen\fast_eval_Parkinsonism\results")

# ── Colours (BGR) ─────────────────────────────────────────────────────────────

BG     = ( 49,  38,  27)   # #1B2631 - dark background
GREEN  = ( 96, 174,  39)   # #27AE60
ACCENT = (193, 134,  46)   # #2E86C1
WHITE  = (240, 240, 240)
LGREY  = (180, 180, 180)
GREY   = (110, 110, 110)
RED    = ( 60,  60, 200)
AMBER  = ( 40, 180, 220)
DARK   = (  8,   8,   8)

SCORE_COL = {0: GREEN, 1: GREEN, 2: AMBER, 3: RED}
SCORE_SEV = {
    0: "Normal - no impairment",
    1: "Slight - mild slowing",
    2: "Mild - moderate impairment",
    3: "Moderate / Severe",
}

# ── Layout ────────────────────────────────────────────────────────────────────

W, H = 1280, 720
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Hand skeleton ─────────────────────────────────────────────────────────────

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

# ── States ────────────────────────────────────────────────────────────────────

IDLE       = "IDLE"
PROCESSING = "PROCESSING"
RESULTS    = "RESULTS"


# ── Drawing primitives ────────────────────────────────────────────────────────

def txt(img, text, pos, scale=0.7, color=WHITE, thickness=None,
        cx=False, cy=False, shadow=True):
    """Draw text, optionally centred on x, y, or both."""
    th = thickness or max(1, int(scale * 2))
    (tw, th2), baseline = cv2.getTextSize(text, FONT, scale, th)
    x, y = pos
    if cx:
        x -= tw // 2
    if cy:
        y += th2 // 2
    if shadow:
        cv2.putText(img, text, (x + 1, y + 1), FONT, scale, DARK, th + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, th, cv2.LINE_AA)


def hline(img, y, color=GREY, alpha=0.6):
    x0, x1 = W // 6, 5 * W // 6
    overlay = img.copy()
    cv2.line(overlay, (x0, y), (x1, y), color, 1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def dark_overlay(img, x0, y0, x1, y1, alpha=0.70):
    roi = img[y0:y1, x0:x1]
    bg  = np.full_like(roi, BG)
    img[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, bg, alpha, 0)


# ── App ───────────────────────────────────────────────────────────────────────

class FastEvalApp:

    def __init__(self):
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        self._lock      = threading.Lock()
        self.state      = IDLE
        self.video_path : Path | None = None
        self.results    : dict = {}
        self.proc_msg   = ""

        # MediaPipe HandLandmarker - IMAGE mode, live overlay only
        base = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)

        # Webcam
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam.")

    # ── MediaPipe ─────────────────────────────────────────────────────────────

    def _detect(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    def _draw_hand(self, canvas, result):
        if not result or not result.hand_landmarks:
            return
        lms = result.hand_landmarks[0]
        pts = [(int(l.x * W), int(l.y * H)) for l in lms]
        for a, b in CONNECTIONS:
            cv2.line(canvas, pts[a], pts[b], GREEN, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(canvas, p, 5, GREEN,  -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 5, WHITE,   1, cv2.LINE_AA)

    # ── File picker ───────────────────────────────────────────────────────────

    def _pick_file(self) -> Path | None:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select finger-tapping video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.MOV *.MP4 *.AVI"),
                ("All files",   "*.*"),
            ],
        )
        root.destroy()
        return Path(path) if path else None

    # ── Inference thread ──────────────────────────────────────────────────────

    def _infer(self):
        try:
            stem    = self.video_path.stem
            ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = OUTPUT_ROOT / f"{stem}_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                str(FP39_PYTHON), str(RUN_SCRIPT),
                str(self.video_path),
                "--hand", "Left",
                "--output", str(out_dir),
            ]
            with self._lock:
                self.proc_msg = "Running MediaPipe keypoint extraction..."

            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HP_DIR))
            print(proc.stdout)
            if proc.returncode != 0:
                print("[stderr]", proc.stderr[-600:])

            updrs_file  = out_dir / f"{stem}_UPDRS_prediction.json"
            params_file = out_dir / f"{stem}_handparams.json"
            res = {}

            if updrs_file.exists():
                with open(updrs_file) as f:
                    res["updrs"] = json.load(f)
                print(f"[UPDRS] {json.dumps(res['updrs'], indent=2)}")
            if params_file.exists():
                with open(params_file) as f:
                    res["params"] = json.load(f)

            if not res:
                tail = proc.stderr[-300:] if proc.stderr else "no output files produced"
                res["error"] = tail

            with self._lock:
                self.results = res
                self.state   = RESULTS

        except Exception as e:
            with self._lock:
                self.results = {"error": str(e)}
                self.state   = RESULTS

    # ── Screens ───────────────────────────────────────────────────────────────

    def _screen_idle(self, frame):
        """Webcam feed with hand skeleton and prompt."""
        mirror = cv2.flip(frame, 1)
        canvas = mirror.copy()

        try:
            self._draw_hand(canvas, self._detect(mirror))
        except Exception:
            pass

        # Top bar
        dark_overlay(canvas, 0, 0, W, 52)
        txt(canvas, "FastEval  -  Parkinson Finger Tapping Scorer",
            (W // 2, 33), 0.75, WHITE, cx=True)

        # Centre prompt
        dark_overlay(canvas, W // 4, H // 2 - 36, 3 * W // 4, H // 2 + 20, alpha=0.75)
        txt(canvas, "Press SPACE to select a video file",
            (W // 2, H // 2 + 2), 0.90, ACCENT, cx=True, cy=True)

        # Bottom bar
        dark_overlay(canvas, 0, H - 44, W, H)
        txt(canvas, "SPACE = select video   |   ESC = quit",
            (W // 2, H - 14), 0.60, GREY, cx=True)

        return canvas

    def _screen_processing(self, frame):
        """Dark frame with filename + animated dots."""
        canvas = np.zeros((H, W, 3), dtype=np.uint8); canvas[:] = BG

        with self._lock:
            msg  = self.proc_msg
            name = self.video_path.name if self.video_path else ""

        dots = "●" * (int(time.time() * 1.5) % 4)

        txt(canvas, "Analysing" + dots,
            (W // 2, H // 2 - 50), 1.1, ACCENT, cx=True, cy=True)
        txt(canvas, name,
            (W // 2, H // 2 + 10), 0.65, LGREY, cx=True, cy=True)
        txt(canvas, msg,
            (W // 2, H // 2 + 55), 0.55, GREY, cx=True, cy=True)

        return canvas

    def _screen_results(self):
        """Dark results panel."""
        canvas = np.zeros((H, W, 3), dtype=np.uint8); canvas[:] = BG

        with self._lock:
            res  = dict(self.results)
            name = self.video_path.name if self.video_path else ""

        cx = W // 2
        cy = 56

        # ── Header ────────────────────────────────────────────────────────────
        txt(canvas, "FastEval  -  Results", (cx, cy), 0.80, WHITE, cx=True, cy=True)
        cy += 22
        txt(canvas, name, (cx, cy), 0.52, GREY, cx=True, cy=True)
        cy += 28
        hline(canvas, cy)
        cy += 28

        # ── Error state ───────────────────────────────────────────────────────
        if "error" in res:
            txt(canvas, "Inference failed", (cx, cy), 0.85, RED, cx=True, cy=True)
            cy += 36
            for line in res["error"].replace("\r", "").split("\n")[-4:]:
                txt(canvas, line[:90], (cx, cy), 0.45, GREY, cx=True, cy=True)
                cy += 22
            txt(canvas, "SPACE = try another video   |   ESC = quit",
                (cx, H - 18), 0.60, GREY, cx=True)
            return canvas

        # ── UPDRS score ───────────────────────────────────────────────────────
        u       = res.get("updrs", {})
        overall = u.get("predict_overall")
        efr     = u.get("error_frame_ratio")

        if overall is not None:
            score = int(round(overall))
            scol  = SCORE_COL.get(score, WHITE)
            sev   = SCORE_SEV.get(score, "Unknown")

            txt(canvas, "UPDRS 3.4  -  Finger Tapping",
                (cx, cy), 0.70, LGREY, cx=True, cy=True)
            cy += 30

            # Big score digit
            (sw, sh), _ = cv2.getTextSize(str(score), FONT, 5.5, 10)
            cv2.putText(canvas, str(score), (cx - sw // 2, cy + sh),
                        FONT, 5.5, scol, 10, cv2.LINE_AA)
            cy += sh + 18

            txt(canvas, sev, (cx, cy), 0.78, scol, cx=True, cy=True)
            cy += 32

            # FG breakdown
            fg_parts = []
            for tag in ("FG_1", "FG_2", "FG_3"):
                key = f"predict_Left_{tag}"
                val = u.get(key)
                v   = str(int(val)) if val is not None else "-"
                fg_parts.append(f"{tag.replace('_', ' ')} : {v}")
            txt(canvas, "   ".join(fg_parts), (cx, cy), 0.58, GREY, cx=True, cy=True)
            cy += 28

        else:
            # Rejected
            txt(canvas, "Prediction rejected", (cx, cy), 0.88, AMBER, cx=True, cy=True)
            cy += 30
            if efr is not None:
                det = (1 - efr) * 100
                txt(canvas, f"Hand detected in only {det:.0f}% of frames - use better lighting",
                    (cx, cy), 0.60, RED, cx=True, cy=True)
                cy += 28

        # ── Divider ───────────────────────────────────────────────────────────
        cy += 4
        hline(canvas, cy)
        cy += 26

        # ── Kinematic parameters ──────────────────────────────────────────────
        p = res.get("params", {})
        if p:
            txt(canvas, "Kinematic Parameters", (cx, cy), 0.65, LGREY, cx=True, cy=True)
            cy += 28

            def _f(v, fmt=""):
                return format(float(v), fmt) if v is not None else "N/A"

            peak_count = len(p.get("peaks", {}).get("time", []))

            rows = [
                ("Frequency",  f"{_f(p.get('freq-mean'), '.2f')} Hz"),
                ("Intensity",  f"{_f(p.get('intensity-mean'), '.4f')}"),
                ("FI value",   f"{_f(p.get('inte-freq-mean'), '.4f')}"),
                ("Peaks",      str(peak_count)),
            ]

            col_w  = 240
            row_x  = cx - (len(rows) * col_w) // 2 + col_w // 2

            for label, val in rows:
                txt(canvas, label, (row_x, cy),      0.52, GREY,  cx=True, cy=True)
                txt(canvas, val,   (row_x, cy + 22), 0.68, WHITE, cx=True, cy=True)
                row_x += col_w

            cy += 56

        # ── Error frame ratio ─────────────────────────────────────────────────
        if efr is not None:
            det     = (1 - efr) * 100
            efr_col = GREEN if det > 80 else (AMBER if det > 50 else RED)
            txt(canvas, f"Detection rate  {det:.0f}%   ({efr*100:.1f}% frames missed)",
                (cx, cy), 0.55, efr_col, cx=True, cy=True)

        # ── Footer ────────────────────────────────────────────────────────────
        hline(canvas, H - 44)
        txt(canvas, "SPACE = load another video   |   ESC = quit",
            (cx, H - 18), 0.60, GREY, cx=True)

        return canvas

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cv2.namedWindow("FastEval", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FastEval", W, H)
        cv2.setWindowProperty("FastEval", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            with self._lock:
                state = self.state

            if state == IDLE:
                canvas = self._screen_idle(frame)
            elif state == PROCESSING:
                canvas = self._screen_processing(frame)
            else:   # RESULTS
                canvas = self._screen_results()

            cv2.imshow("FastEval", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord(" "):
                if state == IDLE:
                    picked = self._pick_file()
                    if picked and picked.exists():
                        self.video_path = picked
                        print(f"\n[FILE] {picked}")
                        with self._lock:
                            self.results  = {}
                            self.proc_msg = "Starting..."
                            self.state    = PROCESSING
                        threading.Thread(target=self._infer, daemon=True).start()
                elif state == RESULTS:
                    with self._lock:
                        self.state      = IDLE
                        self.results    = {}
                        self.video_path = None

        self.cap.release()
        self.landmarker.close()
        cv2.destroyAllWindows()


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("FastEval - loading HandLandmarker model...")
    try:
        app = FastEvalApp()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print("Ready. Press SPACE to select a video, ESC to quit.")
    app.run()
    print("Done.")
