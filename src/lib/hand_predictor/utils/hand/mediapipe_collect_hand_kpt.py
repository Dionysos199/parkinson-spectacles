import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

_MODEL_PATH = Path(__file__).parent.parent / "saved_models" / "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]


def _ensure_model():
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _MODEL_PATH.exists():
        print(f"  Downloading hand landmarker model -> {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, str(_MODEL_PATH))
        print("  Model downloaded.")


def collect_hand_keypoints_pipe(video_path: str, hand_query, output_path: str = None, threshold=0.5, logging=False):
    '''
    video_path: [str] your video path
    hand_query: ["Left", "Right"] which hand you want to extract
    output_path: [str] your output csv
    threshold: [float, 0~1] confidence ratio you want to fix
    '''
    _ensure_model()

    # Read fps before loading frames (needed for VIDEO mode timestamps)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        fps = 60.0

    frames = read_video_data(video_path, logging=logging)
    time_frame, keypoints_list, _, annotated_images = collect_hand_keypoints(
        frames, hand_query, threshold=threshold, logging=logging,
        create_annotated_img=True, fps=fps,
    )

    cols = (
        ["timestamp"]
        + [f"x_{idx}" for idx in range(21)]
        + [f"y_{idx}" for idx in range(21)]
        + [f"z_{idx}" for idx in range(21)]
    )
    if len(time_frame) == 0:
        keypoints_data = pd.DataFrame(columns=cols)
    else:
        t = pd.DataFrame(time_frame)
        x = pd.DataFrame(keypoints_list[0].T)
        y = pd.DataFrame(keypoints_list[1].T)
        z = pd.DataFrame(keypoints_list[2].T)
        keypoints_data = pd.concat([t, x, y, z], axis=1)
        keypoints_data.columns = cols

    if output_path is not None:
        keypoints_data.to_csv(output_path, index=False)
    if logging:
        print(f"{video_path}: DONE.")

    return annotated_images


# read video
def read_video_data(video_path: str, logging: bool = False):
    assert os.path.isfile(video_path), "Files doesn't exist."

    cap = cv2.VideoCapture(video_path)
    counter = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if logging:
                print(f"Total frames: {counter}. Can't receive more frame (stream end?). Exiting ...")
            break
        counter += 1
        frames.append(frame)
    cap.release()

    return frames


# collect hand keypoints by mediapipe
def collect_hand_keypoints(frames, hand_query="Right", create_annotated_img=False, threshold=0.5, logging=False, fps=60.0):
    _ensure_model()

    # The Tasks API float16 model outputs lower raw confidence scores than
    # mp.solutions.hands did, so the external threshold (0.5) would reject most
    # valid frames.  Clamp to 0.1 so the model actually fires; api.py's
    # fallback-to-thre0 path still works correctly.
    conf = min(max(float(threshold), 0.01), 0.1)

    base_options = mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=conf,
        min_hand_presence_confidence=conf,
        min_tracking_confidence=conf,
    )

    keypoints_list = []
    images_series = []
    fail_to_detect_hand = []
    time_frame = []

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        pbar = tqdm(frames) if logging else frames
        for idx, frame in enumerate(pbar):
            # Flip for correct handedness labelling (same as original pipeline)
            image = cv2.flip(frame, 1)

            # Tasks API float16 model detects poorly on tall portrait frames.
            # Rotate to landscape before inference; world landmarks are 3D and
            # camera-relative so they're unaffected by image-plane rotation.
            h_img, w_img = image.shape[:2]
            portrait = h_img > w_img
            if portrait:
                det_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            else:
                det_image = image

            image_rgb = cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            result = landmarker.detect(mp_image)

            # Build annotated frame (drawn on every frame so images_series length == len(frames))
            if create_annotated_img:
                annotated = image.copy()
                if result.hand_landmarks:
                    h, w = annotated.shape[:2]
                    for hand_lms in result.hand_landmarks:
                        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                        for pt in pts:
                            cv2.circle(annotated, pt, 4, (0, 255, 0), -1)
                        for a, b in _CONNECTIONS:
                            cv2.line(annotated, pts[a], pts[b], (0, 200, 0), 2)
                images_series.append(annotated)

            # No hand detected in this frame
            if not result.hand_world_landmarks or not result.handedness:
                fail_to_detect_hand.append(image)
                continue

            # Pick the requested hand; if convention has shifted (Tasks API vs
            # mp.solutions label the same hand oppositely after the flip), fall
            # back to whichever hand was detected so single-hand videos always work.
            world_lms = None
            for jdx, lms in enumerate(result.hand_world_landmarks):
                if result.handedness[jdx][0].category_name == hand_query:
                    world_lms = lms
                    break
            if world_lms is None:
                world_lms = result.hand_world_landmarks[0]  # fallback: take first

            keypoints = [[lm.x, lm.y, lm.z] for lm in world_lms]
            keypoints_list.append(np.array(keypoints).T)  # (3, 21)
            time_frame.append(idx)

    keypoints_list = np.array(keypoints_list) if keypoints_list else np.zeros((0, 3, 21))
    keypoints_list = np.moveaxis(keypoints_list, 0, -1)
    # dimension of keypoints: (xyz, keypoints, timeframe) = (3, 21, timeframe)

    if create_annotated_img:
        return time_frame, keypoints_list, fail_to_detect_hand, images_series
    else:
        return time_frame, keypoints_list, fail_to_detect_hand
