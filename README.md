# parkinson-spectacles

A real-time Parkinson's motor symptom assessment system for **Snap Spectacles**, combining on-device hand tracking with a deep learning backend.

## Overview

The user holds a finger-tapping gesture on Spectacles. A countdown timer triggers a 20-second hand joint recording session. The recording is sent to a Flask API which runs the FastEval Parkinsonism pipeline and returns a UPDRS severity score in real time.

```
Spectacles (Lens Studio)          Flask API (Python)
─────────────────────────         ──────────────────────────────
Hand gesture detected         →   POST /analyze_keypoints
Timer countdown (2s hold)         Preprocessing + normalization
20s joint recording               UPDRS model inference
HTTP POST → 63 coords/frame   →   STFT kinematic analysis
← Results_Ready trigger       ←   { updrs_score, severity, ... }
```

## Repository Structure

```
parkinson-spectacles/
├── api.py                        # Flask REST API
├── spectacles/                   # Lens Studio project
│   ├── Assets/
│   │   ├── HandJointDataCollector.js
│   │   ├── Timer.js
│   │   └── ...
│   └── hand tracking.esproj
├── src/lib/hand_predictor/       # ML pipeline (FastEval)
├── environment.yml               # Conda environment
└── docker-compose.yml
```

## Quick Start

### 1. Python backend

```bash
conda env create -f environment.yml
conda activate fp39
python api.py
# API runs at http://localhost:5000
```

### 2. Spectacles frontend

Open `spectacles/hand tracking.esproj` in Lens Studio.  
Wire up `HandJointDataCollector` and `Timer` scripts in the scene inspector.  
Push the lens to your Spectacles device.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/analyze_keypoints` | Accept pre-extracted hand joint JSON, return UPDRS score |
| POST | `/analyze` | Accept raw video, run full pipeline |

## Credits

Deep learning pipeline based on:
> Yang et al., *FastEval Parkinsonism: an instant deep learning-assisted video-based online system for Parkinsonian motor symptom evaluation*, NPJ Digital Medicine, 2024.

Original repository: [yuyuan871111/fast_eval_Parkinsonism](https://github.com/yuyuan871111/fast_eval_Parkinsonism)
