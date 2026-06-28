# 🚨 Emergency Vehicle Prioritization System

> A Multi-Modal Framework for Emergency Vehicle Prioritization Using Visual Detection and Acoustic Classification

---

## 📌 Overview

This system automatically detects emergency vehicles (ambulances, fire trucks, police cars) approaching a 4-way intersection and dynamically adjusts traffic signals to clear their path — using **two independent AI models working together**.

- 🎥 **Visual Detection** — YOLOv8 identifies the vehicle and its approach direction from camera feed
- 🔊 **Acoustic Classification** — PANNs CNN14 detects siren sounds from audio
- 🔀 **Multi-Modal Fusion** — A sliding vote-queue combines both signals with temporal smoothing (Algorithm 1)
- 🚦 **Signal Control** — The confirmed EV direction gets a green wave; all other lanes go red

---

## 🏗️ System Architecture

```
Camera / Video File ──► VideoModule ──► conf_v, direction, frame_idx
                                               │
Video File ──► AudioModule (ffmpeg extract) ──► conf_a (frame-synced)
Microphone ──► AudioModule (live mic)   ──────┘
                                               │
                              FusionModule.push_frame(conf_v, conf_a, direction)
                                │
                                │  v(t) = 1  if  conf_v > τ_v (0.50)
                                │  a(t) = 1  if  conf_a > τ_a (0.30)
                                │  vote  = v OR a   [default OR mode]
                                │  queue[N=10].append(vote)
                                │  confirmed = Σ(queue) ≥ K=4
                                │
                         EvalModule.push(gt, pred_audio, pred_video, pred_fusion)
                                │
                         SignalModule / IntersectionWidget
                           confirmed + direction → green wave for EV lane
                           audio-only (no direction) → all-red safe fallback
```

---

## ✨ Features

- **Dark command-center GUI** built with PyQt5 — live video feed, 4-way intersection visualizer, 3-lamp animated traffic lights
- **Dual input modes** — upload a video file (audio extracted automatically via ffmpeg) or use a live webcam + microphone
- **Algorithm 1 fusion** — sliding binary vote queue with configurable N (window) and K (threshold), togglable OR / AND logic
- **Evaluation metrics panel** — real-time Precision, Recall, F1, Accuracy, FPR, FNR, MCC, and latency tiles with color-coded health indicators
- **CSV export** — full per-frame evaluation results exportable for Excel / LaTeX
- **Event log** — color-coded live log (purple = audio, blue = video, orange = fusion, gray = system)
- **Runtime tunable** — change N, K, τ_v, τ_a, and fusion mode without restarting

---

## 🗂️ Project Structure

```
your_project/
├── main.py               # PyQt5 GUI — main window, tick loop, layout
├── fusion_module.py      # Algorithm 1: sliding vote queue fusion engine
├── eval_module.py        # Evaluation metrics (Precision/Recall/F1/MCC/latency)
├── signal_module.py      # Traffic signal state machine (3-lamp, yellow state)
├── audio_module.py       # PANNs CNN14 inference (live mic + video extraction)
├── video_module.py       # YOLOv8 inference (confidence, bbox, direction)
├── best.pt               # YOLOv8 custom weights (not included — see below)
└── utils/
    └── utils.py          # OpenCV → QPixmap frame converter
```

---

## ⚙️ Requirements

### Python
```
Python 3.10+
```

### Dependencies
```
pip install PyQt5 opencv-python ultralytics torch torchaudio
pip install panns-inference librosa sounddevice numpy
```

### External Tools
| Tool | Purpose | Install |
|------|---------|---------|
| **ffmpeg** | Extract audio from video files | [ffmpeg.org](https://ffmpeg.org/download.html) — add to PATH |
| **YOLOv8 weights** (`best.pt`) | Custom-trained EV detector | Train your own or use `yolov8n.pt` as a starting point |

> **ffmpeg fallback:** If ffmpeg is not found, the system tries `moviepy` as a secondary extractor. If neither is available, the system runs in video-only mode (fusion still works in OR mode).

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/ev-prioritization-system.git
cd ev-prioritization-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your YOLOv8 weights
cp /path/to/best.pt ./best.pt

# 4. Run
python main.py
```

### Usage

1. Click **Browse Video** and select an MP4/AVI file — audio is extracted automatically
2. Select **Fusion Mode** (OR = high sensitivity, AND = high precision)
3. Tune **N** (queue window, default 10) and **K** (activation threshold, default 4) in the toolbar
4. Click **Start Detection**
5. Toggle **GT: EV PRESENT** in the evaluation panel while a real EV is in frame to collect ground-truth metrics
6. Click **Export CSV** to save evaluation results

---

## 🧠 How the Fusion Works (Algorithm 1)

Each video frame, the system computes two binary signals:

```
v(t) = 1  if  YOLOv8_confidence > τ_v (0.50),  else 0
a(t) = 1  if  CNN14_siren_prob   > τ_a (0.30),  else 0
```

These are combined into a single vote per frame:

| Mode | Vote |
|------|------|
| **OR** (default, safety-first) | `v(t) OR a(t)` |
| **AND** (precision-first) | `v(t) AND a(t)` |

The vote is appended to a rolling queue of length **N**. Priority is confirmed when:

```
D_smooth(t) = 1   if   Σ queue[i]  ≥  K
```

This acts as a digital hysteresis filter — a single noisy frame cannot trigger or cancel priority.

---

## 📊 Evaluation Metrics

The built-in `EvalModule` computes the following in real time:

| Metric | Description |
|--------|-------------|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 Score | Harmonic mean of Precision and Recall |
| Accuracy | (TP + TN) / Total |
| FPR | False Positive Rate |
| FNR | False Negative Rate |
| MCC | Matthews Correlation Coefficient |
| Latency | Mean detection lag (frames) |

Comparative metrics (fusion vs. audio-only, fusion vs. video-only) are included in the CSV export.

---

## 🔧 Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_v` | `0.50` | YOLOv8 confidence threshold (τ_vis) |
| `tau_a` | `0.30` | CNN14 siren probability threshold (τ_aud) |
| `n_frames` | `10` | Vote queue length N |
| `k_thresh` | `4` | Activation threshold K (40% of queue) |
| `mode` | `OR` | Fusion logic — `OR` or `AND` |

---

## 📃 License

This project is submitted as an academic final year project. All rights reserved by the authors and JSS Academy of Technical Education, Noida.
