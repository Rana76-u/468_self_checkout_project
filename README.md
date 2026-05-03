# 🛒 Self-Checkout System
### Real-Time Retail Item Detection & Billing using YOLOv10x + DeepSORT

A computer vision pipeline that automatically detects retail products in video footage, tracks them to prevent duplicate billing, and renders a live receipt overlay — all in real time.

---

## Overview

This system processes a video feed (e.g. a checkout counter camera), identifies items from a catalog of 32 common retail products, and builds a running bill using BDT (Bangladeshi Taka) pricing. Each detected item is assigned a unique track ID via DeepSORT to ensure it is only billed once, even if it remains in frame across multiple frames.

**Key capabilities:**
- Detects 32 retail item classes using YOLOv10x (extra-large variant for maximum accuracy)
- Tracks objects across frames with DeepSORT (prevents duplicate entries)
- Renders a live semi-transparent receipt overlay on each video frame
- Logs per-frame detection and tracking data for post-run analysis
- Exports an annotated output video with bounding boxes and a live bill

---

## Demo Pipeline

```
Input Video → YOLO Detection → Class Filter → DeepSORT Tracking → Bill Registration → Annotated Output Video
```

---

## Demo Video


[![Watch the demo video](https://checklens.ai/wp-content/uploads/2022/09/Loss-prevention-at-the-Self-Checkout.jpg)](https://www.youtube.com/watch?v=jqCTpxwdPgQ?si=0myuqFKH-NujbGa4)


---

## Requirements

Install all dependencies in one command (Google Colab):

```bash
pip install ultralytics deep-sort-realtime pillow
```

**Core libraries used:**

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv10x object detection |
| `deep-sort-realtime` | Multi-object tracking with persistent IDs |
| `opencv-python` | Frame capture and video I/O |
| `Pillow` | Receipt overlay rendering |
| `pandas` / `matplotlib` | Post-run analysis and visualizations |

---

## Supported Item Classes & Prices (BDT)

| Item | Price (Tk) | Item | Price (Tk) |
|---|---|---|---|
| Laptop | 45,000 | Cell Phone | 8,000 |
| Oven | 15,000 | Microwave | 12,000 |
| TV | 12,000 | Keyboard | 2,500 |
| Hair Drier | 1,800 | Mouse | 1,200 |
| Clock | 600 | Handbag | 450 |
| Cake | 400 | Pizza | 350 |
| Teddy Bear | 350 | Toaster | 3,500 |
| Remote | 350 | Book | 200 |
| Sandwich | 120 | Scissors | 120 |
| Knife | 90 | Bowl | 80 |
| Bottle | 60 | Toothbrush | 60 |
| Broccoli | 55 | Donut | 50 |
| Tie | 250 | Sports Ball | 180 |
| Apple | 20 | Orange | 18 |
| Banana | 15 | Carrot | 25 |
| Cup | 40 | Spoon | 35 |

---

## Usage

### 1. Mount Google Drive and set input path

```python
from google.colab import drive
drive.mount('/content/drive')

input_video_path = '/content/drive/MyDrive/468_Self_Checkout/input_4.mp4'
```

### 2. Run the processing pipeline

The notebook will:
1. Load YOLOv10x and initialize the DeepSORT tracker
2. Read the input video frame by frame
3. Detect and filter items to the supported class list
4. Register each unique tracked object in the bill
5. Draw bounding boxes and the receipt overlay on each frame
6. Save the annotated video back to Google Drive

### 3. View analysis results

After processing, the notebook generates:
- A **summary table** (total items, grand total, avg confidence, FPS)
- **Detection charts** (raw vs. filtered vs. tracked counts per frame)
- **Confidence distribution histogram**
- **Detection stability chart** (per-track confidence over time)
- **Tracking continuity report** (ID switch rate)

---

## Architecture

### Core Components

| Component | Description |
|---|---|
| `BillState` | Thread-safe registry mapping track IDs to class names; prevents duplicate billing |
| `FrameLogger` | Collects per-frame detection and tracking metadata for analysis |
| `process_frame()` | Main pipeline: YOLO → filter → DeepSORT → register → render |
| `draw_detections()` | Draws colored bounding boxes with class name, track ID, and price |
| `draw_receipt_overlay()` | Renders a semi-transparent receipt panel (Pillow RGBA compositing) |

### Detection Pipeline (per frame)

```
1. YOLOv10x inference  (conf ≥ 0.4)
2. Filter to ALLOWED_CLASSES
3. DeepSORT track update  (max_age=30, n_init=3)
4. Register confirmed tracks in BillState
5. Log frame data to FrameLogger
6. Render bounding boxes + receipt overlay
```

---

## Configuration

Key parameters at the top of the notebook:

```python
YOLO_CONF_THRESH  = 0.4    # Minimum detection confidence
DEEPSORT_MAX_AGE  = 30     # Frames to keep a lost track alive
DEEPSORT_N_INIT   = 3      # Frames before a track is confirmed
```

---

## Output

- **Annotated video**: saved as `processed_<input_filename>.mp4` in the same Drive folder
- **Analysis figure**: `self_checkout_analysis.png` saved locally in Colab
- **Console summary**: printed statistics including grand total, FPS, and confidence metrics

---

## Notes

- Designed to run on **Google Colab** with GPU acceleration for real-time performance.
- Prices are denominated in **BDT (Bangladeshi Taka)**.
- The system uses YOLOv10x's 80-class COCO weight file; only the 32 classes listed above are billed.
- DeepSORT's `n_init=3` means an object must appear in at least 3 consecutive frames before it is confirmed and added to the bill — reducing false positives from transient detections.
