# video_module.py
"""
Emergency vehicle detection using YOLOv8 + simple centroid tracker.

Changes from v1:
  - Exposes last_confidence, last_class_name, last_bbox attributes so the
    dashboard can show per-detection detail without modifying VideoModule's
    public get_frame() contract.
  - Class names mapped from integer IDs (0=ambulance, 1=firetruck, 2=police).
  - annotate_frame() draws bounding boxes + labels onto the frame before it
    is handed to the GUI (optional, enabled by draw_annotations=True).
  - Tracking radius is now configurable (default 50 px).
  - get_frame() returns (frame, detections) where detections is a list of
    dicts {direction, confidence, class_name, bbox} — backwards-compatible
    callers that only unpack (frame, directions) still work because
    directions is also returned via the .directions property below, but
    _tick() in main.py now reads the richer list.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# Human-readable class names (adjust to match your model's training labels)
CLASS_NAMES = {0: "ambulance", 1: "fire truck", 2: "police car"}


class VideoModule:
    """
    Module for emergency vehicle detection using YOLOv8 and centroid tracking.

    Public attributes set after each get_frame() call:
        last_confidence  (float | None) : highest detection confidence this tick.
        last_class_name  (str)          : class label of the highest-conf detection.
        last_bbox        (str)          : "x1,y1,x2,y2" string of that detection.
        last_detections  (list[dict])   : full list of detection dicts this tick.
    """

    def __init__(self, model_path: str, mode: str = "live",
                 video_file: str = None, conf_threshold: float = 0.5,
                 track_radius: int = 50, draw_annotations: bool = True):
        """
        Args:
            model_path       : Path to YOLOv8 weights file (e.g. best.pt).
            mode             : 'live' for webcam, 'file' for video file.
            video_file       : Path to video file (used when mode='file').
            conf_threshold   : Minimum confidence to accept a detection.
            track_radius     : Pixel radius for centroid matching between frames.
            draw_annotations : If True, bounding boxes are drawn onto the frame.
        """
        self.mode = mode
        self.video_file = video_file
        self.conf_threshold = conf_threshold
        self.track_radius = track_radius
        self.draw_annotations = draw_annotations

        self.model = YOLO(model_path)

        if self.mode == "live":
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(self.video_file)

        # Centroid tracking state
        self.tracks: list[dict] = []   # [{'id', 'center', 'class_id'}]
        self.next_id: int = 0

        # Per-tick metadata (readable by the GUI)
        self.last_confidence: float | None = None
        self.last_class_name: str = "—"
        self.last_bbox: str = "—"
        self.last_detections: list[dict] = []

    # ──────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────

    def get_frame(self):
        """
        Capture one frame, run YOLO, update tracks.

        Returns:
            frame      : Annotated BGR numpy array (or None on read failure).
            directions : List of direction strings, e.g. ['N', 'E'].
                         (Kept for backwards compatibility with main.py v1.)
        """
        ret, frame = self.cap.read()
        if not ret:
            self._reset_metadata()
            return None, []

        results = self.model(frame, verbose=False)
        raw_detections = self._parse_results(results)
        detections_with_dir = self._track_and_infer(raw_detections)

        # Store rich metadata for the GUI
        self.last_detections = detections_with_dir
        if detections_with_dir:
            best = max(detections_with_dir, key=lambda d: d["confidence"])
            self.last_confidence = best["confidence"]
            self.last_class_name = best["class_name"]
            x1, y1, x2, y2 = best["bbox"]
            self.last_bbox = f"{x1},{y1},{x2},{y2}"
        else:
            self._reset_metadata()

        if self.draw_annotations:
            self._annotate(frame, detections_with_dir)

        directions = [d["direction"] for d in detections_with_dir if d["direction"]]
        return frame, directions

    def release(self):
        """Release the video capture device."""
        self.cap.release()

    # ──────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────

    def _parse_results(self, results) -> list[dict]:
        """Extract bounding boxes for emergency-vehicle classes above threshold."""
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy  = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss  = r.boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                cls_id = int(clss[i])
                conf   = float(confs[i])
                if cls_id not in CLASS_NAMES:
                    continue
                if conf < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = (int(v) for v in xyxy[i])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append({
                    "center":     (cx, cy),
                    "bbox":       (x1, y1, x2, y2),
                    "confidence": conf,
                    "class_id":   cls_id,
                    "class_name": CLASS_NAMES[cls_id],
                    "direction":  None,   # filled in by _track_and_infer
                })
        return detections

    def _track_and_infer(self, detections: list[dict]) -> list[dict]:
        """
        Match new detections to existing centroid tracks.
        Compute movement direction for matched tracks.
        Returns the same detection dicts with 'direction' filled in.
        """
        updated_tracks = []
        for det in detections:
            cx, cy = det["center"]
            matched = False
            for track in self.tracks:
                tx, ty = track["center"]
                if abs(cx - tx) < self.track_radius and abs(cy - ty) < self.track_radius:
                    dx = cx - tx
                    dy = cy - ty
                    if abs(dy) >= abs(dx):
                        direction = "N" if dy > 0 else "S"
                    else:
                        direction = "W" if dx > 0 else "E"
                    det["direction"] = direction
                    track["center"] = (cx, cy)
                    track["class_id"] = det["class_id"]
                    updated_tracks.append(track)
                    matched = True
                    break
            if not matched:
                new_track = {
                    "id":       self.next_id,
                    "center":   (cx, cy),
                    "class_id": det["class_id"],
                }
                self.next_id += 1
                updated_tracks.append(new_track)
                # No direction for brand-new track (first appearance)
                det["direction"] = None

        self.tracks = updated_tracks
        return detections

    def _annotate(self, frame: np.ndarray, detections: list[dict]):
        """Draw bounding boxes and labels onto the frame in-place."""
        COLOR = (0, 255, 136)       # green accent matching the dark theme
        FONT  = cv2.FONT_HERSHEY_SIMPLEX

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf      = det["confidence"]
            cls_name  = det["class_name"]
            direction = det["direction"] or "?"

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)

            # Label background
            label = f"{cls_name}  {conf:.2f}  [{direction}]"
            (tw, th), baseline = cv2.getTextSize(label, FONT, 0.45, 1)
            cv2.rectangle(frame,
                          (x1, y1 - th - baseline - 4),
                          (x1 + tw + 4, y1),
                          (13, 21, 32), cv2.FILLED)

            # Label text
            cv2.putText(frame, label,
                        (x1 + 2, y1 - baseline - 2),
                        FONT, 0.45, COLOR, 1, cv2.LINE_AA)

    def _reset_metadata(self):
        self.last_confidence = None
        self.last_class_name = "—"
        self.last_bbox        = "—"
        self.last_detections  = []
