# eval_module.py
"""
Evaluation metrics module for the multi-modal Emergency Vehicle Prioritization system.

Records every frame's ground-truth label and system prediction, then computes
the standard binary-classification metrics used in the paper and in ITS research:

  Per-modality (audio-only, video-only):
    Precision, Recall, F1, Accuracy, FPR, FNR

  Fusion output (Dsmooth):
    Precision, Recall, F1, Accuracy, FPR, FNR
    False-positive rate reduction vs each unimodal baseline
    False-negative rate reduction vs each unimodal baseline

  Temporal metrics:
    Mean detection latency (frames from first ground-truth positive to
    first confirmed positive) — measures how quickly the sliding window
    responds to a new event.
    Mean clearance latency (frames from last ground-truth positive to
    first confirmed negative) — measures how quickly the system clears.

  Signal-quality metrics:
    Signal Stability Index (SSI) — fraction of frames where the output
    did NOT change state.  High SSI means fewer spurious toggling events.
    Toggle rate — number of 0→1 or 1→0 transitions per 100 frames.

Usage
-----
Every tick, call:
    evaluator.push(
        gt=True/False,       # ground truth label for this frame
        pred_audio=True/False,
        pred_video=True/False,
        pred_fusion=True/False,
    )

Then read computed metrics at any time:
    metrics = evaluator.compute()   # returns a dict of all metrics

Or reset between runs:
    evaluator.reset()

Ground-truth labelling
----------------------
In live/file mode without a separate annotation file, ground truth can be
provided manually through the GUI (the operator presses a "GT: EV present"
button while reviewing the feed), or loaded from a CSV annotation file with
columns [frame_index, label] where label is 0 or 1.
"""

import math
import time
from collections import deque
from typing import Optional


class EvalModule:
    """
    Online (streaming) evaluation recorder for the fusion pipeline.

    All internal state is updated incrementally — no stored list of all frames
    is needed for the core confusion-matrix metrics (O(1) memory).
    Latency and toggle history use small bounded deques.
    """

    def __init__(self, latency_window: int = 500):
        """
        Args:
            latency_window: maximum number of frames to keep for latency
                            and stability analysis (a rolling buffer).
        """
        self.latency_window = latency_window
        self.reset()

    # ──────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────

    def reset(self):
        """Clear all counters and history."""
        # Confusion matrix counts for each modality
        # Keys: 'audio', 'video', 'fusion'
        self._tp  = {'audio': 0, 'video': 0, 'fusion': 0}
        self._fp  = {'audio': 0, 'video': 0, 'fusion': 0}
        self._tn  = {'audio': 0, 'video': 0, 'fusion': 0}
        self._fn  = {'audio': 0, 'video': 0, 'fusion': 0}

        self._total_frames: int = 0

        # Previous predictions for toggle detection
        self._prev = {'audio': False, 'video': False, 'fusion': False}
        self._toggles = {'audio': 0, 'video': 0, 'fusion': 0}
        self._stable  = {'audio': 0, 'video': 0, 'fusion': 0}

        # Latency tracking (for the fusion output only)
        # _in_event: True while ground truth is currently positive
        self._in_gt_event: bool = False
        self._event_start_frame: Optional[int] = None
        self._detected_in_event: bool = False
        self._latencies: deque = deque(maxlen=self.latency_window)

        # Clearance latency
        self._in_clear_wait: bool = False
        self._clear_start_frame: Optional[int] = None
        self._clearance_latencies: deque = deque(maxlen=self.latency_window)

        # Confidence tracking (mean confidence when above threshold)
        self._conf_v_sum: float = 0.0
        self._conf_a_sum: float = 0.0
        self._conf_v_n:   int   = 0
        self._conf_a_n:   int   = 0

        # Session start
        self._start_time: float = time.time()

    # ──────────────────────────────────────────────────
    # Per-frame ingestion
    # ──────────────────────────────────────────────────

    def push(
        self,
        gt:           bool,
        pred_audio:   bool,
        pred_video:   bool,
        pred_fusion:  bool,
        conf_v:       float = 0.0,
        conf_a:       float = 0.0,
    ):
        """
        Record one frame's ground-truth label and predictions.

        Args:
            gt           : True if an emergency vehicle is actually present.
            pred_audio   : True if the audio module fired (conf_a > τ_a).
            pred_video   : True if the video module fired (conf_v > τ_v).
            pred_fusion  : True if the fused, smoothed output is active.
            conf_v       : Raw YOLOv8 confidence (for mean-confidence stats).
            conf_a       : Raw CNN14 siren probability.
        """
        t = self._total_frames
        self._total_frames += 1

        preds = {'audio': pred_audio, 'video': pred_video, 'fusion': pred_fusion}

        # ── Confusion matrix ───────────────────────────
        for key, pred in preds.items():
            if gt and pred:
                self._tp[key] += 1
            elif gt and not pred:
                self._fn[key] += 1
            elif not gt and pred:
                self._fp[key] += 1
            else:
                self._tn[key] += 1

        # ── Toggle / stability ─────────────────────────
        for key, pred in preds.items():
            if pred != self._prev[key]:
                self._toggles[key] += 1
            else:
                self._stable[key] += 1
            self._prev[key] = pred

        # ── Detection latency ──────────────────────────
        # Rising edge of ground truth
        if gt and not self._in_gt_event:
            self._in_gt_event = True
            self._event_start_frame = t
            self._detected_in_event = False

        # First fusion confirmation within the event
        if self._in_gt_event and pred_fusion and not self._detected_in_event:
            latency = t - self._event_start_frame
            self._latencies.append(latency)
            self._detected_in_event = True

        # Falling edge of ground truth
        if not gt and self._in_gt_event:
            self._in_gt_event = False
            # If never detected during this event, record a miss (latency = inf sentinel)
            if not self._detected_in_event:
                self._latencies.append(float('inf'))
            # Begin watching for clearance
            self._in_clear_wait = True
            self._clear_start_frame = t

        # Clearance latency: frames until fusion goes False after GT goes False
        if self._in_clear_wait and not pred_fusion:
            clat = t - self._clear_start_frame
            self._clearance_latencies.append(clat)
            self._in_clear_wait = False

        # ── Confidence accumulation ────────────────────
        if conf_v > 0:
            self._conf_v_sum += conf_v
            self._conf_v_n   += 1
        if conf_a > 0:
            self._conf_a_sum += conf_a
            self._conf_a_n   += 1

    # ──────────────────────────────────────────────────
    # Ground-truth helpers (for manual GT annotation)
    # ──────────────────────────────────────────────────

    def load_gt_csv(self, path: str) -> list:
        """
        Load ground-truth labels from a CSV file.

        Expected format (no header):   frame_index,label
        where label is 0 (no EV) or 1 (EV present).

        Returns a list of (frame_index, label) tuples sorted by frame_index.
        """
        labels = []
        with open(path, newline='') as f:
            import csv
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        labels.append((int(row[0]), int(row[1]) == 1))
                    except ValueError:
                        pass  # skip header or malformed rows
        labels.sort(key=lambda x: x[0])
        return labels

    # ──────────────────────────────────────────────────
    # Metric computation
    # ──────────────────────────────────────────────────

    @staticmethod
    def _safe_div(num: float, den: float, default: float = 0.0) -> float:
        return num / den if den > 0 else default

    def _metrics_for(self, key: str) -> dict:
        """Compute all per-key binary classification metrics."""
        tp = self._tp[key]
        fp = self._fp[key]
        tn = self._tn[key]
        fn = self._fn[key]
        total = tp + fp + tn + fn

        precision = self._safe_div(tp, tp + fp)
        recall    = self._safe_div(tp, tp + fn)   # = TPR / sensitivity
        f1        = self._safe_div(2 * precision * recall, precision + recall)
        accuracy  = self._safe_div(tp + tn, total)
        fpr       = self._safe_div(fp, fp + tn)   # false-positive rate
        fnr       = self._safe_div(fn, fn + tp)   # false-negative rate  (= 1 - recall)
        specificity = self._safe_div(tn, tn + fp) # = 1 - FPR

        # Matthews Correlation Coefficient (more informative than accuracy for
        # imbalanced classes — important here since EV events are rare)
        denom_mcc = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = self._safe_div(tp * tn - fp * fn, denom_mcc, default=0.0)

        return {
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Precision':   round(precision,   4),
            'Recall':      round(recall,      4),
            'F1':          round(f1,          4),
            'Accuracy':    round(accuracy,    4),
            'FPR':         round(fpr,         4),
            'FNR':         round(fnr,         4),
            'Specificity': round(specificity, 4),
            'MCC':         round(mcc,         4),
        }

    def compute(self) -> dict:
        """
        Compute and return the full metrics dictionary.

        Structure:
            {
              'frames_processed': int,
              'session_seconds':  float,
              'audio':   { precision, recall, f1, accuracy, fpr, fnr, mcc, ... },
              'video':   { ... },
              'fusion':  { ... },
              'fusion_vs_audio': { fpr_reduction, fnr_reduction, f1_gain },
              'fusion_vs_video': { fpr_reduction, fnr_reduction, f1_gain },
              'latency': {
                  'mean_detection_frames':  float,
                  'median_detection_frames': float,
                  'missed_events':          int,
                  'mean_clearance_frames':  float,
              },
              'stability': {
                  'audio':  { ssi, toggle_rate_per_100 },
                  'video':  { ... },
                  'fusion': { ... },
              },
              'confidence': {
                  'mean_conf_v': float,
                  'mean_conf_a': float,
              },
            }
        """
        n = self._total_frames
        elapsed = time.time() - self._start_time

        audio_m  = self._metrics_for('audio')
        video_m  = self._metrics_for('video')
        fusion_m = self._metrics_for('fusion')

        def gain(baseline, improved, key, lower_is_better=False):
            b = baseline[key]
            i = improved[key]
            if lower_is_better:
                return round(b - i, 4)   # positive = improvement
            return round(i - b, 4)

        fusion_vs_audio = {
            'fpr_reduction': gain(audio_m,  fusion_m, 'FPR', lower_is_better=True),
            'fnr_reduction': gain(audio_m,  fusion_m, 'FNR', lower_is_better=True),
            'f1_gain':       gain(audio_m,  fusion_m, 'F1'),
            'mcc_gain':      gain(audio_m,  fusion_m, 'MCC'),
        }
        fusion_vs_video = {
            'fpr_reduction': gain(video_m,  fusion_m, 'FPR', lower_is_better=True),
            'fnr_reduction': gain(video_m,  fusion_m, 'FNR', lower_is_better=True),
            'f1_gain':       gain(video_m,  fusion_m, 'F1'),
            'mcc_gain':      gain(video_m,  fusion_m, 'MCC'),
        }

        # Detection latency (exclude inf = missed event)
        finite_lats = [x for x in self._latencies if math.isfinite(x)]
        missed      = sum(1 for x in self._latencies if not math.isfinite(x))
        mean_lat    = round(sum(finite_lats) / len(finite_lats), 2) if finite_lats else 0.0
        sorted_lats = sorted(finite_lats)
        med_lat     = round(sorted_lats[len(sorted_lats) // 2], 2) if sorted_lats else 0.0

        # Clearance latency
        finite_clats = [x for x in self._clearance_latencies if math.isfinite(x)]
        mean_clat    = round(sum(finite_clats) / len(finite_clats), 2) if finite_clats else 0.0

        # Stability
        def stability(key):
            total_transitions = n - 1 if n > 1 else 1
            ssi = self._safe_div(self._stable[key], total_transitions)
            toggle_rate = self._safe_div(self._toggles[key] * 100, n)
            return {
                'SSI':                  round(ssi, 4),
                'toggle_rate_per_100':  round(toggle_rate, 4),
                'total_toggles':        self._toggles[key],
            }

        return {
            'frames_processed': n,
            'session_seconds':  round(elapsed, 1),
            'audio':   audio_m,
            'video':   video_m,
            'fusion':  fusion_m,
            'fusion_vs_audio': fusion_vs_audio,
            'fusion_vs_video': fusion_vs_video,
            'latency': {
                'mean_detection_frames':   mean_lat,
                'median_detection_frames': med_lat,
                'missed_events':           missed,
                'mean_clearance_frames':   mean_clat,
            },
            'stability': {
                'audio':  stability('audio'),
                'video':  stability('video'),
                'fusion': stability('fusion'),
            },
            'confidence': {
                'mean_conf_v': round(self._safe_div(self._conf_v_sum, self._conf_v_n), 4),
                'mean_conf_a': round(self._safe_div(self._conf_a_sum, self._conf_a_n), 4),
            },
        }

    # ──────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────

    def export_csv(self, path: str):
        """
        Write a flat CSV summary of all computed metrics to `path`.
        Suitable for direct import into Excel / LaTeX tables.
        """
        import csv
        m = self.compute()

        rows = [
            ['Section', 'Metric', 'Value'],
            # Session info
            ['Session', 'frames_processed', m['frames_processed']],
            ['Session', 'session_seconds',  m['session_seconds']],
        ]
        for modality in ('audio', 'video', 'fusion'):
            for k, v in m[modality].items():
                rows.append([modality.capitalize(), k, v])
        for label, section in [('Fusion vs Audio', 'fusion_vs_audio'),
                                ('Fusion vs Video', 'fusion_vs_video')]:
            for k, v in m[section].items():
                rows.append([label, k, v])
        for k, v in m['latency'].items():
            rows.append(['Latency', k, v])
        for modality in ('audio', 'video', 'fusion'):
            for k, v in m['stability'][modality].items():
                rows.append([f'Stability_{modality}', k, v])
        for k, v in m['confidence'].items():
            rows.append(['Confidence', k, v])

        with open(path, 'w', newline='') as f:
            csv.writer(f).writerows(rows)

    def summary_text(self) -> str:
        """Return a compact plain-text summary for display in the GUI log."""
        m = self.compute()
        f = m['fusion']
        lat = m['latency']
        stab = m['stability']['fusion']
        va = m['fusion_vs_audio']
        vv = m['fusion_vs_video']
        lines = [
            f"=== Fusion Evaluation  ({m['frames_processed']} frames, {m['session_seconds']}s) ===",
            f"  Precision : {f['Precision']:.4f}   Recall : {f['Recall']:.4f}   F1 : {f['F1']:.4f}",
            f"  Accuracy  : {f['Accuracy']:.4f}   MCC    : {f['MCC']:.4f}",
            f"  FPR       : {f['FPR']:.4f}   FNR    : {f['FNR']:.4f}",
            f"  TP={f['TP']}  FP={f['FP']}  TN={f['TN']}  FN={f['FN']}",
            f"  Latency   : {lat['mean_detection_frames']} frames (mean)  "
            f"{lat['median_detection_frames']} (median)  missed={lat['missed_events']}",
            f"  Clearance : {lat['mean_clearance_frames']} frames (mean)",
            f"  Stability : SSI={stab['SSI']:.4f}  toggles/100={stab['toggle_rate_per_100']:.2f}",
            f"  vs Audio  : ΔFPR={va['fpr_reduction']:+.4f}  ΔFNR={va['fnr_reduction']:+.4f}  ΔF1={va['f1_gain']:+.4f}",
            f"  vs Video  : ΔFPR={vv['fpr_reduction']:+.4f}  ΔFNR={vv['fnr_reduction']:+.4f}  ΔF1={vv['f1_gain']:+.4f}",
        ]
        return '\n'.join(lines)
