# fusion_module.py
"""
Fusion module implementing Algorithm 1 from the paper:

  "A Multi-Modal Framework for Emergency Vehicle Prioritization
   Using Visual Detection and Acoustic Classification"

Algorithm 1 — Safety-Critical Fusion:
  Input : Conf_v (YOLOv8 confidence), Conf_a (CNN14 siren probability)
  Params: τ_v, τ_a (thresholds), N (queue length), K (activation threshold)

  1.  Queue <- [0] x N          # rolling binary vote buffer
  2.  While running:
  3.    v(t) = 1  if  Conf_v > tau_v,  else 0
  4.    a(t) = 1  if  Conf_a > tau_a,  else 0
  5a.   OR  mode: Vote = 1  if  v(t) OR a(t)
  5b.   AND mode: Vote = 1  if  v(t) AND a(t)
  6.    Queue.enqueue(Vote); Queue.dequeue()    # sliding window
  7.    If Sum(Queue) >= K:  Signal = GREEN (priority)
  8.    Else               :  Signal = RED   (normal)

Temporal Smoothing formula (Section III-C-2):
  Dsmooth(t) = 1  if  sum_{i=0}^{N-1} Dinst(t-i) >= K,  else 0

This acts as a digital hysteresis filter: a single stray detection does
not trigger priority, and a single missed frame does not cancel it.

Default configuration matches paper:
  - OR logic  (high sensitivity — preferred for safety-critical use)
  - N = 10 frames
  - K = 4  (40% of the window must be positive)
  - tau_v = 0.50,  tau_a = 0.30
"""

from collections import deque
from typing import Optional, Tuple


class FusionModule:
    """
    Sliding-window vote-based fusion engine.

    Each call to push_frame(conf_v, conf_a, direction) adds one vote to the
    queue and returns the smoothed decision immediately.

    check_emergency() returns the latest smoothed result without consuming
    a new frame — safe to call from the GUI tick without double-counting.
    """

    def __init__(
        self,
        mode:      str   = "OR",   # paper default: OR (high sensitivity)
        n_frames:  int   = 10,     # queue length N
        k_thresh:  int   = 4,      # activation threshold K
        tau_v:     float = 0.50,   # tau_vis  — YOLOv8 confidence threshold
        tau_a:     float = 0.30,   # tau_aud  — CNN14 siren probability threshold
        # Legacy param kept for backwards-compat; ignored internally.
        window_sec: int  = 10,
    ):
        self.mode     = mode.upper()
        self.n_frames = n_frames
        self.k_thresh = k_thresh
        self.tau_v    = tau_v
        self.tau_a    = tau_a

        # Sliding vote queue initialised to all-zero (no detections)
        self._queue: deque = deque([0] * self.n_frames, maxlen=self.n_frames)

        # Latest smoothed result (updated by push_frame)
        self._confirmed: bool          = False
        self._direction: Optional[str] = None

        # Last raw inputs stored so check_emergency() can be polled freely
        self._last_conf_v:    float          = 0.0
        self._last_conf_a:    float          = 0.0
        self._last_direction: Optional[str]  = None

        # Diagnostic counters
        self.total_votes_cast:    int = 0
        self.total_confirmations: int = 0

    # ──────────────────────────────────────────────────
    # Configuration
    # ──────────────────────────────────────────────────

    def set_mode(self, mode: str):
        """
        Switch fusion logic at runtime.
        'OR'  -> high sensitivity (paper default / recommended for safety).
        'AND' -> high precision   (fewer false positives, may miss detections).
        """
        mode = (mode or "").upper()
        if mode not in ("AND", "OR"):
            raise ValueError("Fusion mode must be 'AND' or 'OR'.")
        self.mode = mode

    def set_queue_params(self, n_frames: int = None, k_thresh: int = None):
        """Resize queue and/or change K. Queue resets when n_frames changes."""
        if n_frames is not None and n_frames != self.n_frames:
            self.n_frames = n_frames
            self._queue = deque([0] * self.n_frames, maxlen=self.n_frames)
        if k_thresh is not None:
            self.k_thresh = k_thresh

    # ──────────────────────────────────────────────────
    # Primary API  (called from _tick in main.py)
    # ──────────────────────────────────────────────────

    def push_frame(
        self,
        conf_v:    float,           # YOLOv8 max detection confidence (0-1)
        conf_a:    float,           # CNN14 siren probability (0-1)
        direction: Optional[str],   # approach direction from video tracker
    ) -> Tuple[bool, Optional[str]]:
        """
        Ingest one frame's detections, advance the vote queue, and return
        the smoothed decision (Algorithm 1, lines 3-8).

        Returns:
            (confirmed, direction)
        """
        self._last_conf_v    = conf_v
        self._last_conf_a    = conf_a
        self._last_direction = direction

        # Instantaneous binary signals  (lines 3-4)
        v = 1 if conf_v > self.tau_v else 0
        a = 1 if conf_a > self.tau_a else 0

        # Combine according to fusion mode  (lines 5a / 5b)
        if self.mode == "OR":
            vote = 1 if (v or a) else 0       # Dinst = v(t) OR a(t)
        else:  # AND
            vote = 1 if (v and a) else 0      # Dinst = v(t) AND a(t)

        # Sliding queue  (line 6)
        self._queue.append(vote)
        self.total_votes_cast += 1

        # Temporal smoothing  (lines 7-8)
        queue_sum = sum(self._queue)
        self._confirmed = (queue_sum >= self.k_thresh)

        if self._confirmed:
            self._direction = direction      # direction from video modality
            self.total_confirmations += 1
        else:
            self._direction = None

        return self._confirmed, self._direction

    def check_emergency(self) -> Tuple[bool, Optional[str]]:
        """
        Return the latest smoothed result without advancing the queue.
        Backwards-compatible with the v1 API used in main.py.
        """
        return self._confirmed, self._direction

    # ──────────────────────────────────────────────────
    # Legacy event-push API  (backwards-compat shims)
    # ──────────────────────────────────────────────────

    def add_audio_event(self):
        """Legacy: push a frame treating audio confidence as certain (1.0)."""
        self.push_frame(
            conf_v=self._last_conf_v,
            conf_a=1.0,
            direction=self._last_direction,
        )

    def add_video_event(self, direction: str):
        """Legacy: push a frame treating video confidence as certain (1.0)."""
        self.push_frame(
            conf_v=1.0,
            conf_a=self._last_conf_a,
            direction=direction,
        )

    # ──────────────────────────────────────────────────
    # Diagnostics (read by dashboard)
    # ──────────────────────────────────────────────────

    @property
    def queue_sum(self) -> int:
        """Current sum of the vote queue (0 ... N)."""
        return sum(self._queue)

    @property
    def queue_fill(self) -> float:
        """Queue occupancy as a fraction 0.0 - 1.0."""
        return self.queue_sum / self.n_frames if self.n_frames else 0.0

    def reset(self):
        """Clear vote queue and reset confirmed state."""
        self._queue = deque([0] * self.n_frames, maxlen=self.n_frames)
        self._confirmed = False
        self._direction = None
