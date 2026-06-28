# audio_module.py
"""
Siren detection using PANNs CNN14.

Modes
-----
'live'  — microphone capture (sounddevice), unchanged from v1.
'video' — audio is extracted automatically from the same video file that
          VideoModule is reading.  No separate audio file needed.
          The full audio track is decoded once on start() using ffmpeg,
          then each call to get_confidence_for_frame(frame_idx, fps) returns
          the CNN14 siren probability for the audio window that corresponds
          to that exact video frame.
'file'  — explicit separate audio file, unchanged from v1.

Frame-synchronised design ('video' mode)
-----------------------------------------
VideoModule reads frame N at time  t = N / fps  seconds.
AudioModule serves the audio clip centred on that same time:
    clip_start = max(0,  t - clip_duration/2)
    clip_end   = clip_start + clip_duration
This is a half-second look-behind / look-ahead window so the CNN14 model
always has enough context (it needs ~1 s minimum).

CNN14 inference is run in a background thread so the GUI never blocks.
The result is cached per-frame and returned instantly on the next read.
"""

import os
import time
import subprocess
import tempfile
import threading
import numpy as np
import librosa
from panns_inference import AudioTagging


DEFAULT_SIREN_INDICES = [82, 87, 517]
DEFAULT_THRESHOLD     = 0.3
CLIP_DURATION         = 2.0   # seconds of audio analysed per frame window


class AudioModule:
    """
    Siren detection module.  Supports live mic, explicit audio file,
    and automatic audio extraction from a video file.

    Public attributes (updated after each analysis):
        last_confidence   (float) : max siren-class score (0-1).
        last_class_scores (dict)  : {index: score} for watched indices.
    """

    def __init__(
        self,
        mode:          str   = "live",
        audio_file:    str   = None,
        video_file:    str   = None,      # NEW — path to video for audio extraction
        device:        str   = "cpu",
        siren_indices: list  = None,
        threshold:     float = DEFAULT_THRESHOLD,
    ):
        """
        Args:
            mode          : 'live' | 'video' | 'file'
            audio_file    : path to audio file (mode='file' only)
            video_file    : path to video file (mode='video' only)
            device        : 'cpu' or 'cuda'
            siren_indices : AudioSet class indices to watch
            threshold     : minimum score to flag siren event
        """
        self.mode          = mode
        self.audio_file    = audio_file
        self.video_file    = video_file
        self.device        = device
        self.siren_indices = siren_indices or list(DEFAULT_SIREN_INDICES)
        self.threshold     = threshold
        self.clip_duration = CLIP_DURATION
        self.fs            = 32000      # CNN14 required sample rate

        # Load PANNs model
        self.model = AudioTagging(checkpoint_path=None, device=self.device)

        # Threading
        self.running     = False
        self.audio_event = False
        self.lock        = threading.Lock()
        self.thread      = None

        # Per-analysis metadata
        self.last_confidence:   float = 0.0
        self.last_class_scores: dict  = {idx: 0.0 for idx in self.siren_indices}

        # ── Video-mode state ──────────────────────────
        # Full audio array pre-loaded on start()
        self._audio_array:  np.ndarray = None   # shape (n_samples,)
        self._audio_loaded: bool       = False

        # Background inference cache: frame_idx -> (confidence, scores, event)
        self._cache:       dict        = {}
        self._pending:     set         = set()  # frames currently being inferred
        self._infer_queue: list        = []     # (frame_idx, clip) to process
        self._infer_lock   = threading.Lock()

    # ──────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────

    def start(self):
        """Initialise and begin processing."""
        self.running = True
        if self.mode == "video":
            self._load_audio_from_video()
            # Start background inference worker
            self.thread = threading.Thread(
                target=self._inference_worker, daemon=True
            )
            self.thread.start()
        elif self.mode == "live":
            self.thread = threading.Thread(
                target=self._record_loop, daemon=True
            )
            self.thread.start()
        else:  # 'file'
            self.thread = threading.Thread(
                target=self._process_file, daemon=True
            )
            self.thread.start()

    def stop(self):
        """Stop all processing and clean up temp files."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self._audio_array = None
        self._cache.clear()

    def get_audio_event(self) -> bool:
        """
        Legacy API used by main.py v1.
        For 'live' and 'file' modes: returns True if siren detected since last call.
        For 'video' mode: always returns False — use get_confidence_for_frame() instead.
        """
        with self.lock:
            event = self.audio_event
            self.audio_event = False
        return event

    def get_confidence_for_frame(self, frame_idx: int, fps: float) -> float:
        """
        VIDEO MODE ONLY.

        Return the CNN14 siren confidence for the audio window aligned to
        frame `frame_idx`.  Submits a background inference job if the result
        is not yet cached, and returns the last known value immediately
        (non-blocking).

        Args:
            frame_idx : current video frame number (0-based)
            fps       : video frames-per-second

        Returns:
            Siren confidence score 0.0–1.0 (0.0 if not yet computed)
        """
        if not self._audio_loaded:
            return 0.0

        # Check cache first
        with self._infer_lock:
            if frame_idx in self._cache:
                conf, scores, event = self._cache[frame_idx]
                self.last_confidence   = conf
                self.last_class_scores = scores
                with self.lock:
                    if event:
                        self.audio_event = True
                return conf

            # Not cached — enqueue for inference if not already pending
            if frame_idx not in self._pending:
                clip = self._extract_clip(frame_idx, fps)
                if clip is not None:
                    self._pending.add(frame_idx)
                    self._infer_queue.append((frame_idx, clip))

        # Return last known value while inference runs in background
        return self.last_confidence

    # ──────────────────────────────────────────────────
    # Video-mode: audio extraction
    # ──────────────────────────────────────────────────

    def _find_ffmpeg(self) -> str:
        """
        Locate the ffmpeg executable on Windows or Linux/macOS.
        Checks: PATH, common Windows install locations.
        Returns the executable path string, or None if not found.
        """
        import shutil
        # 1. Try PATH first (works if user added ffmpeg to PATH correctly)
        path = shutil.which("ffmpeg")
        if path:
            return path

        # 2. Common Windows install paths
        win_candidates = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        ]
        for p in win_candidates:
            if os.path.isfile(p):
                return p

        return None

    def _load_audio_from_video(self):
        """
        Extract the full audio track from self.video_file.

        Strategy (tried in order):
          1. ffmpeg via PATH or common install locations
          2. moviepy (pip install moviepy) as fallback
          3. Graceful degradation — audio confidence stays 0.0,
             video-only fusion continues to work.
        """
        if not self.video_file or not os.path.isfile(self.video_file):
            print(f"[AudioModule] video_file not found: {self.video_file}")
            return

        # ── Method 1: ffmpeg ──────────────────────────
        ffmpeg_exe = self._find_ffmpeg()
        if ffmpeg_exe:
            self._extract_via_ffmpeg(ffmpeg_exe)
            if self._audio_loaded:
                return
        else:
            print("[AudioModule] ffmpeg not found in PATH or common locations.")
            print("  → Install ffmpeg: https://www.gyan.dev/ffmpeg/builds/")
            print("  → Extract to C:\\ffmpeg, add C:\\ffmpeg\\bin to system PATH.")
            print("  → Trying moviepy fallback...")

        # ── Method 2: moviepy fallback ────────────────
        self._extract_via_moviepy()
        if self._audio_loaded:
            return

        # ── Method 3: graceful degradation ───────────
        print("[AudioModule] WARNING: Audio extraction failed.")
        print("  System will run in VIDEO-ONLY mode (audio confidence = 0.0).")
        print("  Fusion OR-mode will still confirm EVs via visual detection alone.")

    def _extract_via_ffmpeg(self, ffmpeg_exe: str):
        """Extract audio using ffmpeg executable."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            cmd = [
                ffmpeg_exe, "-y",
                "-i", self.video_file,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(self.fs),
                "-ac", "1",
                tmp_path,
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                print("[AudioModule] ffmpeg error:")
                print(result.stderr.decode(errors="replace")[-400:])
                return

            audio, _ = librosa.load(tmp_path, sr=self.fs, mono=True)
            self._audio_array  = audio
            self._audio_loaded = True
            duration = len(audio) / self.fs
            print(f"[AudioModule] Audio extracted via ffmpeg: "
                  f"{duration:.1f}s ({len(audio)} samples @ {self.fs} Hz)")
        except Exception as e:
            print(f"[AudioModule] ffmpeg extraction exception: {e}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _extract_via_moviepy(self):
        """Extract audio using moviepy as a fallback."""
        try:
            from moviepy.editor import VideoFileClip
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                clip = VideoFileClip(self.video_file)
                if clip.audio is None:
                    print("[AudioModule] moviepy: video has no audio track.")
                    clip.close()
                    return
                clip.audio.write_audiofile(
                    tmp_path, fps=self.fs, nbytes=2, codec="pcm_s16le",
                    logger=None,
                )
                clip.close()
                audio, _ = librosa.load(tmp_path, sr=self.fs, mono=True)
                self._audio_array  = audio
                self._audio_loaded = True
                duration = len(audio) / self.fs
                print(f"[AudioModule] Audio extracted via moviepy: "
                      f"{duration:.1f}s ({len(audio)} samples @ {self.fs} Hz)")
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        except ImportError:
            print("[AudioModule] moviepy not installed.")
            print("  → Run:  pip install moviepy")

    def _extract_clip(self, frame_idx: int, fps: float) -> np.ndarray:
        """
        Slice the pre-loaded audio array to the window aligned with frame_idx.
        Returns None if audio is not loaded or frame is out of range.
        """
        if self._audio_array is None:
            return None

        t_center   = frame_idx / fps
        t_start    = max(0.0, t_center - self.clip_duration / 2)
        t_end      = t_start + self.clip_duration

        s_start = int(t_start * self.fs)
        s_end   = int(t_end   * self.fs)

        if s_start >= len(self._audio_array):
            return None

        s_end = min(s_end, len(self._audio_array))
        clip  = self._audio_array[s_start:s_end]

        # Pad with silence if clip is shorter than required
        required = int(self.clip_duration * self.fs)
        if len(clip) < required:
            clip = np.pad(clip, (0, required - len(clip)))

        return clip

    # ──────────────────────────────────────────────────
    # Background inference worker (video mode)
    # ──────────────────────────────────────────────────

    def _inference_worker(self):
        """Process queued (frame_idx, clip) pairs in the background thread."""
        while self.running:
            job = None
            with self._infer_lock:
                if self._infer_queue:
                    job = self._infer_queue.pop(0)
            if job is None:
                time.sleep(0.01)
                continue

            frame_idx, clip = job
            conf, scores, event = self._run_inference(clip)

            with self._infer_lock:
                self._cache[frame_idx] = (conf, scores, event)
                self._pending.discard(frame_idx)

            # Keep cache size bounded (last 300 frames ~ 10 s at 30 fps)
            with self._infer_lock:
                if len(self._cache) > 300:
                    oldest = min(self._cache.keys())
                    del self._cache[oldest]

    # ──────────────────────────────────────────────────
    # Shared inference logic
    # ──────────────────────────────────────────────────

    def _run_inference(self, audio_clip: np.ndarray):
        """Run CNN14 on a clip. Returns (max_score, scores_dict, event_bool)."""
        audio_input = np.expand_dims(audio_clip, axis=0)  # (1, samples)
        clipwise_output, _ = self.model.inference(audio_input)

        scores    = {}
        max_score = 0.0
        detected  = False

        for idx in self.siren_indices:
            if idx < clipwise_output.shape[1]:
                score = float(clipwise_output[0][idx])
                scores[idx] = score
                if score > max_score:
                    max_score = score
                if score >= self.threshold:
                    detected = True
            else:
                scores[idx] = 0.0

        return max_score, scores, detected

    def _analyze_audio(self, audio_clip: np.ndarray):
        """Shared helper for live/file modes — runs inference and updates state."""
        conf, scores, detected = self._run_inference(audio_clip)
        with self.lock:
            self.last_confidence   = conf
            self.last_class_scores = scores
            if detected:
                self.audio_event = True

    # ──────────────────────────────────────────────────
    # Live mic mode
    # ──────────────────────────────────────────────────

    def _record_loop(self):
        import sounddevice as sd
        while self.running:
            recording = sd.rec(
                int(self.clip_duration * self.fs),
                samplerate=self.fs, channels=1, dtype="float32",
            )
            sd.wait()
            self._analyze_audio(recording.flatten())
            time.sleep(0.1)

    # ──────────────────────────────────────────────────
    # File mode (explicit separate audio file)
    # ──────────────────────────────────────────────────

    def _process_file(self):
        if self.audio_file is None:
            return
        audio, _ = librosa.load(self.audio_file, sr=self.fs, mono=True)
        offset = 0
        step   = int(self.clip_duration * self.fs)
        while offset < len(audio) and self.running:
            end = min(offset + step, len(audio))
            self._analyze_audio(audio[offset:end])
            offset = end
        self.running = False
