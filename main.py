# main.py

import sys
import time
import datetime
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QCheckBox, QFileDialog, QLineEdit, QComboBox, QFrame, QSizePolicy,
    QSpinBox, QTextEdit, QGridLayout, QProgressBar, QGroupBox, QSplitter,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QPixmap, QPainter, QPen, QBrush
from audio_module import AudioModule
from video_module import VideoModule
from fusion_module import FusionModule
from signal_module import SignalModule
from eval_module import EvalModule
from utils.utils import convert_cv_qt

# ──────────────────────────────────────────────
# Dark-theme palette
# ──────────────────────────────────────────────
DARK_BG     = "#0a0e1a"
SURFACE     = "#111827"
SURFACE2    = "#1a2235"
BORDER      = "#1e3a5f"
ACCENT      = "#00d4ff"
ACCENT2     = "#ff6b35"
GREEN       = "#00ff88"
RED_COLOR   = "#ff3355"
YELLOW      = "#ffd700"
MUTED       = "#4a6080"
TEXT        = "#e8f4f8"

STYLESHEET = f"""
QWidget {{
    background-color: {DARK_BG};
    color: {TEXT};
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 10px;
    padding: 6px 8px;
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 2px;
    color: {MUTED};
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
}}
QPushButton {{
    background-color: {SURFACE2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 11px;
    letter-spacing: 1px;
}}
QPushButton:hover {{
    background-color: {BORDER};
    border-color: {ACCENT};
    color: {ACCENT};
}}
QPushButton:disabled {{
    color: {MUTED};
    border-color: {SURFACE2};
}}
QPushButton#btn_start {{
    background-color: rgba(0,255,136,0.12);
    border-color: {GREEN};
    color: {GREEN};
    font-weight: bold;
}}
QPushButton#btn_stop {{
    background-color: rgba(255,51,85,0.12);
    border-color: {RED_COLOR};
    color: {RED_COLOR};
    font-weight: bold;
}}
QComboBox, QSpinBox, QLineEdit {{
    background-color: {SURFACE2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    color: {TEXT};
    selection-background-color: {BORDER};
}}
QCheckBox {{
    spacing: 6px;
    font-size: 12px;
}}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {SURFACE2};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QProgressBar {{
    background-color: {SURFACE2};
    border: 1px solid {BORDER};
    border-radius: 3px;
    height: 6px;
    text-align: right;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}
QTextEdit {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    font-family: "Courier New", monospace;
    font-size: 11px;
    color: {TEXT};
}}
QLabel#header_title {{
    font-size: 16px;
    font-weight: bold;
    letter-spacing: 3px;
    color: {ACCENT};
}}
QLabel#header_sub {{
    font-size: 10px;
    color: {MUTED};
    font-family: "Courier New", monospace;
    letter-spacing: 1px;
}}
QLabel#stat_num {{
    font-size: 22px;
    font-weight: bold;
    font-family: "Courier New", monospace;
    color: {ACCENT};
}}
QLabel#stat_lbl {{
    font-size: 9px;
    color: {MUTED};
    letter-spacing: 1px;
}}
QLabel#badge_on {{
    background-color: rgba(0,255,136,0.15);
    color: {GREEN};
    border: 1px solid rgba(0,255,136,0.35);
    border-radius: 3px;
    padding: 2px 8px;
    font-family: "Courier New", monospace;
    font-size: 10px;
    font-weight: bold;
}}
QLabel#badge_off {{
    background-color: rgba(255,51,85,0.08);
    color: {MUTED};
    border: 1px solid rgba(255,51,85,0.18);
    border-radius: 3px;
    padding: 2px 8px;
    font-family: "Courier New", monospace;
    font-size: 10px;
}}
QLabel#badge_warn {{
    background-color: rgba(255,215,0,0.12);
    color: {YELLOW};
    border: 1px solid rgba(255,215,0,0.35);
    border-radius: 3px;
    padding: 2px 8px;
    font-family: "Courier New", monospace;
    font-size: 10px;
    font-weight: bold;
}}
QFrame#separator {{
    color: {BORDER};
}}
QSplitter::handle {{
    background: {BORDER};
    width: 1px;
    height: 1px;
}}
"""


# ──────────────────────────────────────────────
# Intersection widget  (SVG-style painted in Qt)
# ──────────────────────────────────────────────
class IntersectionWidget(QWidget):
    """
    Paints a top-down 4-way intersection with animated traffic light heads.
    Call set_state(ns_color, ew_color) where color in {'green','yellow','red'}.
    Call set_emergency(True/False, direction) to show the EV indicator.
    """
    COLOR_MAP = {
        "green":  QColor(0, 255, 136),
        "yellow": QColor(255, 215, 0),
        "red":    QColor(255, 51, 85),
        "dim":    QColor(26, 34, 53),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 300)
        self.ns_color = "green"
        self.ew_color = "red"
        self.emergency = False
        self.direction = None
        self._anim_step = 0
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(80)
        self._anim_timer.timeout.connect(self._tick_anim)
        self._anim_timer.start()

    def set_state(self, ns_color: str, ew_color: str):
        self.ns_color = ns_color
        self.ew_color = ew_color
        self.update()

    def set_emergency(self, active: bool, direction=None):
        self.emergency = active
        self.direction = direction
        self.update()

    def _tick_anim(self):
        if self.emergency:
            self._anim_step = (self._anim_step + 1) % 20
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        cx, cy = W // 2, H // 2
        road_w = W // 4

        # Background
        p.fillRect(0, 0, W, H, QColor(10, 14, 26))

        # Roads
        p.fillRect(0, cy - road_w // 2, W, road_w, QColor(26, 34, 53))
        p.fillRect(cx - road_w // 2, 0, road_w, H, QColor(26, 34, 53))

        # Center box
        p.fillRect(cx - road_w // 2, cy - road_w // 2, road_w, road_w, QColor(34, 46, 66))

        # Lane dashes
        pen = QPen(QColor(42, 63, 95))
        pen.setWidth(1)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawLine(cx, 0, cx, cy - road_w // 2)
        p.drawLine(cx, cy + road_w // 2, cx, H)
        p.drawLine(0, cy, cx - road_w // 2, cy)
        p.drawLine(cx + road_w // 2, cy, W, cy)

        # Emergency ring
        if self.emergency:
            alpha = int(abs(10 - self._anim_step) / 10 * 180) + 30
            ring_color = QColor(255, 107, 53, alpha)
            pen2 = QPen(ring_color)
            pen2.setWidth(2)
            p.setPen(pen2)
            p.setBrush(Qt.NoBrush)
            r = int(road_w * 0.7)
            p.drawEllipse(cx - r, cy - r, r * 2, r * 2)

        # Direction arrow
        if self.emergency and self.direction:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(255, 107, 53)))
            arr = road_w // 3
            d = self.direction.upper()
            if d == 'N':
                pts = [(cx, cy - arr), (cx - arr // 2, cy), (cx + arr // 2, cy)]
            elif d == 'S':
                pts = [(cx, cy + arr), (cx - arr // 2, cy), (cx + arr // 2, cy)]
            elif d == 'E':
                pts = [(cx + arr, cy), (cx, cy - arr // 2), (cx, cy + arr // 2)]
            else:  # W
                pts = [(cx - arr, cy), (cx, cy - arr // 2), (cx, cy + arr // 2)]
            from PyQt5.QtGui import QPolygon
            from PyQt5.QtCore import QPoint
            poly = QPolygon([QPoint(x, y) for x, y in pts])
            p.drawPolygon(poly)

        # Traffic light heads
        self._draw_light_head(p, cx - 14, cy - road_w // 2 - 52, 'V', self.ns_color)   # N
        self._draw_light_head(p, cx + 4,  cy + road_w // 2 + 4,  'V', self.ns_color)   # S
        self._draw_light_head(p, cx + road_w // 2 + 4,  cy - 11, 'H', self.ew_color)   # E
        self._draw_light_head(p, cx - road_w // 2 - 52, cy + 1,  'H', self.ew_color)   # W

        # Direction labels
        p.setPen(QPen(QColor(74, 96, 128)))
        p.setFont(QFont("Courier New", 9))
        p.drawText(cx - 5, 14, "N")
        p.drawText(cx - 5, H - 4, "S")
        p.drawText(W - 12, cy + 4, "E")
        p.drawText(4, cy + 4, "W")

        p.end()

    def _draw_light_head(self, p, x, y, orientation, active_color):
        """Draw a 3-lamp traffic light head (red/yellow/green)."""
        if orientation == 'V':
            bw, bh = 20, 52
        else:
            bw, bh = 52, 20

        # Housing
        p.setBrush(QBrush(QColor(13, 21, 32)))
        p.setPen(QPen(QColor(30, 58, 95)))
        p.drawRoundedRect(x, y, bw, bh, 3, 3)

        colors = ["red", "yellow", "green"]
        for i, lamp in enumerate(colors):
            lit = (lamp == active_color)
            c = self.COLOR_MAP[lamp] if lit else self.COLOR_MAP["dim"]
            if lit:
                glow = QColor(c.red(), c.green(), c.blue(), 60)
                p.setBrush(QBrush(glow))
                p.setPen(Qt.NoPen)
                if orientation == 'V':
                    p.drawEllipse(x + 2, y + i * 16 + 2, 16, 16)
                else:
                    p.drawEllipse(x + i * 16 + 2, y + 2, 16, 16)
            p.setBrush(QBrush(c))
            p.setPen(QPen(QColor(0, 0, 0, 80)))
            if orientation == 'V':
                p.drawEllipse(x + 3, y + i * 16 + 3, 14, 14)
            else:
                p.drawEllipse(x + i * 16 + 3, y + 3, 14, 14)


# ──────────────────────────────────────────────
# Confidence bar row
# ──────────────────────────────────────────────
class ConfBar(QWidget):
    def __init__(self, label: str, accent: str, parent=None):
        super().__init__(parent)
        self._accent = accent
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        row = QHBoxLayout()
        self.lbl = QLabel(label)
        self.lbl.setStyleSheet(f"color:{MUTED}; font-size:10px; letter-spacing:1px;")
        self.val = QLabel("0%")
        self.val.setStyleSheet(f"color:{accent}; font-family:'Courier New'; font-size:13px; font-weight:bold;")
        row.addWidget(self.lbl)
        row.addStretch()
        row.addWidget(self.val)
        layout.addLayout(row)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(5)
        self.bar.setStyleSheet(
            f"QProgressBar{{background:{SURFACE2};border:none;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:{accent};border-radius:2px;}}"
        )
        layout.addWidget(self.bar)

    def set_value(self, pct: float):
        v = max(0, min(100, int(pct)))
        self.val.setText(f"{v}%")
        self.bar.setValue(v)


# ──────────────────────────────────────────────
# Stat tile
# ──────────────────────────────────────────────
def make_stat_tile(num_text: str, label_text: str):
    frame = QFrame()
    frame.setStyleSheet(
        f"QFrame{{background:{SURFACE2};border:1px solid {BORDER};border-radius:4px;}}"
    )
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(10, 8, 10, 8)
    lay.setSpacing(2)
    num = QLabel(num_text)
    num.setObjectName("stat_num")
    lbl = QLabel(label_text.upper())
    lbl.setObjectName("stat_lbl")
    lay.addWidget(num)
    lay.addWidget(lbl)
    return frame, num


# ──────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────
class MainWindow(QWidget):
    """
    Command-center GUI for the Emergency Vehicle Prioritization System.

    Left panel  : Audio detection confidence, fusion config, input source.
    Center      : Live video feed + painted intersection diagram.
    Right panel : Video detection details, direction indicator, event log.
    Bottom bar  : Alert status + stats row.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EV Priority Control System")
        self.resize(1300, 820)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(STYLESHEET)

        # ── Runtime state ──────────────────────────────
        self._start_time = None
        self._total_audio = 0
        self._total_video = 0
        self._total_confirm = 0
        self._last_direction = None

        # ── Modules (created on Start) ─────────────────
        self.audio_module = None
        self.video_module = None
        # Paper default is OR (high sensitivity) — Section III-C-1
        self.fusion_module = FusionModule(mode="OR", n_frames=10, k_thresh=4,
                                          tau_v=0.50, tau_a=0.30)
        # signal_module is wired to the intersection widget, not QLabel circles
        self._light_labels = {k: QLabel() for k in ('N', 'S', 'E', 'W')}
        self.signal_module = SignalModule(self._light_labels)

        self._build_ui()

        # ── Evaluation module ──────────────────────────
        self.evaluator = EvalModule()
        self._gt_active: bool = False   # operator-toggled ground truth

        # ── Tick timer ─────────────────────────────────
        self.timer = QTimer()
        self.timer.setInterval(100)   # 10 FPS
        self.timer.timeout.connect(self._tick)

        # ── Clock timer ────────────────────────────────
        self._clock_timer = QTimer()
        self._clock_timer.setInterval(1000)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start()
        self._update_clock()

    # ──────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        root.addWidget(self._build_toolbar(), 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([260, 720, 260])
        root.addWidget(splitter, 1)

        root.addWidget(self._build_bottom_bar())

    # ── Header ────────────────────────────────────────
    def _build_header(self):
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame{{background:{SURFACE};border-bottom:1px solid {BORDER};}}"
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(20, 10, 20, 10)

        dot = QLabel("●")
        dot.setStyleSheet(f"color:{GREEN}; font-size:10px;")

        title = QLabel("EV PRIORITY CONTROL SYSTEM")
        title.setObjectName("header_title")

        self._clock_lbl = QLabel()
        self._clock_lbl.setObjectName("header_sub")

        self._mode_badge = QLabel("OR MODE")
        self._mode_badge.setObjectName("badge_on")

        self._sys_badge = QLabel("STOPPED")
        self._sys_badge.setObjectName("badge_off")

        lay.addWidget(dot)
        lay.addSpacing(8)
        lay.addWidget(title)
        lay.addStretch()
        lay.addWidget(self._clock_lbl)
        lay.addSpacing(12)
        lay.addWidget(self._mode_badge)
        lay.addSpacing(6)
        lay.addWidget(self._sys_badge)
        return frame

    # ── Toolbar ───────────────────────────────────────
    def _build_toolbar(self):
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame{{background:{SURFACE2};border-bottom:1px solid {BORDER};}}"
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(14, 7, 14, 7)
        lay.setSpacing(10)

        self.live_checkbox = QCheckBox("Live Mode")
        self.live_checkbox.setChecked(True)
        self.live_checkbox.stateChanged.connect(self._on_live_toggle)

        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("Video file path…")
        self.video_path_edit.setFixedWidth(200)
        self.video_path_edit.setEnabled(False)

        btn_video = QPushButton("Browse Video")
        btn_video.setEnabled(False)
        btn_video.clicked.connect(self.browse_video)
        self._btn_browse_video = btn_video

        # Audio path/browse removed — audio is extracted automatically from the video file

        sep = QLabel("|")
        sep.setStyleSheet(f"color:{BORDER};")

        lbl_fusion = QLabel("Fusion:")
        lbl_fusion.setStyleSheet(f"color:{MUTED}; font-size:11px;")

        self.fusion_combo = QComboBox()
        self.fusion_combo.addItems(["OR", "AND"])   # OR first — paper default
        self.fusion_combo.setCurrentText("OR")
        self.fusion_combo.setFixedWidth(70)
        self.fusion_combo.currentTextChanged.connect(self._on_fusion_mode_change)

        lbl_n = QLabel("N:")
        lbl_n.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        self.n_frames_spin = QSpinBox()
        self.n_frames_spin.setRange(1, 30)
        self.n_frames_spin.setValue(10)
        self.n_frames_spin.setToolTip("Vote queue length N (Algorithm 1)")
        self.n_frames_spin.setFixedWidth(55)
        self.n_frames_spin.valueChanged.connect(self._on_queue_params_change)

        lbl_k = QLabel("K:")
        lbl_k.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        self.k_thresh_spin = QSpinBox()
        self.k_thresh_spin.setRange(1, 30)
        self.k_thresh_spin.setValue(4)
        self.k_thresh_spin.setToolTip("Activation threshold K")
        self.k_thresh_spin.setFixedWidth(55)
        self.k_thresh_spin.valueChanged.connect(self._on_queue_params_change)

        self.window_spin = self.n_frames_spin  # alias kept for compat

        lbl_win = lbl_n  # alias

        sep2 = QLabel("|")
        sep2.setStyleSheet(f"color:{BORDER};")

        self.btn_start = QPushButton("▶  START")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setFixedWidth(100)
        self.btn_start.clicked.connect(self.start_detection)

        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setFixedWidth(100)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_detection)

        for w in [self.live_checkbox, self.video_path_edit, btn_video,
                  sep,
                  lbl_fusion, self.fusion_combo, lbl_n, self.n_frames_spin,
                  lbl_k, self.k_thresh_spin,
                  sep2, self.btn_start, self.btn_stop]:
            lay.addWidget(w)
        lay.addStretch()
        return frame

    # ── Left panel ────────────────────────────────────
    def _build_left_panel(self):
        panel = QWidget()
        panel.setStyleSheet(
            f"QWidget{{background:{SURFACE};border-right:1px solid {BORDER};}}"
        )
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # Audio detection
        grp_audio = QGroupBox("Audio Detection")
        ga = QVBoxLayout(grp_audio)
        ga.setSpacing(8)

        self._audio_bar = ConfBar("SIREN CONFIDENCE", "#a78bfa")
        ga.addWidget(self._audio_bar)

        self._audio_status = QLabel("NO SIREN")
        self._audio_status.setObjectName("badge_off")
        self._audio_status.setAlignment(Qt.AlignCenter)
        ga.addWidget(self._audio_status)

        self._audio_classes_lbl = QLabel(
            "Siren  [82]: 0.00\nAlarm  [87]: 0.00\nEvent [517]: 0.00"
        )
        self._audio_classes_lbl.setStyleSheet(
            f"color:{MUTED}; font-family:'Courier New'; font-size:10px; line-height:1.8;"
        )
        ga.addWidget(self._audio_classes_lbl)

        self._audio_src_lbl = QLabel("Source: —")
        self._audio_src_lbl.setStyleSheet(
            f"color:{MUTED}; font-size:9px; font-style:italic;"
        )
        ga.addWidget(self._audio_src_lbl)
        lay.addWidget(grp_audio)

        # Video detection
        grp_video = QGroupBox("Video Detection")
        gv = QVBoxLayout(grp_video)
        gv.setSpacing(8)

        self._video_bar = ConfBar("VEHICLE CONFIDENCE", "#38bdf8")
        gv.addWidget(self._video_bar)

        self._video_status = QLabel("NO VEHICLE")
        self._video_status.setObjectName("badge_off")
        self._video_status.setAlignment(Qt.AlignCenter)
        gv.addWidget(self._video_status)

        self._video_details_lbl = QLabel("Class: —\nBbox:  —\nDir:   —\nTracks: 0")
        self._video_details_lbl.setStyleSheet(
            f"color:{MUTED}; font-family:'Courier New'; font-size:10px; line-height:1.8;"
        )
        gv.addWidget(self._video_details_lbl)
        lay.addWidget(grp_video)

        # Stats
        grp_stats = QGroupBox("Session Stats")
        gs = QGridLayout(grp_stats)
        gs.setSpacing(6)

        self._stat_audio_tile, self._stat_audio_num = make_stat_tile("0", "Audio")
        self._stat_video_tile, self._stat_video_num = make_stat_tile("0", "Video")
        self._stat_conf_tile,  self._stat_conf_num  = make_stat_tile("0", "Confirmed")
        self._stat_up_tile,    self._stat_up_num    = make_stat_tile("0s", "Uptime")

        gs.addWidget(self._stat_audio_tile, 0, 0)
        gs.addWidget(self._stat_video_tile, 0, 1)
        gs.addWidget(self._stat_conf_tile,  1, 0)
        gs.addWidget(self._stat_up_tile,    1, 1)
        lay.addWidget(grp_stats)

        # Vote-queue panel (Algorithm 1 temporal smoothing visualiser)
        grp_queue = QGroupBox("Temporal Smoothing (Algorithm 1)")
        gq = QVBoxLayout(grp_queue)
        gq.setSpacing(6)

        row_q = QHBoxLayout()
        lbl_q = QLabel("Vote queue  Q / N")
        lbl_q.setStyleSheet(f"color:{MUTED}; font-size:10px; letter-spacing:1px;")
        row_q.addWidget(lbl_q)
        row_q.addStretch()
        self._queue_fill_lbl = QLabel("0 / 10")
        self._queue_fill_lbl.setStyleSheet(
            f"color:{ACCENT}; font-family:'Courier New'; font-size:12px; font-weight:bold;"
        )
        row_q.addWidget(self._queue_fill_lbl)
        gq.addLayout(row_q)

        self._queue_bar = ConfBar("", "#ff6b35")
        self._queue_bar.lbl.hide()
        gq.addWidget(self._queue_bar)

        self._queue_info_lbl = QLabel("K=4 of N=10 votes needed")
        self._queue_info_lbl.setStyleSheet(
            f"color:{MUTED}; font-size:10px; font-family:'Courier New';")
        gq.addWidget(self._queue_info_lbl)
        lay.addWidget(grp_queue)

        lay.addStretch()
        return panel

    # ── Center panel ──────────────────────────────────
    def _build_center_panel(self):
        panel = QWidget()
        panel.setStyleSheet(f"QWidget{{background:{DARK_BG};}}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # Video feed
        grp_video = QGroupBox("Live Camera Feed")
        gvl = QVBoxLayout(grp_video)
        self.video_label = QLabel()
        self.video_label.setMinimumSize(540, 360)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            f"background:{SURFACE}; border:1px solid {BORDER}; border-radius:4px;"
        )
        self.video_label.setText("— Feed not started —")
        gvl.addWidget(self.video_label)
        lay.addWidget(grp_video, 3)

        # Intersection + fusion status
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)

        grp_inter = QGroupBox("Intersection — Live Signal State")
        gi = QVBoxLayout(grp_inter)
        self._intersection = IntersectionWidget()
        gi.addWidget(self._intersection, alignment=Qt.AlignCenter)
        bottom_row.addWidget(grp_inter, 0)

        grp_fusion = QGroupBox("Fusion Decision")
        gf = QVBoxLayout(grp_fusion)
        gf.setSpacing(10)

        self._fusion_result_lbl = QLabel("CLEAR")
        self._fusion_result_lbl.setAlignment(Qt.AlignCenter)
        self._fusion_result_lbl.setStyleSheet(
            f"color:{MUTED}; font-size:20px; font-weight:bold;"
            f"font-family:'Courier New'; letter-spacing:2px;"
        )
        gf.addWidget(self._fusion_result_lbl)

        self._fusion_dir_lbl = QLabel("DIRECTION: —")
        self._fusion_dir_lbl.setAlignment(Qt.AlignCenter)
        self._fusion_dir_lbl.setStyleSheet(
            f"color:{MUTED}; font-size:12px; font-family:'Courier New';"
        )
        gf.addWidget(self._fusion_dir_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{BORDER};")
        gf.addWidget(sep)

        lbl_ns = QLabel("N/S SIGNAL")
        lbl_ns.setStyleSheet(f"color:{MUTED}; font-size:9px; letter-spacing:1px; text-align:center;")
        lbl_ns.setAlignment(Qt.AlignCenter)
        self._ns_badge = QLabel("GREEN")
        self._ns_badge.setObjectName("badge_on")
        self._ns_badge.setAlignment(Qt.AlignCenter)

        lbl_ew = QLabel("E/W SIGNAL")
        lbl_ew.setStyleSheet(f"color:{MUTED}; font-size:9px; letter-spacing:1px;")
        lbl_ew.setAlignment(Qt.AlignCenter)
        self._ew_badge = QLabel("RED")
        self._ew_badge.setObjectName("badge_off")
        self._ew_badge.setAlignment(Qt.AlignCenter)

        gf.addWidget(lbl_ns)
        gf.addWidget(self._ns_badge)
        gf.addWidget(lbl_ew)
        gf.addWidget(self._ew_badge)
        gf.addStretch()
        bottom_row.addWidget(grp_fusion, 1)

        lay.addLayout(bottom_row, 2)
        return panel

    # ── Right panel ───────────────────────────────────
    def _build_right_panel(self):
        panel = QWidget()
        panel.setStyleSheet(
            f"QWidget{{background:{SURFACE};border-left:1px solid {BORDER};}}"
        )
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # Direction indicator
        grp_dir = QGroupBox("Approach Direction")
        gd = QVBoxLayout(grp_dir)

        self._dir_arrow_lbl = QLabel("—")
        self._dir_arrow_lbl.setAlignment(Qt.AlignCenter)
        self._dir_arrow_lbl.setStyleSheet(
            f"color:{MUTED}; font-size:40px; font-family:'Courier New';"
        )
        gd.addWidget(self._dir_arrow_lbl)

        self._dir_sub_lbl = QLabel("No active vehicle")
        self._dir_sub_lbl.setAlignment(Qt.AlignCenter)
        self._dir_sub_lbl.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        gd.addWidget(self._dir_sub_lbl)
        lay.addWidget(grp_dir)

        # Alert banner
        grp_alert = QGroupBox("Alert Status")
        ga2 = QVBoxLayout(grp_alert)
        self._alert_lbl = QLabel("All systems nominal.")
        self._alert_lbl.setWordWrap(True)
        self._alert_lbl.setStyleSheet(
            f"color:{MUTED}; font-family:'Courier New'; font-size:11px;"
        )
        ga2.addWidget(self._alert_lbl)
        lay.addWidget(grp_alert)

        # Evaluation metrics panel
        grp_eval = QGroupBox("Fusion Evaluation Metrics")
        ge = QVBoxLayout(grp_eval)
        ge.setSpacing(5)

        # Ground-truth toggle button
        gt_row = QHBoxLayout()
        self._gt_btn = QPushButton("GT: No EV")
        self._gt_btn.setCheckable(True)
        self._gt_btn.setStyleSheet(
            f"QPushButton{{background:{SURFACE2};color:{MUTED};border:1px solid {BORDER};"
            f"border-radius:4px;padding:5px 10px;font-family:'Courier New';font-size:10px;}}"
            f"QPushButton:checked{{background:rgba(255,51,85,0.15);color:{RED_COLOR};"
            f"border-color:{RED_COLOR};}}"
        )
        self._gt_btn.toggled.connect(self._on_gt_toggle)
        gt_row.addWidget(QLabel("Ground Truth:").also(
            lambda l: l.setStyleSheet(f"color:{MUTED};font-size:10px;")) if False
            else self._make_muted_label("Ground Truth:"))
        gt_row.addWidget(self._gt_btn)
        ge.addLayout(gt_row)

        gt_hint = QLabel("Hold while EV is actually present in frame")
        gt_hint.setStyleSheet(f"color:{MUTED};font-size:9px;font-style:italic;")
        ge.addWidget(gt_hint)

        # Live metric tiles — 2×4 grid
        self._eval_tiles = {}
        tile_grid = QGridLayout()
        tile_grid.setSpacing(4)
        metrics_to_show = [
            ('Precision', 'precision'), ('Recall',   'recall'),
            ('F1 Score',  'f1'),        ('Accuracy', 'accuracy'),
            ('FPR',       'fpr'),       ('FNR',      'fnr'),
            ('MCC',       'mcc'),       ('Latency',  'latency'),
        ]
        for i, (label, key) in enumerate(metrics_to_show):
            cell = QFrame()
            cell.setStyleSheet(
                f"QFrame{{background:{SURFACE2};border:1px solid {BORDER};border-radius:3px;}}"
            )
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(6, 5, 6, 5)
            cl.setSpacing(1)
            num_lbl = QLabel("—")
            num_lbl.setStyleSheet(
                f"color:{ACCENT};font-family:'Courier New';font-size:13px;font-weight:bold;"
            )
            txt_lbl = QLabel(label)
            txt_lbl.setStyleSheet(f"color:{MUTED};font-size:9px;letter-spacing:1px;")
            cl.addWidget(num_lbl)
            cl.addWidget(txt_lbl)
            self._eval_tiles[key] = num_lbl
            tile_grid.addWidget(cell, i // 2, i % 2)
        ge.addLayout(tile_grid)

        # Export + reset row
        btn_row = QHBoxLayout()
        btn_export = QPushButton("Export CSV")
        btn_export.clicked.connect(self._export_metrics)
        btn_reset_eval = QPushButton("Reset Eval")
        btn_reset_eval.clicked.connect(self._reset_eval)
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_reset_eval)
        ge.addLayout(btn_row)

        lay.addWidget(grp_eval)

        # Event log
        grp_log = QGroupBox("Event Log")
        gl = QVBoxLayout(grp_log)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(120)
        gl.addWidget(self._log)

        btn_clear = QPushButton("Clear Log")
        btn_clear.clicked.connect(self._log.clear)
        gl.addWidget(btn_clear)
        lay.addWidget(grp_log, 1)

        return panel

    def _make_muted_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{MUTED};font-size:10px;")
        return lbl

    # ── Bottom bar ────────────────────────────────────
    def _build_bottom_bar(self):
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame{{background:{SURFACE};border-top:1px solid {BORDER};}}"
        )
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(16, 6, 16, 6)

        self._status_lbl = QLabel("● System ready")
        self._status_lbl.setStyleSheet(
            f"color:{GREEN}; font-family:'Courier New'; font-size:11px;"
        )
        lay.addWidget(self._status_lbl)
        lay.addStretch()

        lay.addWidget(QLabel("v2.0  |  PyQt5 + YOLOv8 + PANNs CNN14").setStyleSheet if False else QLabel(""))
        ver = QLabel("v2.0  |  YOLOv8 + PANNs CNN14")
        ver.setStyleSheet(f"color:{MUTED}; font-size:10px;")
        lay.addWidget(ver)
        return frame

    # ──────────────────────────────────────────────────
    # Toolbar callbacks
    # ──────────────────────────────────────────────────
    def _on_live_toggle(self, state):
        enabled = (state != Qt.Checked)
        self.video_path_edit.setEnabled(enabled)
        self._btn_browse_video.setEnabled(enabled)

    def browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", filter="Video Files (*.mp4 *.avi *.mov)"
        )
        if path:
            self.video_path_edit.setText(path)

    def _on_fusion_mode_change(self, mode_text):
        self.fusion_module.set_mode(mode_text)
        # OR = paper default (high sensitivity / safety-critical) -> green badge
        # AND = high precision (may miss detections)              -> yellow badge
        if mode_text == "OR":
            self._mode_badge.setObjectName("badge_on")
        else:
            self._mode_badge.setObjectName("badge_warn")
        self._mode_badge.setText(f"{mode_text} MODE")
        self._mode_badge.setStyle(self._mode_badge.style())
        desc = ("OR: high sensitivity — confirms if EITHER modality fires (paper default)"
                if mode_text == "OR" else
                "AND: high precision — requires BOTH modalities to fire simultaneously")
        self._log_event(f"Fusion mode → {mode_text}  [{desc}]", "sys")

    def _on_queue_params_change(self, _value=None):
        """Sync N and K from spinboxes into fusion_module (Algorithm 1 params)."""
        n = self.n_frames_spin.value()
        k = self.k_thresh_spin.value()
        # Clamp K so it can never exceed N
        if k > n:
            self.k_thresh_spin.blockSignals(True)
            self.k_thresh_spin.setValue(n)
            self.k_thresh_spin.blockSignals(False)
            k = n
        self.fusion_module.set_queue_params(n_frames=n, k_thresh=k)
        if hasattr(self, '_queue_info_lbl'):
            self._queue_info_lbl.setText(f"K={k} of N={n} votes needed for GREEN")
        self._log_event(f"Queue params → N={n}, K={k}", "sys")

    # ──────────────────────────────────────────────────
    # Start / Stop
    # ──────────────────────────────────────────────────
    def start_detection(self):
        live = self.live_checkbox.isChecked()
        video_file = None if live else (self.video_path_edit.text() or None)

        # Audio is now extracted automatically from the video file.
        # In live mode: microphone. In file mode: audio track from the same video.
        if live:
            self.audio_module = AudioModule(mode='live')
        else:
            self.audio_module = AudioModule(mode='video', video_file=video_file)

        self.video_module = VideoModule(
            model_path="best.pt",
            mode='live' if live else 'file',
            video_file=video_file,
        )

        # Reconfigure fusion from UI values before starting
        self.fusion_module.set_mode(self.fusion_combo.currentText())
        self.fusion_module.set_queue_params(
            n_frames=self.n_frames_spin.value(),
            k_thresh=self.k_thresh_spin.value(),
        )
        self.fusion_module.reset()   # clear any stale votes from previous run
        self.evaluator.reset()
        self._gt_active = False
        self._gt_btn.setChecked(False)
        self.audio_module.start()

        self._start_time = time.time()
        self._total_audio = self._total_video = self._total_confirm = 0

        self.timer.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._sys_badge.setText("RUNNING")
        self._sys_badge.setObjectName("badge_on")
        self._sys_badge.setStyle(self._sys_badge.style())
        self._status_lbl.setText("● Detection running")
        # Update audio source label
        if live:
            self._audio_src_lbl.setText("Source: Microphone (live)")
        else:
            vf = self.video_path_edit.text() or "video file"
            self._audio_src_lbl.setText(f"Source: Extracted from {vf.split('/')[-1].split(chr(92))[-1]}")
        m = self.fusion_combo.currentText()
        n = self.n_frames_spin.value()
        k = self.k_thresh_spin.value()
        src = "LIVE" if live else "FILE"
        self._log_event(
            f"System started — src:{src}  fusion:{m}  N={n}  K={k}  "
            f"τ_v={self.fusion_module.tau_v}  τ_a={self.fusion_module.tau_a}",
            "sys",
        )

    def stop_detection(self):
        self.timer.stop()
        if self.audio_module:
            self.audio_module.stop()
        if self.video_module:
            self.video_module.release()

        self.fusion_module.reset()
        self.signal_module.update_signals(None)
        self._intersection.set_state("green", "red")
        self._intersection.set_emergency(False)

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        self._sys_badge.setText("STOPPED")
        self._sys_badge.setObjectName("badge_off")
        self._sys_badge.setStyle(self._sys_badge.style())
        self._status_lbl.setText("● System stopped")
        self._log_event("System stopped", "sys")

    # ──────────────────────────────────────────────────
    # Main tick
    # ──────────────────────────────────────────────────
    def _tick(self):
        """
        Called every 100 ms (10 FPS). Implements Algorithm 1 of the paper end-to-end:
          1. Read raw confidence values from both modalities.
          2. Pass them to fusion_module.push_frame() which runs
             the instantaneous decision + temporal smoothing.
          3. Update GUI based on the smoothed result.
        """
        # ── Video ──────────────────────────────────────
        frame, directions = self.video_module.get_frame()
        if frame is None:
            return

        pix = convert_cv_qt(frame, width=self.video_label.width())
        self.video_label.setPixmap(pix)

        # ── Collect raw confidences (Section III-C lines 3-4) ──────────────
        # conf_a: CNN14 siren probability — frame-synchronised in video mode
        if self.audio_module.mode == 'video':
            fps = self.video_module.cap.get(5) or 25.0   # cv2.CAP_PROP_FPS = 5
            frame_idx = int(self.video_module.cap.get(1)) - 1  # CAP_PROP_POS_FRAMES, -1 = just-read
            conf_a = self.audio_module.get_confidence_for_frame(max(0, frame_idx), fps)
        else:
            # Live mic: last_confidence is updated continuously by background thread
            conf_a = getattr(self.audio_module, 'last_confidence', 0.0) or 0.0

        # conf_v: highest YOLO detection confidence this frame
        conf_v = getattr(self.video_module, 'last_confidence', 0.0) or 0.0
        # direction: most recent approach direction from tracker (may be None)
        direction_now = directions[-1] if directions else None

        # ── Log threshold crossings ─────────────────────────────────────────
        audio_over = conf_a > self.fusion_module.tau_a
        video_over = conf_v > self.fusion_module.tau_v and bool(directions)

        if audio_over:
            self._total_audio += 1
            self._stat_audio_num.setText(str(self._total_audio))
            self._log_event(
                f"Siren — P(siren)={conf_a:.2f} > τ_a={self.fusion_module.tau_a}",
                "audio",
            )
        if video_over:
            self._total_video += 1
            self._stat_video_num.setText(str(self._total_video))
            self._log_event(
                f"Vehicle — conf={conf_v:.2f} > τ_v={self.fusion_module.tau_v}  dir={direction_now}",
                "video",
            )

        # ── Algorithm 1 lines 5-8: push one frame through fusion ───────────
        prev_confirmed = self.fusion_module._confirmed
        confirmed, direction = self.fusion_module.push_frame(
            conf_v=conf_v,
            conf_a=conf_a,
            direction=direction_now,
        )

        # Log rising edge of confirmation only (avoid log spam)
        if confirmed and not prev_confirmed:
            self._total_confirm += 1
            self._stat_conf_num.setText(str(self._total_confirm))
            self._log_event(
                f"FUSION CONFIRMED — mode:{self.fusion_module.mode} "
                f"Q={self.fusion_module.queue_sum}/{self.fusion_module.n_frames} "
                f"dir:{direction or '—'}",
                "fusion",
            )

        # ── Audio confidence panel ─────────────────────
        self._audio_bar.set_value(conf_a * 100)
        classes = getattr(self.audio_module, 'last_class_scores', {})
        self._audio_classes_lbl.setText(
            "Siren  [82]: {:.2f}\nAlarm  [87]: {:.2f}\nEvent [517]: {:.2f}".format(
                classes.get(82, 0.0), classes.get(87, 0.0), classes.get(517, 0.0)
            )
        )
        if audio_over:
            self._audio_status.setText("SIREN DETECTED")
            self._audio_status.setObjectName("badge_on")
        else:
            self._audio_status.setText("NO SIREN")
            self._audio_status.setObjectName("badge_off")
        self._audio_status.setStyle(self._audio_status.style())

        # ── Video confidence panel ─────────────────────
        self._video_bar.set_value(conf_v * 100)
        if directions:
            cls_name = getattr(self.video_module, 'last_class_name', '—')
            bbox     = getattr(self.video_module, 'last_bbox', '—')
            n_tracks = len(self.video_module.tracks)
            self._video_details_lbl.setText(
                "Class:  {}\nBbox:   {}\nDir:    {}\nTracks: {}".format(
                    cls_name, bbox, direction_now, n_tracks
                )
            )
            self._video_status.setText("VEHICLE DETECTED")
            self._video_status.setObjectName("badge_on")
        else:
            self._video_status.setText("NO VEHICLE")
            self._video_status.setObjectName("badge_off")
        self._video_status.setStyle(self._video_status.style())

        # ── Vote-queue fill bar (visualises temporal smoothing) ────────────
        if hasattr(self, '_queue_bar'):
            self._queue_bar.set_value(self.fusion_module.queue_fill * 100)
            q  = self.fusion_module.queue_sum
            n  = self.fusion_module.n_frames
            k  = self.fusion_module.k_thresh
            self._queue_fill_lbl.setText(f"{q} / {n}")
            self._queue_fill_lbl.setStyleSheet(
                f"color:{ACCENT2 if q >= k else ACCENT}; "
                "font-family:'Courier New'; font-size:12px; font-weight:bold;"
            )

        self._update_signals_ui(confirmed, direction)

        # ── Evaluation metrics ─────────────────────────
        self._push_eval(
            conf_v=conf_v,
            conf_a=conf_a,
            pred_audio=audio_over,
            pred_video=video_over,
            pred_fusion=confirmed,
        )

        # ── Uptime

        # ── Uptime ─────────────────────────────────────
        if self._start_time:
            elapsed = int(time.time() - self._start_time)
            if elapsed < 60:
                self._stat_up_num.setText(f"{elapsed}s")
            else:
                self._stat_up_num.setText(f"{elapsed // 60}m{elapsed % 60:02d}s")

    # ──────────────────────────────────────────────────
    # Signal / UI update helpers
    # ──────────────────────────────────────────────────
    def _update_signals_ui(self, confirmed: bool, direction):
        if confirmed:
            if direction is not None:
                self.signal_module.update_signals(direction)
                ns_c = "green" if direction.upper() in ('N', 'S') else "red"
                ew_c = "green" if direction.upper() in ('E', 'W') else "red"
                self._intersection.set_state(ns_c, ew_c)
                self._intersection.set_emergency(True, direction)
                self._set_ns_ew_badges(ns_c, ew_c)
                self._fusion_result_lbl.setText("ACTIVE")
                self._fusion_result_lbl.setStyleSheet(
                    f"color:{ACCENT2}; font-size:20px; font-weight:bold;"
                    f"font-family:'Courier New'; letter-spacing:2px;"
                )
                self._fusion_dir_lbl.setText(f"DIRECTION: {direction}")
                self._fusion_dir_lbl.setStyleSheet(
                    f"color:{ACCENT2}; font-size:12px; font-family:'Courier New';"
                )
                arrows = {'N': '↑', 'S': '↓', 'E': '→', 'W': '←'}
                self._dir_arrow_lbl.setText(arrows.get(direction.upper(), '—'))
                self._dir_arrow_lbl.setStyleSheet(
                    f"color:{ACCENT2}; font-size:40px; font-family:'Courier New';"
                )
                self._dir_sub_lbl.setText(f"Vehicle approaching from {direction}")
                self._dir_sub_lbl.setStyleSheet(f"color:{ACCENT2}; font-size:10px;")
                self._alert_lbl.setText(
                    f"⚠ EMERGENCY VEHICLE CONFIRMED\nPrioritizing {direction} approach"
                )
                self._alert_lbl.setStyleSheet(
                    f"color:{ACCENT2}; font-family:'Courier New'; font-size:11px;"
                )
            else:
                # Audio-only OR mode — all red safe fallback
                self.signal_module.set_all_red()
                self._intersection.set_state("red", "red")
                self._intersection.set_emergency(True, None)
                self._set_ns_ew_badges("red", "red")
                self._fusion_result_lbl.setText("ALL-RED")
                self._fusion_result_lbl.setStyleSheet(
                    f"color:{YELLOW}; font-size:20px; font-weight:bold;"
                    f"font-family:'Courier New'; letter-spacing:2px;"
                )
                self._fusion_dir_lbl.setText("AUDIO ONLY — SAFE FALLBACK")
                self._fusion_dir_lbl.setStyleSheet(
                    f"color:{YELLOW}; font-size:12px; font-family:'Courier New';"
                )
                self._dir_arrow_lbl.setText("!")
                self._dir_arrow_lbl.setStyleSheet(
                    f"color:{YELLOW}; font-size:40px; font-family:'Courier New';"
                )
                self._dir_sub_lbl.setText("Audio only — direction unknown")
                self._dir_sub_lbl.setStyleSheet(f"color:{YELLOW}; font-size:10px;")
                self._alert_lbl.setText(
                    "⚠ AUDIO ONLY DETECTION\nAll-red safe fallback active"
                )
                self._alert_lbl.setStyleSheet(
                    f"color:{YELLOW}; font-family:'Courier New'; font-size:11px;"
                )
        else:
            self.signal_module.update_signals(None)
            self._intersection.set_state("green", "red")
            self._intersection.set_emergency(False)
            self._set_ns_ew_badges("green", "red")
            self._fusion_result_lbl.setText("CLEAR")
            self._fusion_result_lbl.setStyleSheet(
                f"color:{MUTED}; font-size:20px; font-weight:bold;"
                f"font-family:'Courier New'; letter-spacing:2px;"
            )
            self._fusion_dir_lbl.setText("DIRECTION: —")
            self._fusion_dir_lbl.setStyleSheet(
                f"color:{MUTED}; font-size:12px; font-family:'Courier New';"
            )
            self._dir_arrow_lbl.setText("—")
            self._dir_arrow_lbl.setStyleSheet(
                f"color:{MUTED}; font-size:40px; font-family:'Courier New';"
            )
            self._dir_sub_lbl.setText("No active vehicle")
            self._dir_sub_lbl.setStyleSheet(f"color:{MUTED}; font-size:10px;")
            self._alert_lbl.setText("All systems nominal. Monitoring active.")
            self._alert_lbl.setStyleSheet(
                f"color:{MUTED}; font-family:'Courier New'; font-size:11px;"
            )

    def _set_ns_ew_badges(self, ns: str, ew: str):
        color_to_obj = {"green": "badge_on", "red": "badge_off", "yellow": "badge_warn"}
        self._ns_badge.setText(ns.upper())
        self._ns_badge.setObjectName(color_to_obj.get(ns, "badge_off"))
        self._ns_badge.setStyle(self._ns_badge.style())
        self._ew_badge.setText(ew.upper())
        self._ew_badge.setObjectName(color_to_obj.get(ew, "badge_off"))
        self._ew_badge.setStyle(self._ew_badge.style())

    # ──────────────────────────────────────────────────
    # Event log
    # ──────────────────────────────────────────────────
    _LOG_COLORS = {
        "audio":  "#a78bfa",
        "video":  "#38bdf8",
        "fusion": "#ff6b35",
        "sys":    "#4a6080",
    }

    def _log_event(self, msg: str, kind: str = "sys"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        color = self._LOG_COLORS.get(kind, TEXT)
        html = (
            f'<span style="color:{MUTED};font-size:10px;">{ts}</span>&nbsp;&nbsp;'
            f'<span style="color:{color};font-size:11px;">{msg}</span>'
        )
        self._log.append(html)
        # Auto-scroll to bottom
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ──────────────────────────────────────────────────
    # Clock
    # ──────────────────────────────────────────────────
    def _update_clock(self):
        self._clock_lbl.setText(
            datetime.datetime.utcnow().strftime("%Y-%m-%d  %H:%M:%S UTC")
        )

    # ──────────────────────────────────────────────────
    # Evaluation helpers
    # ──────────────────────────────────────────────────
    def _on_gt_toggle(self, checked: bool):
        """Operator toggles ground-truth: EV is actually present in the scene."""
        self._gt_active = checked
        self._gt_btn.setText("GT: EV PRESENT" if checked else "GT: No EV")
        self._log_event(
            "Ground truth SET — EV present" if checked else "Ground truth CLEAR",
            "sys",
        )

    def _push_eval(self, conf_v: float, conf_a: float,
                   pred_audio: bool, pred_video: bool, pred_fusion: bool):
        """Push one frame into the evaluator and refresh the metric tiles."""
        self.evaluator.push(
            gt=self._gt_active,
            pred_audio=pred_audio,
            pred_video=pred_video,
            pred_fusion=pred_fusion,
            conf_v=conf_v,
            conf_a=conf_a,
        )
        # Refresh tile display every 10 frames to avoid UI thrashing
        if self.evaluator._total_frames % 10 == 0:
            self._refresh_eval_tiles()

    def _refresh_eval_tiles(self):
        """Update the 8 live metric tiles from the evaluator."""
        m = self.evaluator.compute()
        f = m['fusion']
        lat = m['latency']
        mapping = {
            'precision': f'{f["Precision"]:.3f}',
            'recall':    f'{f["Recall"]:.3f}',
            'f1':        f'{f["F1"]:.3f}',
            'accuracy':  f'{f["Accuracy"]:.3f}',
            'fpr':       f'{f["FPR"]:.3f}',
            'fnr':       f'{f["FNR"]:.3f}',
            'mcc':       f'{f["MCC"]:.3f}',
            'latency':   f'{lat["mean_detection_frames"]}f',
        }
        for key, val in mapping.items():
            lbl = self._eval_tiles.get(key)
            if lbl:
                lbl.setText(val)
                # Color-code: good values green, warning orange, bad red
                if key in ('precision', 'recall', 'f1', 'accuracy', 'mcc'):
                    v = float(val)
                    color = GREEN if v >= 0.85 else (YELLOW if v >= 0.60 else RED_COLOR)
                elif key in ('fpr', 'fnr'):
                    v = float(val)
                    color = GREEN if v <= 0.10 else (YELLOW if v <= 0.25 else RED_COLOR)
                else:
                    color = ACCENT
                lbl.setStyleSheet(
                    f"color:{color};font-family:'Courier New';"
                    f"font-size:13px;font-weight:bold;"
                )

    def _export_metrics(self):
        """Save metrics CSV via file dialog."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Evaluation Metrics", "fusion_eval.csv",
            filter="CSV Files (*.csv)"
        )
        if path:
            self.evaluator.export_csv(path)
            self._log_event(f"Metrics exported → {path}", "sys")
            # Also print summary to log
            for line in self.evaluator.summary_text().split('\n'):
                self._log_event(line, "fusion")

    def _reset_eval(self):
        """Reset evaluator counters and tile display."""
        self.evaluator.reset()
        for lbl in self._eval_tiles.values():
            lbl.setText("—")
            lbl.setStyleSheet(
                f"color:{ACCENT};font-family:'Courier New';"
                f"font-size:13px;font-weight:bold;"
            )
        self._log_event("Evaluation counters reset", "sys")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Apply dark base palette so native widgets also go dark
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(10, 14, 26))
    palette.setColor(QPalette.WindowText,      QColor(232, 244, 248))
    palette.setColor(QPalette.Base,            QColor(17, 24, 39))
    palette.setColor(QPalette.AlternateBase,   QColor(26, 34, 53))
    palette.setColor(QPalette.ToolTipBase,     QColor(17, 24, 39))
    palette.setColor(QPalette.ToolTipText,     QColor(232, 244, 248))
    palette.setColor(QPalette.Text,            QColor(232, 244, 248))
    palette.setColor(QPalette.Button,          QColor(26, 34, 53))
    palette.setColor(QPalette.ButtonText,      QColor(232, 244, 248))
    palette.setColor(QPalette.Highlight,       QColor(0, 212, 255))
    palette.setColor(QPalette.HighlightedText, QColor(10, 14, 26))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
