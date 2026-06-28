# signal_module.py
"""
Traffic signal simulation for a fixed 4-way intersection (N, S, E, W).
Provides methods to set default state, override for a given direction,
and set all-red safe fallback (used when OR mode confirms via audio only).

Changes from v1:
  - Yellow state supported in update_signals() and set_all_yellow().
  - _set_light() draws a full 3-lamp housing (red / yellow / green stack)
    so the QLabel always shows a proper traffic-light column rather than
    a single dot.
  - Direction label (N / S / E / W) painted below each housing.
  - set_all_yellow() convenience method for transition phases.
"""

from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
from PyQt5.QtCore import Qt


# Colour palette matches the dark-theme main window
_GREEN  = QColor(0, 255, 136)
_YELLOW = QColor(255, 215, 0)
_RED    = QColor(255, 51, 85)
_DIM    = QColor(26, 34, 53)      # unlit lamp colour
_HOUSING = QColor(13, 21, 32)     # light housing background
_BORDER  = QColor(30, 58, 95)     # housing border


class SignalModule:
    """
    Manage 4 QLabel widgets representing traffic lights for N, S, E, W.
    Each label receives a QPixmap with a full 3-lamp traffic light drawn.
    """

    # Ordered top-to-bottom (red, yellow, green) for a vertical housing.
    _LAMP_ORDER = ("red", "yellow", "green")

    def __init__(self, labels: dict):
        """
        Args:
            labels: dict mapping 'N','S','E','W' to QLabel objects.
        """
        self.labels = labels
        # Keep track of current colours so external code can query them.
        self._current = {k: "green" if k in ("N", "S") else "red"
                         for k in ("N", "S", "E", "W")}
        self.set_default_state()

    # ──────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────

    def set_default_state(self):
        """Default: North-South = green, East-West = red."""
        self._set_light("N", "green")
        self._set_light("S", "green")
        self._set_light("E", "red")
        self._set_light("W", "red")

    def update_signals(self, direction):
        """
        If direction is None  → reset to default.
        If direction is 'N'/'S' → give N/S green, E/W red.
        If direction is 'E'/'W' → give E/W green, N/S red.
        Unknown direction      → safe fallback (all red).
        """
        if direction is None:
            self.set_default_state()
            return

        direction = direction.upper()
        if direction in ("N", "S"):
            self._set_light("N", "green")
            self._set_light("S", "green")
            self._set_light("E", "red")
            self._set_light("W", "red")
        elif direction in ("E", "W"):
            self._set_light("E", "green")
            self._set_light("W", "green")
            self._set_light("N", "red")
            self._set_light("S", "red")
        else:
            self.set_all_red()

    def set_all_red(self):
        """Set all signals to red (safe fallback)."""
        for k in ("N", "S", "E", "W"):
            self._set_light(k, "red")

    def set_all_yellow(self):
        """Set all signals to yellow (transition / caution phase)."""
        for k in ("N", "S", "E", "W"):
            self._set_light(k, "yellow")

    def get_state(self, direction_key: str) -> str:
        """Return current colour string for a given direction key."""
        return self._current.get(direction_key.upper(), "red")

    # ──────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────

    def _set_light(self, dir_key: str, color: str):
        """
        Draw a 3-lamp vertical traffic-light housing on the QLabel pixmap.
        color in {'red', 'yellow', 'green'}.
        """
        self._current[dir_key] = color
        label = self.labels.get(dir_key)
        if label is None:
            return

        w = max(label.width()  or 44, 44)
        h = max(label.height() or 80, 80)

        pixmap = QPixmap(w, h)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Housing rectangle (leaves a few px margin on all sides)
        margin = 4
        lamp_area_h = h - margin * 2 - 14   # 14 px reserved for the dir label
        housing_w = min(w - margin * 2, lamp_area_h // 3 + 6)
        housing_x = (w - housing_w) // 2
        housing_y = margin

        painter.setBrush(_HOUSING)
        painter.setPen(_BORDER)
        painter.drawRoundedRect(housing_x, housing_y, housing_w, lamp_area_h, 4, 4)

        # Three lamps
        lamp_r = (housing_w - 8) // 2
        lamp_r = max(lamp_r, 4)
        lamp_spacing = lamp_area_h // 3
        for i, lamp_color in enumerate(self._LAMP_ORDER):
            lit = (lamp_color == color)
            lamp_cx = w // 2
            lamp_cy = housing_y + lamp_spacing * i + lamp_spacing // 2

            # Lit lamp: glow halo
            if lit:
                glow_c = self._color_for(lamp_color)
                halo = QColor(glow_c.red(), glow_c.green(), glow_c.blue(), 50)
                painter.setBrush(halo)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(lamp_cx - lamp_r - 3, lamp_cy - lamp_r - 3,
                                    (lamp_r + 3) * 2, (lamp_r + 3) * 2)

            fill = self._color_for(lamp_color) if lit else _DIM
            painter.setBrush(fill)
            painter.setPen(QColor(0, 0, 0, 60))
            painter.drawEllipse(lamp_cx - lamp_r, lamp_cy - lamp_r,
                                lamp_r * 2, lamp_r * 2)

        # Direction label at bottom
        painter.setPen(QColor(74, 96, 128))
        font = QFont("Courier New", 9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(0, h - 14, w, 14, Qt.AlignHCenter | Qt.AlignVCenter, dir_key)

        painter.end()
        label.setPixmap(pixmap)

    @staticmethod
    def _color_for(name: str) -> QColor:
        return {"green": _GREEN, "yellow": _YELLOW, "red": _RED}.get(name, _DIM)
