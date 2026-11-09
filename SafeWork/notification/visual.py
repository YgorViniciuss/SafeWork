"""Alerta visual: overlay piscante do Estado 1 e destaque do Estado 2.

É desenhado sobre o frame já anotado pelo Módulo 1. O resultado é também a
imagem de evidência entregue ao Módulo 3 (caixa + ROI + alerta + data e hora).
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from safework.core.config import AlertConfig
from safework.core.models import PersonAnalysis
from safework.logic.states import AlertSnapshot, AlertState, OperatorAlert
from safework.vision.renderer import ascii_text

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_RED = (0, 0, 255)
_YELLOW = (0, 220, 255)
_LIME = (0, 230, 160)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_STATUS_BAR_HEIGHT = 28  # altura da barra desenhada pelo FrameRenderer


class AlertOverlay:
    """Desenha caixas, contagem regressiva, borda piscante e faixa de alerta."""

    def __init__(self, config: AlertConfig) -> None:
        self._config = config

    def draw(
        self,
        image: np.ndarray,
        persons: Sequence[PersonAnalysis],
        snapshot: AlertSnapshot,
    ) -> np.ndarray:
        canvas = image.copy()
        if not snapshot.operators:
            return canvas

        blink_on = self._blink_on(snapshot.timestamp)
        for person in persons:
            alert = snapshot.operators.get(person.track_id) if person.track_id is not None else None
            if alert is not None:
                self._draw_operator(canvas, person, alert, blink_on)

        highest = snapshot.highest_state
        if highest is AlertState.OK:  # apenas operadores em histerese
            return canvas
        self._draw_border(canvas, highest, blink_on)
        self._draw_banner(canvas, snapshot, highest, blink_on)
        return canvas

    # ------------------------------------------------------------------ partes
    def _blink_on(self, timestamp: float) -> bool:
        """Fase do pisca: usa o tempo do frame, então é igual ao vivo e em vídeo."""
        return int(timestamp * self._config.blink_hz * 2) % 2 == 0

    def _draw_operator(
        self,
        canvas: np.ndarray,
        person: PersonAnalysis,
        alert: OperatorAlert,
        blink_on: bool,
    ) -> None:
        box = person.detection.box
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)

        if alert.clearing:
            color, label, thickness = _LIME, "EPI detectado - verificando", 2
        elif alert.state is AlertState.VIOLATION:
            color, label, thickness = _RED, "VIOLACAO CONFIRMADA", 4
        else:
            color = _RED if blink_on else _YELLOW
            label, thickness = f"ALERTA {alert.remaining_s:.1f}s", 3

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        self._label_below(canvas, label, x1, y2, color)

        if alert.state is AlertState.WARNING and not alert.clearing:
            self._progress_bar(canvas, x1, x2, y2, alert)

    def _progress_bar(
        self,
        canvas: np.ndarray,
        x1: int,
        x2: int,
        y2: int,
        alert: OperatorAlert,
    ) -> None:
        """Barra que enche até a confirmação da infração."""
        total = self._config.warning_seconds or 1.0
        ratio = min(alert.elapsed_s / total, 1.0)
        height = canvas.shape[0]
        top = min(y2 + 24, height - 6)
        cv2.rectangle(canvas, (x1, top), (x2, top + 6), _BLACK, -1)
        cv2.rectangle(canvas, (x1, top), (x1 + int((x2 - x1) * ratio), top + 6), _RED, -1)

    def _draw_border(self, canvas: np.ndarray, state: AlertState, blink_on: bool) -> None:
        if state is AlertState.VIOLATION:
            color = _RED
        else:
            color = _RED if blink_on else _YELLOW
        height, width = canvas.shape[:2]
        half = 4  # a linha de 8 px é centrada no contorno: recua para ficar toda visível
        cv2.rectangle(canvas, (half, _STATUS_BAR_HEIGHT + half),
                      (width - 1 - half, height - 1 - half), color, 2 * half)

    def _draw_banner(
        self,
        canvas: np.ndarray,
        snapshot: AlertSnapshot,
        state: AlertState,
        blink_on: bool,
    ) -> None:
        operators = [item for item in snapshot.operators.values() if item.state is state]
        if state is AlertState.VIOLATION:
            first = min(operators, key=lambda item: item.track_id)
            zone = first.zone_name or "zona de risco"
            missing = ", ".join(sorted(item.label for item in first.missing_ppe)) or "EPI"
            text = f"INFRACAO CONFIRMADA - Operador #{first.track_id} sem {missing} em {zone}"
            background, foreground = _RED, _WHITE
        else:
            remaining = min(item.remaining_s for item in operators)
            text = f"ATENCAO: operador sem EPI na zona de risco - {remaining:.1f}s"
            background = _RED if blink_on else _YELLOW
            foreground = _WHITE if blink_on else _BLACK
        if len(operators) > 1:
            text += f" (+{len(operators) - 1})"

        width = canvas.shape[1]
        top, bottom = _STATUS_BAR_HEIGHT + 8, _STATUS_BAR_HEIGHT + 40
        cv2.rectangle(canvas, (8, top), (width - 9, bottom), background, -1)
        content = ascii_text(text)
        scale = 0.6
        while scale > 0.35 and cv2.getTextSize(content, _FONT, scale, 2)[0][0] > width - 32:
            scale -= 0.05
        cv2.putText(canvas, content, (16, bottom - 10), _FONT, scale, foreground, 2,
                    cv2.LINE_AA)

    @staticmethod
    def _label_below(
        canvas: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        content = ascii_text(text)
        (text_width, text_height), baseline = cv2.getTextSize(content, _FONT, 0.55, 2)
        height, width = canvas.shape[:2]
        top = min(y + 2, height - text_height - baseline - 6)
        left = min(max(x, 0), max(width - text_width - 6, 0))
        bottom = top + text_height + baseline + 4
        cv2.rectangle(canvas, (left, top), (left + text_width + 6, bottom), color, -1)
        text_color = _WHITE if color == _RED else _BLACK
        cv2.putText(canvas, content, (left + 3, bottom - baseline - 1), _FONT, 0.55, text_color,
                    2, cv2.LINE_AA)
