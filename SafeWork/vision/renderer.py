"""Desenho dos overlays: zonas, caixas, rótulos e barra de status.

Esta classe é a base do frame de evidência, por isso a imagem original nunca
é alterada: todo desenho ocorre sobre uma cópia.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from datetime import datetime

import cv2
import numpy as np

from safework.core.config import CameraConfig, DisplayConfig
from safework.core.models import Detection, ObjectClass, PersonAnalysis, PersonStatus
from safework.vision.roi import RiskZone

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Cores no padrão BGR do OpenCV.
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_ZONE_COLOR = (0, 165, 255)
_PPE_COLOR = (255, 200, 0)
_STATUS_COLORS = {
    PersonStatus.OUTSIDE_ZONE: (200, 200, 200),
    PersonStatus.COMPLIANT: (0, 200, 0),
    PersonStatus.NON_COMPLIANT: (0, 0, 255),
    PersonStatus.UNVERIFIED: (0, 220, 255),
}
_STATUS_LABELS = {
    PersonStatus.OUTSIDE_ZONE: "Fora da zona",
    PersonStatus.COMPLIANT: "OK",
    PersonStatus.NON_COMPLIANT: "SEM EPI",
    PersonStatus.UNVERIFIED: "EPI nao verificado",
}
_PPE_LABELS = {ObjectClass.HELMET: "Capacete", ObjectClass.VEST: "Colete"}


def ascii_text(text: str) -> str:
    """Remove acentos: as fontes do cv2.putText só desenham caracteres ASCII."""
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


class FrameRenderer:
    """Compõe a imagem anotada exibida ao operador."""

    def __init__(
        self,
        camera: CameraConfig,
        display: DisplayConfig,
        zones: Sequence[RiskZone],
    ) -> None:
        self._camera = camera
        self._display = display
        self._zones = list(zones)

    def update_zones(self, zones: Sequence[RiskZone]) -> None:
        self._zones = list(zones)

    def render(
        self,
        image: np.ndarray,
        persons: Sequence[PersonAnalysis],
        detections: Sequence[Detection],
        captured_at: datetime,
        fps: float = 0.0,
    ) -> np.ndarray:
        canvas = image.copy()
        self._draw_zones(canvas, persons)
        for detection in detections:
            if detection.object_class is not ObjectClass.PERSON:
                self._draw_ppe(canvas, detection)
        for person in persons:
            self._draw_person(canvas, person)
        self._draw_status_bar(canvas, persons, captured_at, fps)
        return canvas

    def _draw_zones(self, canvas: np.ndarray, persons: Sequence[PersonAnalysis]) -> None:
        height, width = canvas.shape[:2]
        violated = {
            person.zone_id for person in persons if person.status is PersonStatus.NON_COMPLIANT
        }
        overlay = canvas.copy()
        for zone in self._zones:
            color = (0, 0, 255) if zone.id in violated else _ZONE_COLOR
            cv2.fillPoly(overlay, [zone.polygon_px(width, height)], color)
        opacity = self._display.zone_opacity
        cv2.addWeighted(overlay, opacity, canvas, 1 - opacity, 0, canvas)

        for zone in self._zones:
            color = (0, 0, 255) if zone.id in violated else _ZONE_COLOR
            polygon = zone.polygon_px(width, height)
            cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
            x, y = polygon[np.argmin(polygon[:, 1])]
            self._draw_label(canvas, zone.name, int(x), int(y), color)

    def _draw_person(self, canvas: np.ndarray, person: PersonAnalysis) -> None:
        box = person.detection.box
        color = _STATUS_COLORS[person.status]
        top_left = (int(box.x1), int(box.y1))
        cv2.rectangle(canvas, top_left, (int(box.x2), int(box.y2)), color, 2, cv2.LINE_AA)
        center = (int(person.anchor[0]), int(person.anchor[1]))
        cv2.circle(canvas, center, 4, color, -1, cv2.LINE_AA)

        identifier = f"#{person.track_id} " if person.track_id is not None else ""
        label = f"{identifier}{_STATUS_LABELS[person.status]}"
        if person.missing_ppe:
            names = ", ".join(sorted(item.label for item in person.missing_ppe))
            label = f"{identifier}Falta: {names}"
        self._draw_label(canvas, label, *top_left, color)

    def _draw_ppe(self, canvas: np.ndarray, detection: Detection) -> None:
        box = detection.box
        top_left = (int(box.x1), int(box.y1))
        cv2.rectangle(canvas, top_left, (int(box.x2), int(box.y2)), _PPE_COLOR, 1, cv2.LINE_AA)
        self._draw_label(canvas, _PPE_LABELS[detection.object_class], *top_left, _PPE_COLOR, 0.4)

    def _draw_status_bar(
        self,
        canvas: np.ndarray,
        persons: Sequence[PersonAnalysis],
        captured_at: datetime,
        fps: float,
    ) -> None:
        width = canvas.shape[1]
        cv2.rectangle(canvas, (0, 0), (width, 28), _BLACK, -1)
        in_zone = sum(person.zone_id is not None for person in persons)
        alerts = sum(person.status is PersonStatus.NON_COMPLIANT for person in persons)
        left = f"{self._camera.id} | {self._camera.area} | Na zona: {in_zone} | Alertas: {alerts}"
        right = f"{captured_at:%Y-%m-%d %H:%M:%S} | {fps:4.1f} FPS"
        cv2.putText(canvas, ascii_text(left), (8, 19), _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)
        (text_width, _), _ = cv2.getTextSize(right, _FONT, 0.5, 1)
        cv2.putText(canvas, right, (width - text_width - 8, 19), _FONT, 0.5, _WHITE, 1,
                    cv2.LINE_AA)

    @staticmethod
    def _draw_label(
        canvas: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        scale: float = 0.5,
    ) -> None:
        """Escreve um rótulo com fundo colorido, mantendo-o dentro da imagem."""
        content = ascii_text(text)
        (text_width, text_height), baseline = cv2.getTextSize(content, _FONT, scale, 1)
        top = max(y - text_height - baseline - 4, 30)  # não cobre a barra de status
        left = min(max(x, 0), max(canvas.shape[1] - text_width - 6, 0))
        bottom = top + text_height + baseline + 4
        cv2.rectangle(canvas, (left, top), (left + text_width + 6, bottom), color, -1)
        cv2.putText(canvas, content, (left + 3, bottom - baseline - 1), _FONT, scale, _BLACK, 1,
                    cv2.LINE_AA)
