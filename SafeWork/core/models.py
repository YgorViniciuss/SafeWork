"""Modelos de domínio: os dados que trafegam entre as etapas do pipeline.

Este arquivo não depende de OpenCV nem de YOLO. Assim, os próximos módulos
(máquina de estados, auditoria, notificação) dependem apenas destes contratos.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, IntEnum

import numpy as np

Point = tuple[float, float]


class ObjectClass(IntEnum):
    """Classes de detecção definidas na especificação."""

    PERSON = 0
    HELMET = 1
    VEST = 2


class PPE(str, Enum):
    """EPIs que uma zona de risco pode exigir."""

    HELMET = "helmet"
    VEST = "vest"

    @property
    def object_class(self) -> ObjectClass:
        return ObjectClass.HELMET if self is PPE.HELMET else ObjectClass.VEST

    @property
    def label(self) -> str:
        return "Capacete" if self is PPE.HELMET else "Colete"


class PersonStatus(Enum):
    """Situação de um operador em um único frame (ainda sem temporização)."""

    OUTSIDE_ZONE = "fora_da_zona"
    COMPLIANT = "conforme"
    NON_COMPLIANT = "nao_conforme"
    # O modelo carregado não detecta o EPI exigido: não há como afirmar infração.
    UNVERIFIED = "nao_verificado"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Caixa delimitadora em pixels, no formato (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Point:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def bottom_center(self) -> Point:
        return (self.x1 + self.x2) / 2, self.y2


@dataclass(frozen=True, slots=True)
class Detection:
    """Um objeto detectado pelo modelo."""

    object_class: ObjectClass
    confidence: float
    box: BoundingBox
    track_id: int | None = None


@dataclass(frozen=True, slots=True)
class PersonAnalysis:
    """Resultado espacial de um operador em um frame."""

    detection: Detection
    status: PersonStatus
    anchor: Point
    zone_id: str | None = None
    zone_name: str | None = None
    worn_ppe: frozenset[PPE] = frozenset()
    missing_ppe: frozenset[PPE] = frozenset()
    unverified_ppe: frozenset[PPE] = frozenset()

    @property
    def track_id(self) -> int | None:
        return self.detection.track_id


@dataclass(frozen=True, slots=True)
class Frame:
    """Frame capturado, com a identificação temporal necessária à auditoria."""

    image: np.ndarray
    index: int
    timestamp: float  # segundos: relógio monotônico (ao vivo) ou tempo do vídeo (arquivo)
    captured_at: datetime


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    """Saída do Módulo 1 para cada frame. É a entrada da máquina de estados."""

    frame: Frame
    annotated_image: np.ndarray
    persons: tuple[PersonAnalysis, ...]
    detections: tuple[Detection, ...]
    inference_ms: float
