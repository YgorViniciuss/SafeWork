"""Zonas de risco (ROI) poligonais e sua persistência em JSON.

Os vértices são salvos normalizados (0 a 1). Assim, a zona continua válida se
a resolução da câmera ou a largura de processamento mudar.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from safework.core.models import PPE, Point


@dataclass(frozen=True)
class RiskZone:
    """Polígono de risco e os EPIs exigidos dentro dele."""

    id: str
    name: str
    points: tuple[Point, ...]
    required_ppe: frozenset[PPE] = frozenset({PPE.HELMET, PPE.VEST})

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(f"A zona '{self.name}' precisa de ao menos 3 vértices.")
        if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in self.points):
            raise ValueError(f"A zona '{self.name}' tem vértices fora do intervalo 0 a 1.")

    def polygon_px(self, width: int, height: int) -> np.ndarray:
        """Vértices em pixels, no formato exigido pelas funções do OpenCV."""
        pixels = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in self.points]
        return np.array(pixels, dtype=np.int32)

    def contains(self, point: Point, width: int, height: int) -> bool:
        """Teste ponto-no-polígono. A própria fronteira conta como dentro da zona."""
        contour = self.polygon_px(width, height).astype(np.float32)
        return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0

    def mask(self, width: int, height: int) -> np.ndarray:
        """Definição matricial da ROI: 255 dentro do polígono e 0 fora."""
        matrix = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(matrix, [self.polygon_px(width, height)], 255)
        return matrix

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "required_ppe": sorted(item.value for item in self.required_ppe),
            "points": [[round(x, 5), round(y, 5)] for x, y in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RiskZone":
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            points=tuple((float(x), float(y)) for x, y in data["points"]),
            required_ppe=frozenset(PPE(item) for item in data.get("required_ppe", ["helmet"])),
        )


class ZoneRepository:
    """Carrega e salva as zonas de risco."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[RiskZone]:
        if not self._path.exists():
            return []
        with self._path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        zones = [RiskZone.from_dict(item) for item in data.get("zones", [])]
        identifiers = [zone.id for zone in zones]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"IDs de zona duplicados em {self._path}")
        return zones

    def save(self, zones: list[RiskZone]) -> None:
        """Grava em arquivo temporário e substitui: uma falha não corrompe o original."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        payload = {"zones": [zone.to_dict() for zone in zones]}
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, self._path)
