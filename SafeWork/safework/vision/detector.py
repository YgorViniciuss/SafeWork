"""Etapa 3 - Inferência: detecção multiclasse com YOLOv8."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from safework.core.config import DetectorConfig, resolve_path
from safework.core.models import BoundingBox, Detection, ObjectClass

logger = logging.getLogger(__name__)


class Detector(Protocol):
    """Contrato do detector: permite trocar o YOLO por um dublê nos testes."""

    supported_classes: frozenset[ObjectClass]

    def detect(self, image: np.ndarray) -> list[Detection]:
        ...


def _normalize_name(name: str) -> str:
    return name.lower().replace("-", " ").replace("_", " ").strip()


def build_class_map(
    model_names: dict[int, str],
    aliases: dict[str, list[str]],
) -> dict[int, ObjectClass]:
    """Relaciona os índices do modelo às classes do SafeWork pelo nome.

    Pesos COCO, por exemplo, têm "bicycle" no índice 1. Usar o índice direto
    faria uma bicicleta ser tratada como capacete. Com o vínculo por nome,
    esses pesos resultam apenas em {0: PERSON}.
    """
    lookup = {
        _normalize_name(alias): ObjectClass[class_name.upper()]
        for class_name, names in aliases.items()
        for alias in names
    }
    return {
        index: lookup[_normalize_name(name)]
        for index, name in model_names.items()
        if _normalize_name(name) in lookup
    }


class YoloDetector:
    """Encapsula o Ultralytics YOLOv8 e devolve objetos de domínio."""

    def __init__(self, config: DetectorConfig) -> None:
        from ultralytics import YOLO  # importação tardia: carregar o torch é demorado

        self._config = config
        weights = self._resolve_weights()
        logger.info("Carregando pesos: %s", weights)
        self._model = YOLO(str(weights))

        self._class_map = build_class_map(dict(self._model.names), config.class_aliases)
        self.supported_classes = frozenset(self._class_map.values())
        self._class_filter = sorted(self._class_map)

        if ObjectClass.PERSON not in self.supported_classes:
            raise ValueError(f"Os pesos não têm classe de pessoa: {self._model.names}")
        missing = {ObjectClass.HELMET, ObjectClass.VEST} - self.supported_classes
        if missing:
            logger.warning(
                "Modelo sem detecção de: %s. Esses EPIs ficarão como NÃO VERIFICADOS.",
                ", ".join(item.name for item in sorted(missing)),
            )

    def _resolve_weights(self) -> Path:
        primary = resolve_path(self._config.weights)
        if primary.exists():
            return primary
        if self._config.fallback_weights is None:
            raise FileNotFoundError(f"Pesos não encontrados: {primary}")
        fallback = resolve_path(self._config.fallback_weights)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "Pesos customizados ausentes (%s). Usando %s: somente pessoas serão detectadas.",
            primary,
            fallback.name,
        )
        return fallback  # o Ultralytics baixa automaticamente os pesos oficiais

    def detect(self, image: np.ndarray) -> list[Detection]:
        options = {
            "conf": self._config.confidence,
            "iou": self._config.iou,
            "imgsz": self._config.image_size,
            "classes": self._class_filter,
            "device": self._config.device,
            "verbose": False,
        }
        if self._config.tracking:
            results = self._model.track(
                image, persist=True, tracker=self._config.tracker, **options
            )
        else:
            results = self._model.predict(image, **options)
        return self._to_detections(results[0].boxes)

    def _to_detections(self, boxes: object) -> list[Detection]:
        if boxes is None or len(boxes) == 0:
            return []
        coordinates = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = (
            boxes.id.cpu().numpy().astype(int).tolist()
            if boxes.id is not None
            else [None] * len(class_ids)
        )
        return [
            Detection(
                object_class=self._class_map[class_id],
                confidence=float(confidence),
                box=BoundingBox(*map(float, xyxy)),
                track_id=track_id,
            )
            for xyxy, confidence, class_id, track_id in zip(
                coordinates, confidences, class_ids, track_ids
            )
        ]
