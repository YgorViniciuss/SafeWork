"""Orquestração do Módulo 1: captura -> pré-processa -> detecta -> analisa -> desenha."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import replace

from safework.core.models import Frame, FrameAnalysis
from safework.utils.fps import FpsMeter
from safework.vision.capture import VideoSource
from safework.vision.detector import Detector
from safework.vision.preprocessing import FramePreprocessor
from safework.vision.renderer import FrameRenderer
from safework.vision.spatial import SpatialAnalyzer


class VisionPipeline:
    """Liga as etapas por injeção de dependência, sem conhecer suas implementações."""

    def __init__(
        self,
        preprocessor: FramePreprocessor,
        detector: Detector,
        analyzer: SpatialAnalyzer,
        renderer: FrameRenderer,
    ) -> None:
        self._preprocessor = preprocessor
        self._detector = detector
        self._analyzer = analyzer
        self._renderer = renderer
        self._fps = FpsMeter()

    def process(self, frame: Frame) -> FrameAnalysis:
        """Executa todas as etapas para um frame."""
        image = self._preprocessor.process(frame.image)
        height, width = image.shape[:2]

        started = time.perf_counter()
        detections = self._detector.detect(image)
        inference_ms = (time.perf_counter() - started) * 1000

        persons = self._analyzer.analyze(
            detections, width, height, self._detector.supported_classes
        )
        annotated = self._renderer.render(
            image, persons, detections, frame.captured_at, self._fps.tick()
        )
        return FrameAnalysis(
            frame=replace(frame, image=image),
            annotated_image=annotated,
            persons=tuple(persons),
            detections=tuple(detections),
            inference_ms=inference_ms,
        )

    def stream(self, source: VideoSource) -> Iterator[FrameAnalysis | None]:
        """Gera uma análise por frame.

        Gera None quando não há frame novo (câmera reconectando, por exemplo),
        para que a interface siga responsiva. Termina quando a fonte encerra.
        """
        while True:
            frame = source.read(timeout=0.1)
            if frame is not None:
                yield self.process(frame)
            elif source.finished:
                return
            else:
                yield None
