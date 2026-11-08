"""Composição da aplicação: cria os objetos e conduz o laço de exibição."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from safework.core.config import Settings, resolve_path
from safework.core.models import FrameAnalysis
from safework.vision.capture import VideoSource
from safework.vision.detector import YoloDetector
from safework.vision.pipeline import VisionPipeline
from safework.vision.preprocessing import FramePreprocessor
from safework.vision.renderer import FrameRenderer
from safework.vision.roi import ZoneRepository
from safework.vision.roi_editor import RoiEditor
from safework.vision.spatial import SpatialAnalyzer

logger = logging.getLogger(__name__)

_EXIT_KEYS = (ord("q"), ord("Q"), 27)


class SafeWorkApp:
    """Ponto de entrada orientado a objetos do sistema."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._zones = ZoneRepository(resolve_path(settings.zones_file))

    def edit_zones(self) -> int:
        """Captura um frame da fonte e abre o editor de ROI."""
        preprocessor = FramePreprocessor(self._settings.preprocessing)
        with VideoSource(self._settings.camera) as source:
            frame = source.read(timeout=15.0)
        if frame is None:
            logger.error("Nenhum frame recebido da fonte: %s", source.error or "tempo esgotado")
            return 1
        RoiEditor(self._zones).run(preprocessor.process(frame.image))
        return 0

    def run(
        self,
        headless: bool = False,
        output: str | None = None,
        max_frames: int = 0,
    ) -> int:
        """Executa o monitoramento até o usuário sair ou a fonte terminar."""
        zones = self._zones.load()
        if not zones:
            logger.warning("Nenhuma zona de risco configurada. Use --edit-zones.")

        settings = self._settings
        pipeline = VisionPipeline(
            preprocessor=FramePreprocessor(settings.preprocessing),
            detector=YoloDetector(settings.detector),
            analyzer=SpatialAnalyzer(settings.spatial, zones),
            renderer=FrameRenderer(settings.camera, settings.display, zones),
        )

        writer: cv2.VideoWriter | None = None
        processed = 0
        total_ms = 0.0
        source = VideoSource(settings.camera)
        try:
            with source:
                for result in pipeline.stream(source):
                    if result is not None:
                        processed += 1
                        total_ms += result.inference_ms
                        if output:
                            writer = writer or self._create_writer(output, result)
                            writer.write(result.annotated_image)
                        if not headless:
                            cv2.imshow(settings.display.window_name, result.annotated_image)
                    if not headless and self._user_requested_exit():
                        break
                    if max_frames and processed >= max_frames:
                        break
        except KeyboardInterrupt:
            logger.info("Interrompido pelo usuário.")
        finally:
            if writer is not None:
                writer.release()
            if not headless:
                cv2.destroyAllWindows()

        if processed:
            logger.info(
                "%d frame(s) processado(s). Inferência média: %.1f ms.",
                processed,
                total_ms / processed,
            )
        return 1 if source.error else 0

    def _user_requested_exit(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        window = self._settings.display.window_name
        try:
            closed = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            closed = False  # a janela ainda não foi criada (sem frames até agora)
        return key in _EXIT_KEYS or closed

    @staticmethod
    def _create_writer(output: str, result: FrameAnalysis) -> cv2.VideoWriter:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = result.annotated_image.shape[:2]
        return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))
