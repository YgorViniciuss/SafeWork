"""Etapa 1 - Input: captura de webcam, RTSP ou arquivo em thread dedicada.

Fontes ao vivo: a thread lê continuamente e guarda só o frame mais recente.
Se a inferência for mais lenta que a câmera, frames antigos são descartados,
o que evita acumular atraso (essencial para um alerta de segurança).

Arquivos: nenhum frame é descartado e o timestamp é o tempo do próprio vídeo,
o que torna os testes de homologação reproduzíveis.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2

from safework.core.config import CameraConfig, resolve_path
from safework.core.models import Frame

logger = logging.getLogger(__name__)

# Força RTSP sobre TCP, que evita imagem corrompida por perda de pacotes UDP.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

_NETWORK_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")


def mask_credentials(source: int | str) -> str:
    """Remove usuário e senha de uma URL antes de escrevê-la em logs."""
    return re.sub(r"(?<=://)[^/@]+@", "***@", str(source))


def _default_capture_factory(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, str) and source.lower().startswith(_NETWORK_SCHEMES):
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)


class VideoSource:
    """Fonte de vídeo com leitura em segundo plano e reconexão automática."""

    def __init__(
        self,
        config: CameraConfig,
        capture_factory: Callable[[int | str], Any] = _default_capture_factory,
    ) -> None:
        self._config = config
        self._capture_factory = capture_factory
        self._source = self._normalize_source(config.source)
        self._is_file = isinstance(self._source, str) and not self._source.lower().startswith(
            _NETWORK_SCHEMES
        )
        self._condition = threading.Condition()
        self._latest: Frame | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_index = 0
        self.finished = False
        self.error: str | None = None

    @staticmethod
    def _normalize_source(source: int | str) -> int | str:
        """Converte "0" em 0 e resolve caminhos relativos de arquivos."""
        if isinstance(source, str) and source.isdigit():
            return int(source)
        if isinstance(source, str) and "://" not in source:
            return str(resolve_path(Path(source)))
        return source

    # ------------------------------------------------------------------ ciclo de vida
    def start(self) -> "VideoSource":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._reader_loop, name="video-reader", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> "VideoSource":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ consumo
    def read(self, timeout: float = 0.5) -> Frame | None:
        """Devolve o próximo frame, ou None se nenhum chegar dentro do timeout."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._latest is not None or self.finished or self._stop_event.is_set(),
                timeout=timeout,
            )
            frame, self._latest = self._latest, None
            self._condition.notify_all()  # libera o leitor de arquivo
            return frame

    # ------------------------------------------------------------------ thread de leitura
    def _reader_loop(self) -> None:
        attempts = 0
        delay = self._config.reconnect_delay_s
        label = mask_credentials(self._source)

        while not self._stop_event.is_set():
            capture = self._open()
            if capture is None:
                if self._is_file:
                    self._finish(f"Não foi possível abrir o arquivo: {label}")
                    return
                attempts += 1
                limit = self._config.max_reconnect_attempts
                if limit and attempts >= limit:
                    self._finish(f"Fonte indisponível após {attempts} tentativas: {label}")
                    return
                logger.warning("Fonte indisponível (%s). Nova tentativa em %.1fs.", label, delay)
                self._stop_event.wait(delay)
                delay = min(delay * 2, self._config.max_reconnect_delay_s)
                continue

            logger.info("Fonte de vídeo aberta: %s", label)
            attempts = 0
            delay = self._config.reconnect_delay_s
            self._consume(capture)
            capture.release()

            if self._is_file:
                self._finish()
                return
            if not self._stop_event.is_set():
                logger.warning("Conexão perdida (%s). Reconectando...", label)
                self._stop_event.wait(delay)

    def _open(self) -> Any | None:
        capture = self._capture_factory(self._source)
        if not capture.isOpened():
            capture.release()
            return None
        if not self._is_file:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduz a latência quando suportado
        return capture

    def _consume(self, capture: Any) -> None:
        """Lê frames até a fonte falhar, o vídeo acabar ou stop() ser chamado."""
        file_fps = capture.get(cv2.CAP_PROP_FPS) if self._is_file else 0.0
        file_index = 0

        while not self._stop_event.is_set():
            ok, image = capture.read()
            if not ok or image is None:
                return

            if self._is_file and file_fps > 0:
                timestamp = file_index / file_fps
            else:
                timestamp = time.monotonic()
            file_index += 1

            frame = Frame(
                image=image,
                index=self._frame_index,
                timestamp=timestamp,
                captured_at=datetime.now(),
            )
            self._frame_index += 1

            with self._condition:
                if self._is_file:  # sem perda: aguarda o consumidor pegar o frame anterior
                    self._condition.wait_for(
                        lambda: self._latest is None or self._stop_event.is_set()
                    )
                self._latest = frame
                self._condition.notify_all()

    def _finish(self, error: str | None = None) -> None:
        with self._condition:
            self.error = error
            self.finished = True
            self._condition.notify_all()
        if error:
            logger.error(error)
