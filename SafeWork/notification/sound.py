"""Alarme sonoro local, tocado em thread própria para não travar o vídeo.

Backends (escolhidos por `sound.backend` no settings.yaml):
    winsound  Windows, sem dependências (winsound.Beep)
    pygame    Linux/macOS/Windows, tom senoidal gerado com NumPy
    bell      caractere BEL no terminal (último recurso)
    none      silencioso
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Protocol

import numpy as np

from safework.core.config import SoundConfig
from safework.logic.states import AlertState

logger = logging.getLogger(__name__)


class SoundBackend(Protocol):
    name: str

    def beep(self, frequency_hz: int, duration_ms: int) -> None:
        """Toca um tom. Pode bloquear durante a duração (roda na thread do alarme)."""


class SilentBackend:
    name = "none"

    def beep(self, frequency_hz: int, duration_ms: int) -> None:
        time.sleep(duration_ms / 1000)


class BellBackend:
    name = "bell"

    def beep(self, frequency_hz: int, duration_ms: int) -> None:
        sys.stdout.write("\a")
        sys.stdout.flush()
        time.sleep(duration_ms / 1000)


class WinsoundBackend:
    name = "winsound"

    def __init__(self) -> None:
        import winsound  # disponível apenas no Windows

        self._winsound = winsound

    def beep(self, frequency_hz: int, duration_ms: int) -> None:
        self._winsound.Beep(frequency_hz, duration_ms)


class PygameBackend:
    name = "pygame"
    _SAMPLE_RATE = 44_100

    def __init__(self) -> None:
        import os

        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        pygame.mixer.init(frequency=self._SAMPLE_RATE, size=-16, channels=1)
        self._pygame = pygame
        self._cache: dict[tuple[int, int], object] = {}

    def _tone(self, frequency_hz: int, duration_ms: int) -> object:
        key = (frequency_hz, duration_ms)
        if key not in self._cache:
            rate, _, channels = self._pygame.mixer.get_init()
            samples = int(rate * duration_ms / 1000)
            t = np.arange(samples) / rate
            wave = np.sin(2 * np.pi * frequency_hz * t)
            fade = min(samples // 10, int(rate * 0.005))  # evita estalos nas bordas
            if fade:
                ramp = np.linspace(0.0, 1.0, fade)
                wave[:fade] *= ramp
                wave[-fade:] *= ramp[::-1]
            pcm = (wave * 0.6 * 32767).astype(np.int16)
            if channels > 1:
                pcm = np.repeat(pcm[:, None], channels, axis=1)
            self._cache[key] = self._pygame.sndarray.make_sound(np.ascontiguousarray(pcm))
        return self._cache[key]

    def beep(self, frequency_hz: int, duration_ms: int) -> None:
        self._tone(frequency_hz, duration_ms).play()
        time.sleep(duration_ms / 1000)


def create_backend(name: str) -> SoundBackend:
    """Cria o backend pedido; em 'auto' tenta winsound, pygame e por fim bell."""
    if name == "none":
        return SilentBackend()
    if name == "bell":
        return BellBackend()

    candidates = {"winsound": [WinsoundBackend], "pygame": [PygameBackend]}.get(name)
    if candidates is None:  # auto
        candidates = [WinsoundBackend, PygameBackend] if sys.platform == "win32" else [PygameBackend]
    for backend_class in candidates:
        try:
            return backend_class()
        except Exception as exc:  # módulo ausente ou sem dispositivo de áudio
            logger.warning("Áudio %s indisponível (%s).", backend_class.name, exc)
    logger.warning("Usando o sinal do terminal como alarme sonoro.")
    return BellBackend()


class AlarmPlayer:
    """Toca o padrão sonoro do estado mais grave enquanto houver alerta.

    O laço de vídeo só chama `set_level()`, que retorna na hora; quem toca é
    uma thread daemon. Estado 1: bipe intermitente. Estado 2: bipe agudo e rápido.
    """

    def __init__(self, backend: SoundBackend, config: SoundConfig) -> None:
        self._backend = backend
        self._patterns = {
            AlertState.WARNING: config.warning_tone,
            AlertState.VIOLATION: config.violation_tone,
        }
        self._level = AlertState.OK
        self._condition = threading.Condition()
        self._stopped = False
        self._thread: threading.Thread | None = None
        self.beeps = 0  # contador, útil para testes e diagnóstico

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def level(self) -> AlertState:
        return self._level

    def start(self) -> "AlarmPlayer":
        if self._thread is None:
            self._stopped = False
            self._thread = threading.Thread(target=self._loop, name="alarm", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "AlarmPlayer":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def set_level(self, level: AlertState) -> None:
        if level is self._level:
            return
        with self._condition:
            self._level = level
            self._condition.notify_all()

    def _loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stopped or self._level is not AlertState.OK
                )
                if self._stopped:
                    return
                frequency, duration, pause = self._patterns[self._level]
            try:
                self._backend.beep(frequency, duration)
                self.beeps += 1
            except Exception as exc:
                logger.error("Falha no áudio %s (%s). Alarme silenciado.", self._backend.name, exc)
                self._backend = SilentBackend()
            with self._condition:  # pausa interrompível por stop() ou mudança de nível
                self._condition.wait(pause / 1000)
