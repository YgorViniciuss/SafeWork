"""Medição de FPS por janela deslizante."""

from __future__ import annotations

import time
from collections import deque


class FpsMeter:
    """Calcula o FPS médio dos últimos `window` frames."""

    def __init__(self, window: int = 30) -> None:
        self._ticks: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Registra um frame e devolve o FPS atual."""
        self._ticks.append(time.perf_counter())
        if len(self._ticks) < 2:
            return 0.0
        elapsed = self._ticks[-1] - self._ticks[0]
        return (len(self._ticks) - 1) / elapsed if elapsed > 0 else 0.0
