"""Barramento de eventos (padrão Observer).

O motor não conhece quem consome os eventos. Os Módulos 3 (auditoria) e 4
(e-mail) apenas se inscrevem aqui, sem alterar o código da lógica.

Os ouvintes rodam no laço de vídeo: devem ser rápidos e delegar qualquer I/O
lento (SMTP, por exemplo) a uma thread própria.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Protocol

import numpy as np

from safework.core.config import CameraConfig
from safework.logic.states import AlertEvent, EventType

logger = logging.getLogger(__name__)


class AlertListener(Protocol):
    def handle(self, event: AlertEvent, evidence: np.ndarray | None) -> None:
        """Recebe o evento e o frame com overlay (None se não houver frame)."""


class EventBus:
    """Distribui eventos aos ouvintes, isolando falhas de cada um."""

    def __init__(self) -> None:
        self._subscriptions: list[tuple[AlertListener, frozenset[EventType] | None]] = []

    def subscribe(
        self,
        listener: AlertListener,
        types: Iterable[EventType] | None = None,
    ) -> None:
        """Inscreve um ouvinte. Sem `types`, ele recebe todos os eventos."""
        self._subscriptions.append((listener, frozenset(types) if types else None))

    def publish(self, event: AlertEvent, evidence: np.ndarray | None = None) -> None:
        for listener, types in self._subscriptions:
            if types is not None and event.type not in types:
                continue
            try:
                listener.handle(event, evidence)
            except Exception:  # um ouvinte com defeito não pode derrubar o monitoramento
                logger.exception("Falha no ouvinte %s", type(listener).__name__)


class LoggingAlertListener:
    """Registra cada evento no log estruturado (trilha mínima até o Módulo 3)."""

    _LEVELS = {
        EventType.WARNING_STARTED: logging.WARNING,
        EventType.VIOLATION_CONFIRMED: logging.ERROR,
        EventType.VIOLATION_REMINDER: logging.ERROR,
        EventType.RESOLVED: logging.INFO,
        EventType.TRACK_LOST: logging.INFO,
    }

    def __init__(self, camera: CameraConfig) -> None:
        self._camera = camera
        self._logger = logging.getLogger("safework.alerts")

    def handle(self, event: AlertEvent, evidence: np.ndarray | None) -> None:
        self._logger.log(
            self._LEVELS[event.type],
            "%s | %s | %s | operador #%d | %s | %.1fs | %s",
            event.type.value.upper(),
            self._camera.id,
            self._camera.area,
            event.track_id,
            event.description,
            event.elapsed_s,
            f"{event.occurred_at:%Y-%m-%d %H:%M:%S}",
        )
