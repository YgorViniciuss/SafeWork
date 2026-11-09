"""Motor de alertas: mantém uma máquina de estados por operador (ID do ByteTrack)."""

from __future__ import annotations

import logging
from datetime import datetime

from safework.core.config import AlertConfig
from safework.core.models import FrameAnalysis, PersonStatus
from safework.logic.state_machine import OperatorStateMachine
from safework.logic.states import AlertEvent, AlertSnapshot

logger = logging.getLogger(__name__)


class AlertEngine:
    """Recebe a análise de cada frame do Módulo 1 e devolve um `AlertSnapshot`.

    Só operadores com alguma pendência têm máquina em memória; os demais são
    descartados assim que voltam ao Estado 0, mantendo o custo constante.
    """

    def __init__(self, config: AlertConfig) -> None:
        self._config = config
        self._machines: dict[int, OperatorStateMachine] = {}
        self._warned_untracked = False

    @property
    def active_tracks(self) -> int:
        return len(self._machines)

    def update(self, analysis: FrameAnalysis) -> AlertSnapshot:
        frame = analysis.frame
        events: list[AlertEvent] = []
        seen: set[int] = set()

        for person in analysis.persons:
            track_id = person.track_id
            if track_id is None:
                self._warn_untracked(person.status)
                continue
            seen.add(track_id)
            machine = self._machines.get(track_id)
            if machine is None:
                if person.status is not PersonStatus.NON_COMPLIANT:
                    continue  # nada a acompanhar
                machine = OperatorStateMachine(track_id, self._config)
                self._machines[track_id] = machine
            events.extend(machine.observe(person, frame))
            if machine.is_idle:
                del self._machines[track_id]

        events.extend(self._expire_missing(seen, frame.index, frame.timestamp, frame.captured_at))
        return self._snapshot(frame.timestamp, events)

    def expire_all(self, frame_index: int = -1, occurred_at: datetime | None = None) -> AlertSnapshot:
        """Encerra todos os episódios (ex.: a câmera caiu e nada pode ser verificado)."""
        occurred_at = occurred_at or datetime.now()
        events: list[AlertEvent] = []
        for machine in self._machines.values():
            events.extend(machine.lost(frame_index, machine.last_seen or 0.0, occurred_at))
        self._machines.clear()
        return AlertSnapshot(timestamp=0.0, operators={}, events=tuple(events))

    # ------------------------------------------------------------------ auxiliares
    def _expire_missing(
        self,
        seen: set[int],
        frame_index: int,
        timestamp: float,
        occurred_at: datetime,
    ) -> list[AlertEvent]:
        """Descarta operadores que não aparecem há mais que `lost_track_timeout_s`.

        Ausências curtas (oclusão, falha pontual do detector) não encerram o
        episódio: o ByteTrack costuma recuperar o mesmo ID.
        """
        events: list[AlertEvent] = []
        for track_id, machine in list(self._machines.items()):
            if track_id in seen or machine.last_seen is None:
                continue
            if timestamp - machine.last_seen >= self._config.lost_track_timeout_s:
                events.extend(machine.lost(frame_index, timestamp, occurred_at))
                del self._machines[track_id]
        return events

    def _snapshot(self, timestamp: float, events: list[AlertEvent]) -> AlertSnapshot:
        operators = {
            track_id: machine.snapshot(timestamp)
            for track_id, machine in self._machines.items()
        }
        return AlertSnapshot(timestamp=timestamp, operators=operators, events=tuple(events))

    def _warn_untracked(self, status: PersonStatus) -> None:
        if status is PersonStatus.NON_COMPLIANT and not self._warned_untracked:
            logger.warning(
                "Operador sem ID de rastreamento: o timer por operador exige "
                "detector.tracking: true no settings.yaml."
            )
            self._warned_untracked = True
