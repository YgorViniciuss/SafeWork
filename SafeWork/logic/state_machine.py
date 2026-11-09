"""Máquina de estados de UM operador, identificado pelo ID do ByteTrack.

Diagrama:

            sem EPI                 sem EPI por >= 5 s
    [0 OK] ---------> [1 WARNING] ---------------------> [2 VIOLATION]
       ^                   |                                   |
       |  conforme/fora da zona por >= clear_after_s           |
       +-------------------+-----------------------------------+

Regras de robustez:
- Histerese na saída: um único frame em que o detector "vê" um capacete não
  zera o timer. É preciso conformidade contínua por `clear_after_s`.
- Enquanto o operador aparenta estar conforme (aguardando a histerese), o
  timer não escala para violação: quem colocou o capacete aos 4,5 s não é
  punido porque a confirmação chegou aos 5,2 s.
- Se ele volta a ficar sem EPI dentro da histerese, o timer original continua,
  impedindo que oscilações reiniciem a contagem.

A classe é pura: recebe o tempo pelo frame e não lê relógio nem desenha nada,
o que a torna determinística e fácil de testar.
"""

from __future__ import annotations

from datetime import datetime

from safework.core.config import AlertConfig
from safework.core.models import PPE, Frame, PersonAnalysis, PersonStatus
from safework.logic.states import AlertEvent, AlertState, EventType, OperatorAlert


class OperatorStateMachine:
    """Controla o timer e as transições de um operador."""

    def __init__(self, track_id: int, config: AlertConfig) -> None:
        self.track_id = track_id
        self._config = config
        self.state = AlertState.OK
        self.last_seen: float | None = None
        self._warning_started_ts: float | None = None
        self._warning_started_at: datetime | None = None
        self._compliant_since: float | None = None
        self._last_violation_event_ts: float | None = None
        self._zone_id: str | None = None
        self._zone_name: str | None = None
        self._missing: frozenset[PPE] = frozenset()

    # ------------------------------------------------------------------ entrada
    def observe(self, person: PersonAnalysis, frame: Frame) -> list[AlertEvent]:
        """Processa a observação do operador neste frame e devolve os eventos gerados."""
        self.last_seen = frame.timestamp
        if person.status is PersonStatus.NON_COMPLIANT:
            return self._on_non_compliant(person, frame)
        # Conforme, fora da zona ou EPI não verificável: nenhuma infração a sustentar.
        return self._on_clear_condition(frame)

    def lost(self, frame_index: int, timestamp: float, occurred_at: datetime) -> list[AlertEvent]:
        """O operador saiu de cena. Encerra o episódio sem confirmar violação."""
        if self.state is AlertState.OK:
            return []
        event = self._event(EventType.TRACK_LOST, AlertState.OK, frame_index, timestamp,
                            occurred_at)
        self._reset()
        return [event]

    # ------------------------------------------------------------------ consulta
    @property
    def is_idle(self) -> bool:
        """True quando não há nada pendente e a máquina pode ser descartada."""
        return self.state is AlertState.OK and self._compliant_since is None

    def snapshot(self, timestamp: float) -> OperatorAlert:
        elapsed = self._elapsed(timestamp)
        remaining = (
            max(self._config.warning_seconds - elapsed, 0.0)
            if self.state is AlertState.WARNING
            else 0.0
        )
        return OperatorAlert(
            track_id=self.track_id,
            state=self.state,
            elapsed_s=elapsed,
            remaining_s=remaining,
            clearing=self._compliant_since is not None,
            zone_name=self._zone_name,
            missing_ppe=self._missing,
        )

    # ------------------------------------------------------------------ transições
    def _on_non_compliant(self, person: PersonAnalysis, frame: Frame) -> list[AlertEvent]:
        self._compliant_since = None
        self._zone_id = person.zone_id
        self._zone_name = person.zone_name
        self._missing = person.missing_ppe
        now = frame.timestamp
        events: list[AlertEvent] = []

        if self.state is AlertState.OK:
            self._warning_started_ts = now
            self._warning_started_at = frame.captured_at
            events.append(self._transition(EventType.WARNING_STARTED, AlertState.WARNING, frame))

        if (
            self.state is AlertState.WARNING
            and self._elapsed(now) >= self._config.warning_seconds
        ):
            events.append(
                self._transition(EventType.VIOLATION_CONFIRMED, AlertState.VIOLATION, frame)
            )
            self._last_violation_event_ts = now
        elif (
            self.state is AlertState.VIOLATION
            and self._config.violation_repeat_s > 0
            and self._last_violation_event_ts is not None
            and now - self._last_violation_event_ts >= self._config.violation_repeat_s
        ):
            events.append(
                self._transition(EventType.VIOLATION_REMINDER, AlertState.VIOLATION, frame)
            )
            self._last_violation_event_ts = now
        return events

    def _on_clear_condition(self, frame: Frame) -> list[AlertEvent]:
        if self.state is AlertState.OK:
            self._compliant_since = None
            return []
        if self._compliant_since is None:
            self._compliant_since = frame.timestamp
        if frame.timestamp - self._compliant_since < self._config.clear_after_s:
            return []
        event = self._transition(EventType.RESOLVED, AlertState.OK, frame)
        self._reset()
        return [event]

    # ------------------------------------------------------------------ auxiliares
    def _transition(self, kind: EventType, new_state: AlertState, frame: Frame) -> AlertEvent:
        event = self._event(kind, new_state, frame.index, frame.timestamp, frame.captured_at)
        self.state = new_state
        return event

    def _event(
        self,
        kind: EventType,
        new_state: AlertState,
        frame_index: int,
        timestamp: float,
        occurred_at: datetime,
    ) -> AlertEvent:
        return AlertEvent(
            type=kind,
            track_id=self.track_id,
            previous_state=self.state,
            state=new_state,
            zone_id=self._zone_id,
            zone_name=self._zone_name,
            missing_ppe=self._missing,
            occurred_at=occurred_at,
            timestamp=timestamp,
            frame_index=frame_index,
            warning_started_at=self._warning_started_at,
            elapsed_s=self._elapsed(timestamp),
        )

    def _elapsed(self, timestamp: float) -> float:
        if self._warning_started_ts is None:
            return 0.0
        return max(timestamp - self._warning_started_ts, 0.0)

    def _reset(self) -> None:
        self.state = AlertState.OK
        self._warning_started_ts = None
        self._warning_started_at = None
        self._compliant_since = None
        self._last_violation_event_ts = None
        self._missing = frozenset()
