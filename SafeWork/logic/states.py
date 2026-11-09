"""Estados, eventos e fotografias (snapshots) da máquina de alerta.

Estes tipos são o contrato entre o Módulo 2 e os módulos seguintes:
a auditoria (Módulo 3) e o e-mail (Módulo 4) consomem `AlertEvent`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, IntEnum

from safework.core.models import PPE


class AlertState(IntEnum):
    """Estados da especificação. A ordem numérica indica a gravidade."""

    OK = 0         # Estado 0: na ROI com EPI (ou sem pendências)
    WARNING = 1    # Estado 1: infração inicial, contagem de 5 s em andamento
    VIOLATION = 2  # Estado 2: infração confirmada, dispara evidência e auditoria


class EventType(str, Enum):
    """Transições relevantes publicadas no barramento de eventos."""

    WARNING_STARTED = "infracao_inicial"
    VIOLATION_CONFIRMED = "infracao_confirmada"
    VIOLATION_REMINDER = "infracao_persistente"
    RESOLVED = "regularizado"
    TRACK_LOST = "operador_perdido"


@dataclass(frozen=True, slots=True)
class AlertEvent:
    """Fato ocorrido com um operador, com os metadados exigidos pela auditoria."""

    type: EventType
    track_id: int
    previous_state: AlertState
    state: AlertState
    zone_id: str | None
    zone_name: str | None
    missing_ppe: frozenset[PPE]
    occurred_at: datetime              # data e hora de parede (YYYY-MM-DD HH:MM:SS)
    timestamp: float                   # tempo do pipeline, usado nas contas do timer
    frame_index: int
    warning_started_at: datetime | None
    elapsed_s: float                   # tempo desde o início da infração

    @property
    def description(self) -> str:
        """Texto do tipo de infração, ex.: 'Ausência de Capacete em Zona Crítica 1'."""
        items = sorted(item.label for item in self.missing_ppe)
        if not items:
            missing = "EPI"
        elif len(items) == 1:
            missing = items[0]
        else:
            missing = ", ".join(items[:-1]) + " e " + items[-1]
        zone = self.zone_name or "zona de risco"
        return f"Ausência de {missing} em {zone}"


@dataclass(frozen=True, slots=True)
class OperatorAlert:
    """Situação de alerta de um operador no instante atual (usada para desenhar)."""

    track_id: int
    state: AlertState
    elapsed_s: float      # tempo desde o início da infração
    remaining_s: float    # contagem regressiva até o Estado 2 (0 fora do Estado 1)
    clearing: bool        # em conformidade, aguardando a histerese para voltar ao OK
    zone_name: str | None
    missing_ppe: frozenset[PPE]


@dataclass(frozen=True, slots=True)
class AlertSnapshot:
    """Resultado do motor para um frame: estado de cada operador e eventos novos."""

    timestamp: float
    operators: dict[int, OperatorAlert]
    events: tuple[AlertEvent, ...]

    @property
    def highest_state(self) -> AlertState:
        """Estado mais grave entre os operadores (define o alarme sonoro)."""
        return max((item.state for item in self.operators.values()), default=AlertState.OK)
