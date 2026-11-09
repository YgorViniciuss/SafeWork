"""Protocolo de evidência: ouvinte do evento VIOLATION_CONFIRMED (Módulo 2).

Sequência para cada infração confirmada:
    1. Grava o print (frame com caixa + ROI + alerta + data/hora) em evidences/.
    2. Registra os metadados na trilha de auditoria SQLite (data/).
    3. Entrega o registro ao notificador, que envia o e-mail em outra thread.

Os passos 1 e 2 são locais e rápidos (milissegundos, uma vez por episódio);
o passo 3, que depende da rede, nunca roda no laço de vídeo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from safework.audit.evidence import EvidenceStore
from safework.audit.records import ViolationRecord
from safework.audit.repository import AuditRepository
from safework.core.config import CameraConfig
from safework.logic.states import AlertEvent

logger = logging.getLogger(__name__)


class ViolationNotifier(Protocol):
    initial_status: str

    def submit(self, record: ViolationRecord, attachment: Path | None) -> None:
        """Agenda a notificação. Deve retornar imediatamente."""


class ViolationAuditor:
    """Ouvinte do EventBus que executa o protocolo de evidência e auditoria."""

    def __init__(
        self,
        store: EvidenceStore,
        repository: AuditRepository,
        notifier: ViolationNotifier,
        camera: CameraConfig,
    ) -> None:
        self._store = store
        self._repository = repository
        self._notifier = notifier
        self._camera = camera

    def handle(self, event: AlertEvent, evidence: np.ndarray | None) -> ViolationRecord:
        path: Path | None = None
        digest: str | None = None
        if evidence is None:
            logger.warning("Violação do operador #%d sem frame de evidência.", event.track_id)
        else:
            try:
                path, digest = self._store.save(evidence, event.track_id, event.occurred_at)
            except Exception:  # sem disco, o registro ainda deve existir
                logger.exception("Falha ao gravar a evidência do operador #%d.", event.track_id)

        record = self._repository.append(
            ViolationRecord.from_event(
                event,
                self._camera,
                evidence_path=EvidenceStore.relative(path) if path else None,
                evidence_sha256=digest,
                email_status=self._notifier.initial_status,
            )
        )
        logger.info(
            "Auditoria #%d registrada: %s | evidência: %s",
            record.id,
            record.infraction_type,
            record.evidence_path or "nenhuma",
        )
        self._notifier.submit(record, path)
        return record
