"""Registro de auditoria de uma violação (Metadata Stamp da especificação)."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, replace

from safework.core.config import CameraConfig
from safework.logic.states import AlertEvent
from safework.vision.capture import mask_credentials

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
GENESIS_HASH = "0" * 64  # "hash anterior" do primeiro registro da cadeia

# Campos fora do hash: são operacionais e mudam depois da gravação.
_UNHASHED = {"id", "record_hash", "email_status", "email_sent_at", "email_error"}


class EmailStatus:
    PENDING = "pendente"
    SENT = "enviado"
    FAILED = "falhou"
    DISABLED = "desativado"
    DROPPED = "descartado"


@dataclass(frozen=True, slots=True)
class ViolationRecord:
    """Linha da trilha de auditoria. Os campos imutáveis entram no hash encadeado."""

    event_uid: str
    event_type: str
    occurred_at: str                  # YYYY-MM-DD HH:MM:SS
    warning_started_at: str | None
    camera_id: str
    camera_name: str
    camera_area: str
    camera_source: str                # IP/URL da câmera, sem usuário e senha
    zone_id: str | None
    zone_name: str | None
    track_id: int
    missing_ppe: str                  # ex.: "capacete,colete"
    infraction_type: str              # ex.: "Ausência de Capacete em Zona Crítica 1"
    elapsed_s: float
    frame_index: int
    evidence_path: str | None
    evidence_sha256: str | None
    previous_hash: str = ""
    record_hash: str = ""
    email_status: str = EmailStatus.PENDING
    email_sent_at: str | None = None
    email_error: str | None = None
    id: int | None = None

    @classmethod
    def from_event(
        cls,
        event: AlertEvent,
        camera: CameraConfig,
        evidence_path: str | None,
        evidence_sha256: str | None,
        email_status: str,
    ) -> "ViolationRecord":
        return cls(
            event_uid=str(uuid.uuid4()),
            event_type=event.type.value,
            occurred_at=event.occurred_at.strftime(DATETIME_FORMAT),
            warning_started_at=(
                event.warning_started_at.strftime(DATETIME_FORMAT)
                if event.warning_started_at
                else None
            ),
            camera_id=camera.id,
            camera_name=camera.name,
            camera_area=camera.area,
            camera_source=mask_credentials(camera.source),
            zone_id=event.zone_id,
            zone_name=event.zone_name,
            track_id=event.track_id,
            missing_ppe=",".join(sorted(item.label.lower() for item in event.missing_ppe)),
            infraction_type=event.description,
            elapsed_s=round(event.elapsed_s, 3),
            frame_index=event.frame_index,
            evidence_path=evidence_path,
            evidence_sha256=evidence_sha256,
            email_status=email_status,
        )

    def compute_hash(self) -> str:
        """SHA-256 do conteúdo imutável + hash anterior (cadeia à prova de adulteração)."""
        payload = {
            key: value for key, value in asdict(self).items() if key not in _UNHASHED
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def chained(self, previous_hash: str) -> "ViolationRecord":
        linked = replace(self, previous_hash=previous_hash)
        return replace(linked, record_hash=linked.compute_hash())
