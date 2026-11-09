"""Leitura e validação do arquivo config/settings.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Configuração ausente ou inválida."""


def resolve_path(path: str | Path) -> Path:
    """Resolve um caminho relativo a partir da raiz do projeto."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


@dataclass(frozen=True)
class CameraConfig:
    source: int | str = 0
    id: str = "CAM-01"
    name: str = "Camera"
    area: str = "Setor"
    reconnect_delay_s: float = 2.0
    max_reconnect_delay_s: float = 30.0
    max_reconnect_attempts: int = 0


@dataclass(frozen=True)
class PreprocessingConfig:
    target_width: int = 960
    denoise: bool = False
    denoise_kernel: int = 3

    def __post_init__(self) -> None:
        if self.target_width < 0:
            raise ConfigError("preprocessing.target_width não pode ser negativo.")
        if self.denoise_kernel < 1 or self.denoise_kernel % 2 == 0:
            raise ConfigError("preprocessing.denoise_kernel deve ser ímpar e positivo.")


@dataclass(frozen=True)
class DetectorConfig:
    weights: str = "models/safework_ppe.pt"
    fallback_weights: str | None = "models/yolov8n.pt"
    confidence: float = 0.45
    iou: float = 0.50
    image_size: int = 640
    device: int | str | None = None
    tracking: bool = True
    tracker: str = "bytetrack.yaml"
    class_aliases: dict[str, list[str]] = field(default_factory=lambda: {
        "person": ["person"],
        "helmet": ["helmet", "hardhat"],
        "vest": ["vest"],
    })

    def __post_init__(self) -> None:
        for name, value in (("confidence", self.confidence), ("iou", self.iou)):
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"detector.{name} deve estar entre 0 e 1.")


@dataclass(frozen=True)
class SpatialConfig:
    anchor: str = "centroid"
    helmet_region: tuple[float, float] = (0.0, 0.40)
    vest_region: tuple[float, float] = (0.15, 0.80)

    def __post_init__(self) -> None:
        if self.anchor not in ("centroid", "bottom_center"):
            raise ConfigError("spatial.anchor deve ser 'centroid' ou 'bottom_center'.")
        for name, (top, bottom) in (
            ("helmet_region", self.helmet_region),
            ("vest_region", self.vest_region),
        ):
            if not 0.0 <= top < bottom <= 1.0:
                raise ConfigError(f"spatial.{name} deve ter 0 <= início < fim <= 1.")


@dataclass(frozen=True)
class AlertConfig:
    """Parâmetros da máquina de estados (Módulo 2)."""

    warning_seconds: float = 5.0       # Estado 1 -> Estado 2
    clear_after_s: float = 1.0         # tempo contínuo em conformidade para voltar ao Estado 0
    lost_track_timeout_s: float = 2.0  # operador sumido por mais que isso é descartado
    violation_repeat_s: float = 0.0    # reemite a violação a cada N s (0 = uma vez por episódio)
    blink_hz: float = 2.0              # frequência do piscar do Estado 1

    def __post_init__(self) -> None:
        for name in ("warning_seconds", "clear_after_s", "lost_track_timeout_s",
                     "violation_repeat_s"):
            if getattr(self, name) < 0:
                raise ConfigError(f"alerts.{name} não pode ser negativo.")
        if self.blink_hz <= 0:
            raise ConfigError("alerts.blink_hz deve ser positivo.")


@dataclass(frozen=True)
class SoundConfig:
    """Alarme sonoro local. Tons no formato (frequência Hz, duração ms, pausa ms)."""

    enabled: bool = True
    backend: str = "auto"  # auto | winsound | pygame | bell | none
    warning_tone: tuple[int, int, int] = (880, 200, 300)
    violation_tone: tuple[int, int, int] = (1320, 400, 100)

    def __post_init__(self) -> None:
        if self.backend not in ("auto", "winsound", "pygame", "bell", "none"):
            raise ConfigError("sound.backend deve ser auto, winsound, pygame, bell ou none.")
        for name in ("warning_tone", "violation_tone"):
            tone = getattr(self, name)
            if len(tone) != 3 or not 37 <= tone[0] <= 32767 or tone[1] <= 0 or tone[2] < 0:
                raise ConfigError(f"sound.{name} deve ser [frequência 37-32767, duração>0, pausa>=0].")


@dataclass(frozen=True)
class EvidenceConfig:
    """Prints de evidência (Módulo 3)."""

    directory: str = "evidences"
    jpeg_quality: int = 95  # 0-100; 95 mantém detalhes com tamanho razoável

    def __post_init__(self) -> None:
        if not 50 <= self.jpeg_quality <= 100:
            raise ConfigError("evidence.jpeg_quality deve estar entre 50 e 100.")


@dataclass(frozen=True)
class AuditConfig:
    """Trilha de auditoria em SQLite (Módulo 3)."""

    database: str = "data/safework_audit.db"


@dataclass(frozen=True)
class EmailConfig:
    """Servidor SMTP e destinatários das notificações (Módulo 3)."""

    enabled: bool = False
    host: str = "smtp.gmail.com"
    port: int = 587
    security: str = "starttls"   # starttls (587) | ssl (465) | none (servidor local de teste)
    username: str = ""
    password: str = ""
    password_env: str = "SAFEWORK_SMTP_PASSWORD"  # se definida, tem prioridade sobre password
    sender: str = ""
    recipients: tuple[str, ...] = ()
    timeout_s: float = 15.0
    max_retries: int = 3
    retry_delay_s: float = 5.0
    queue_size: int = 100
    shutdown_timeout_s: float = 10.0  # tempo para enviar pendências ao encerrar

    def __post_init__(self) -> None:
        if self.security not in ("starttls", "ssl", "none"):
            raise ConfigError("email.security deve ser starttls, ssl ou none.")
        if not 1 <= self.port <= 65535:
            raise ConfigError("email.port inválida.")
        if self.max_retries < 1 or self.queue_size < 1:
            raise ConfigError("email.max_retries e email.queue_size devem ser >= 1.")
        if self.enabled:
            if not self.host or not self.sender or not self.recipients:
                raise ConfigError("Com email.enabled, informe host, sender e recipients.")
            invalid = [item for item in (self.sender, *self.recipients) if "@" not in item]
            if invalid:
                raise ConfigError(f"Endereço de e-mail inválido: {', '.join(invalid)}")

    def resolved_password(self) -> str:
        """Senha da variável de ambiente, se existir; senão a do arquivo."""
        from_env = os.environ.get(self.password_env, "") if self.password_env else ""
        return from_env or self.password


@dataclass(frozen=True)
class DisplayConfig:
    window_name: str = "SafeWork - Monitoramento"
    zone_opacity: float = 0.25


@dataclass(frozen=True)
class Settings:
    camera: CameraConfig = field(default_factory=CameraConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    sound: SoundConfig = field(default_factory=SoundConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    zones_file: str = "config/zones.json"

    @classmethod
    def load(cls, path: str | Path = "config/settings.yaml") -> "Settings":
        """Carrega o YAML; chaves omitidas assumem o valor padrão."""
        file_path = resolve_path(path)
        if not file_path.exists():
            raise ConfigError(f"Arquivo de configuração não encontrado: {file_path}")
        with file_path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}

        try:
            spatial = dict(raw.get("spatial") or {})
            for key in ("helmet_region", "vest_region"):
                if key in spatial:
                    spatial[key] = tuple(spatial[key])
            sound = dict(raw.get("sound") or {})
            for key in ("warning_tone", "violation_tone"):
                if key in sound:
                    sound[key] = tuple(int(value) for value in sound[key])
            email = dict(raw.get("email") or {})
            recipients = email.get("recipients", ())
            if isinstance(recipients, str):  # aceita um único endereço como texto
                recipients = [recipients]
            email["recipients"] = tuple(str(item).strip() for item in recipients or ())
            return cls(
                camera=CameraConfig(**(raw.get("camera") or {})),
                preprocessing=PreprocessingConfig(**(raw.get("preprocessing") or {})),
                detector=DetectorConfig(**(raw.get("detector") or {})),
                spatial=SpatialConfig(**spatial),
                display=DisplayConfig(**(raw.get("display") or {})),
                alerts=AlertConfig(**(raw.get("alerts") or {})),
                sound=SoundConfig(**sound),
                evidence=EvidenceConfig(**(raw.get("evidence") or {})),
                audit=AuditConfig(**(raw.get("audit") or {})),
                email=EmailConfig(**email),
                zones_file=raw.get("zones_file", "config/zones.json"),
            )
        except TypeError as exc:  # chave desconhecida no YAML
            raise ConfigError(f"Chave inválida em {file_path.name}: {exc}") from exc
