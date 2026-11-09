"""Serviço de monitoramento: o laço Módulo 1 -> 2 -> 3, independente da interface.

O mesmo serviço atende três modos:
    - GUI (Módulo 4): roda numa thread; a interface lê `latest` e `stats` periodicamente.
    - Janela OpenCV / headless: roda no laço principal com callbacks por frame.
    - Testes: com um detector simulado injetado por `detector_factory`.

Separar o laço da interface mantém a GUI fluida: se a inferência demorar,
a janela continua respondendo e mostra sempre o frame mais recente.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from safework.audit import AuditRepository
from safework.core.config import Settings
from safework.core.models import FrameAnalysis, ObjectClass, PersonStatus
from safework.logic import AlertEngine, AlertEvent, AlertSnapshot, AlertState, EventBus, EventType
from safework.notification import AlarmPlayer, AlertOverlay, SilentBackend, create_backend
from safework.notification.sound import SoundBackend
from safework.utils.fps import FpsMeter
from safework.vision.capture import VideoSource
from safework.vision.detector import Detector, YoloDetector
from safework.vision.pipeline import VisionPipeline
from safework.vision.preprocessing import FramePreprocessor
from safework.vision.renderer import FrameRenderer
from safework.vision.roi import ZoneRepository
from safework.vision.spatial import SpatialAnalyzer

logger = logging.getLogger(__name__)

DetectorFactory = Callable[..., Detector]


class SourceStatus(str, Enum):
    STARTING = "iniciando"
    LOADING_MODEL = "carregando_modelo"
    CONNECTING = "conectando"
    LIVE = "ao_vivo"
    NO_SIGNAL = "sem_sinal"
    FINISHED = "finalizado"
    STOPPED = "parado"
    ERROR = "erro"


@dataclass(frozen=True, slots=True)
class MonitorStats:
    """Métricas exibidas no painel. Imutável: a GUI lê uma cópia consistente."""

    status: SourceStatus = SourceStatus.STARTING
    frames_processed: int = 0
    fps: float = 0.0
    inference_ms: float = 0.0
    avg_inference_ms: float = 0.0
    persons_detected: int = 0
    persons_in_zone: int = 0
    compliant: int = 0
    non_compliant: int = 0
    unverified: int = 0
    warnings: int = 0              # operadores no Estado 1
    violations_active: int = 0     # operadores no Estado 2
    min_remaining_s: float | None = None
    highest_state: AlertState = AlertState.OK
    session_violations: int = 0    # infrações confirmadas desde que o programa abriu
    total_violations: int = 0      # total na trilha de auditoria
    zones: int = 0
    muted: bool = False
    ppe_verifiable: bool = True
    email_enabled: bool = False
    emails_sent: int = 0
    emails_failed: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FrameUpdate:
    """Frame pronto para exibir, com as métricas daquele instante."""

    sequence: int
    image: np.ndarray   # frame final: ROI + caixas + alertas
    raw: np.ndarray     # frame pré-processado sem desenhos (fundo do editor de zonas)
    stats: MonitorStats


class MonitoringService:
    """Executa o pipeline completo e publica o estado para a interface."""

    def __init__(
        self,
        settings: Settings,
        events: EventBus,
        audit: AuditRepository,
        notifier: object,
        zones: ZoneRepository,
        detector_factory: DetectorFactory = YoloDetector,
        mute: bool = False,
    ) -> None:
        self._settings = settings
        self._events = events
        self._audit = audit
        self._notifier = notifier
        self._zone_repository = zones
        self._detector_factory = detector_factory

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: FrameUpdate | None = None
        self._stats = MonitorStats(
            muted=mute,
            email_enabled=getattr(notifier, "initial_status", "") == "pendente",
        )
        self._recent: deque[AlertEvent] = deque(maxlen=50)
        self._muted = mute
        self._alarm: AlarmPlayer | None = None
        self._pipeline: VisionPipeline | None = None
        self._finished = threading.Event()
        self.exit_code = 0

    # ------------------------------------------------------------------ consulta (thread-safe)
    @property
    def latest(self) -> FrameUpdate | None:
        with self._lock:
            return self._latest

    @property
    def stats(self) -> MonitorStats:
        with self._lock:
            return self._stats

    def recent_events(self) -> list[AlertEvent]:
        """Eventos mais recentes primeiro."""
        with self._lock:
            return list(reversed(self._recent))

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    @property
    def muted(self) -> bool:
        return self._muted

    # ------------------------------------------------------------------ comandos
    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if self._alarm is not None:
            self._alarm.set_muted(muted)
        self._update_stats(muted=muted)
        logger.warning("Alarme sonoro %s pelo operador.", "SILENCIADO" if muted else "reativado")

    def reload_zones(self) -> int:
        """Relê o zones.json e aplica as zonas sem reiniciar a câmera."""
        zones = self._zone_repository.load()
        if self._pipeline is not None:
            self._pipeline.update_zones(zones)
        self._update_stats(zones=len(zones))
        logger.info("Zonas de risco recarregadas: %d.", len(zones))
        return len(zones)

    def start(self, max_frames: int = 0) -> "MonitoringService":
        """Executa `run()` numa thread (modo GUI)."""
        if self._thread is None:
            self._thread = threading.Thread(
                target=self.run, kwargs={"max_frames": max_frames},
                name="monitoring", daemon=True,
            )
            self._thread.start()
        return self

    def stop(self, timeout: float | None = None) -> None:
        """Pede o encerramento e aguarda (inclui o envio de e-mails pendentes)."""
        self._stop.set()
        if self._thread is None:
            self._finished.set()  # nunca foi iniciado: nada a aguardar
        elif self._thread is not threading.current_thread():
            self._thread.join(timeout)

    # ------------------------------------------------------------------ laço principal
    def run(
        self,
        on_frame: Callable[[FrameUpdate], bool | None] | None = None,
        on_idle: Callable[[], bool | None] | None = None,
        max_frames: int = 0,
    ) -> int:
        """Laço bloqueante. Um callback que devolve False encerra o monitoramento."""
        try:
            self.exit_code = self._run(on_frame, on_idle, max_frames)
        except Exception as exc:  # erro inesperado não pode deixar a GUI esperando
            logger.exception("Falha no monitoramento.")
            self._update_stats(status=SourceStatus.ERROR, error=str(exc))
            self.exit_code = 1
        finally:
            self._finished.set()
        return self.exit_code

    def _run(
        self,
        on_frame: Callable[[FrameUpdate], bool | None] | None,
        on_idle: Callable[[], bool | None] | None,
        max_frames: int,
    ) -> int:
        settings = self._settings
        self._update_stats(status=SourceStatus.LOADING_MODEL)
        try:
            detector = self._detector_factory(settings.detector)
        except Exception as exc:
            logger.exception("Não foi possível carregar o modelo.")
            self._update_stats(status=SourceStatus.ERROR, error=f"Modelo: {exc}")
            return 2

        zones = self._zone_repository.load()
        if not zones:
            logger.warning("Nenhuma zona de risco configurada. Use 'Editar zona'.")
        self._pipeline = VisionPipeline(
            preprocessor=FramePreprocessor(settings.preprocessing),
            detector=detector,
            analyzer=SpatialAnalyzer(settings.spatial, zones),
            renderer=FrameRenderer(settings.camera, settings.display, zones),
        )
        engine = AlertEngine(settings.alerts)
        overlay = AlertOverlay(settings.alerts)
        alarm = AlarmPlayer(self._sound_backend(), settings.sound)
        alarm.set_muted(self._muted)
        self._alarm = alarm

        supported = detector.supported_classes
        self._update_stats(
            status=SourceStatus.CONNECTING,
            zones=len(zones),
            total_violations=self._audit.count(),
            ppe_verifiable={ObjectClass.HELMET, ObjectClass.VEST} <= supported,
        )

        fps = FpsMeter()
        processed = 0
        total_ms = 0.0
        last_frame_at = time.monotonic()
        source = VideoSource(settings.camera)
        try:
            with source, alarm, self._notifier:
                logger.info("Alarme sonoro: %s.", alarm.backend_name)
                for result in self._pipeline.stream(source):
                    if self._stop.is_set():
                        break
                    if result is None:
                        idle = time.monotonic() - last_frame_at
                        if idle >= 1.0 and processed:
                            self._update_stats(status=SourceStatus.NO_SIGNAL)
                        if idle >= settings.alerts.lost_track_timeout_s and engine.active_tracks:
                            self._publish(engine.expire_all(), None)
                            alarm.set_level(AlertState.OK)
                        if on_idle is not None and on_idle() is False:
                            break
                        continue

                    last_frame_at = time.monotonic()
                    processed += 1
                    total_ms += result.inference_ms
                    snapshot = engine.update(result)
                    image = overlay.draw(result.annotated_image, result.persons, snapshot)
                    alarm.set_level(snapshot.highest_state)
                    self._publish(snapshot, image)

                    stats = self._frame_stats(result, snapshot, fps.tick(), processed, total_ms)
                    update = FrameUpdate(processed, image, result.frame.image, stats)
                    with self._lock:
                        self._stats = stats
                        self._latest = update
                    if on_frame is not None and on_frame(update) is False:
                        break
                    if max_frames and processed >= max_frames:
                        break
        except KeyboardInterrupt:
            logger.info("Interrompido pelo usuário.")
        finally:
            self._alarm = None

        if processed:
            logger.info("%d frame(s) processado(s). Inferência média: %.1f ms.",
                        processed, total_ms / processed)
        if source.error:
            self._update_stats(status=SourceStatus.ERROR, error=source.error)
            return 1
        final = SourceStatus.STOPPED if self._stop.is_set() else SourceStatus.FINISHED
        self._update_stats(status=final)
        return 0

    # ------------------------------------------------------------------ auxiliares
    def _publish(self, snapshot: AlertSnapshot, image: np.ndarray | None) -> None:
        for event in snapshot.events:
            self._events.publish(event, image)
            with self._lock:
                self._recent.append(event)

    def _frame_stats(
        self,
        result: FrameAnalysis,
        snapshot: AlertSnapshot,
        fps: float,
        processed: int,
        total_ms: float,
    ) -> MonitorStats:
        persons = result.persons
        operators = snapshot.operators.values()
        warnings = [item for item in operators if item.state is AlertState.WARNING]
        confirmed = sum(
            event.type is EventType.VIOLATION_CONFIRMED for event in snapshot.events
        )
        with self._lock:
            previous = self._stats
        return replace(
            previous,
            status=SourceStatus.LIVE,
            frames_processed=processed,
            fps=fps,
            inference_ms=result.inference_ms,
            avg_inference_ms=total_ms / processed,
            persons_detected=len(persons),
            persons_in_zone=sum(person.zone_id is not None for person in persons),
            compliant=sum(person.status is PersonStatus.COMPLIANT for person in persons),
            non_compliant=sum(person.status is PersonStatus.NON_COMPLIANT for person in persons),
            unverified=sum(person.status is PersonStatus.UNVERIFIED for person in persons),
            warnings=len(warnings),
            violations_active=sum(item.state is AlertState.VIOLATION for item in operators),
            min_remaining_s=min((item.remaining_s for item in warnings), default=None),
            highest_state=snapshot.highest_state,
            session_violations=previous.session_violations + confirmed,
            total_violations=previous.total_violations + confirmed,
            muted=self._muted,
            emails_sent=getattr(self._notifier, "sent", 0),
            emails_failed=getattr(self._notifier, "failed", 0),
            error=None,
        )

    def _update_stats(self, **changes: object) -> None:
        with self._lock:
            self._stats = replace(self._stats, **changes)

    def _sound_backend(self) -> SoundBackend:
        if not self._settings.sound.enabled:
            return SilentBackend()
        return create_backend(self._settings.sound.backend)
