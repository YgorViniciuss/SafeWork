"""Testes do Módulo 4 que não precisam de tela: serviço de monitoramento e viewmodel."""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime

import cv2
import numpy as np
import pytest

from safework.audit import AuditRepository, EvidenceStore, ViolationAuditor
from safework.core.config import (
    AuditConfig,
    CameraConfig,
    EvidenceConfig,
    Settings,
    SoundConfig,
)
from safework.core.models import PPE, BoundingBox, Detection, ObjectClass
from safework.gui import theme
from safework.gui.viewmodel import (
    audit_message,
    compliance_badge,
    email_text,
    fit_size,
    format_event,
    plural,
    status_pill,
)
from safework.logic import AlertEvent, AlertState, EventBus, EventType
from safework.monitoring import MonitoringService, MonitorStats, SourceStatus
from safework.notification import AlarmPlayer, NullNotifier
from safework.vision.roi import RiskZone, ZoneRepository

ZONE = RiskZone("zona-1", "Zona Crítica 1", ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))


class ScriptedDetector:
    """Operador #1 sempre sem capacete no centro da imagem; #2 com EPI completo."""

    supported_classes = frozenset(ObjectClass)

    def __init__(self, config=None, delay: float = 0.0):
        self.delay = delay

    def detect(self, image):
        time.sleep(self.delay)
        h, w = image.shape[:2]
        return [
            Detection(ObjectClass.PERSON, 0.9, BoundingBox(w * .3, h * .3, w * .45, h * .8), 1),
            Detection(ObjectClass.VEST, 0.8, BoundingBox(w * .32, h * .45, w * .43, h * .6)),
            Detection(ObjectClass.PERSON, 0.9, BoundingBox(w * .55, h * .3, w * .7, h * .8), 2),
            Detection(ObjectClass.HELMET, 0.8, BoundingBox(w * .6, h * .3, w * .65, h * .36)),
            Detection(ObjectClass.VEST, 0.8, BoundingBox(w * .57, h * .45, w * .68, h * .6)),
        ]


def make_video(path, frames=80, fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for index in range(frames):
        image = np.full((240, 320, 3), 60, np.uint8)
        cv2.putText(image, str(index), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(image)
    writer.release()


@pytest.fixture
def env(tmp_path):
    video = tmp_path / "cena.mp4"
    make_video(video)
    zones = ZoneRepository(tmp_path / "zones.json")
    zones.save([ZONE])
    settings = replace(
        Settings(),
        camera=CameraConfig(source=str(video)),
        audit=AuditConfig(database=str(tmp_path / "audit.db")),
        evidence=EvidenceConfig(directory=str(tmp_path / "evidences")),
        sound=SoundConfig(backend="none"),
    )
    audit = AuditRepository(tmp_path / "audit.db")
    return settings, zones, audit


def service_for(env, detector=ScriptedDetector, **kwargs):
    """Barramento montado como na aplicação: o auditor do Módulo 3 grava as infrações."""
    settings, zones, audit = env
    bus = EventBus()
    bus.subscribe(ViolationAuditor(EvidenceStore(settings.evidence, settings.camera.id), audit,
                                   NullNotifier(), settings.camera),
                  [EventType.VIOLATION_CONFIRMED])
    return MonitoringService(settings, bus, audit, NullNotifier(), zones,
                             detector_factory=detector, **kwargs)


# ------------------------------------------------------------------ serviço
def test_service_reports_metrics_and_violation(env):
    service = service_for(env)
    assert service.run() == 0
    stats = service.stats
    assert stats.status is SourceStatus.FINISHED
    assert stats.frames_processed == 80
    assert stats.persons_detected == 2 and stats.persons_in_zone == 2
    assert stats.compliant == 1 and stats.non_compliant == 1
    assert stats.session_violations == 1 and stats.total_violations == 1
    assert stats.violations_active == 1 and stats.highest_state is AlertState.VIOLATION
    assert stats.zones == 1 and stats.ppe_verifiable
    kinds = [event.type for event in service.recent_events()]
    assert kinds == [EventType.VIOLATION_CONFIRMED, EventType.WARNING_STARTED]  # recente primeiro


def test_latest_frame_has_overlay_and_clean_raw(env):
    service = service_for(env)
    service.run(max_frames=10)
    update = service.latest
    assert update.sequence == 10
    assert update.image.shape == update.raw.shape
    assert not np.array_equal(update.image, update.raw)  # overlays só na imagem final


def test_warning_countdown_in_stats(env):
    seen = []
    service = service_for(env)
    service.run(on_frame=lambda update: seen.append(update.stats), max_frames=30)
    warning = seen[20]  # t = 2,0 s
    assert warning.warnings == 1
    assert warning.min_remaining_s == pytest.approx(3.0, abs=0.11)


def test_total_includes_previous_audit_records(env):
    service_for(env).run()
    second = service_for(env)
    second.run()
    assert second.stats.session_violations == 1 and second.stats.total_violations == 2
    assert env[2].count() == 2


def test_on_frame_false_stops(env):
    service = service_for(env)
    service.run(on_frame=lambda update: update.sequence < 5)
    assert service.stats.frames_processed == 5


def test_threaded_start_and_stop(env):
    service = service_for(env, detector=lambda cfg: ScriptedDetector(delay=0.05)).start()
    deadline = time.monotonic() + 5
    while service.latest is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert service.stats.status is SourceStatus.LIVE
    service.stop(timeout=5)
    assert service.finished and service.stats.status is SourceStatus.STOPPED


def test_stop_before_start_marks_finished(env):
    service = service_for(env)
    service.stop()
    assert service.finished


def test_reload_zones_applies_without_restart(env):
    settings, zones, audit = env
    service = service_for(env)
    holder = {}

    def on_frame(update):
        if update.sequence == 5:
            zones.save([])  # remove todas as zonas pelo "editor"
            holder["count"] = service.reload_zones()
        return True

    service.run(on_frame=on_frame)
    assert holder["count"] == 0
    assert service.stats.persons_in_zone == 0 and service.stats.zones == 0
    assert service.stats.session_violations == 0  # sem zona, ninguém é cobrado


def test_mute_reaches_alarm(env):
    service = service_for(env, detector=lambda cfg: ScriptedDetector(delay=0.02)).start()
    deadline = time.monotonic() + 5
    while service._alarm is None and time.monotonic() < deadline:
        time.sleep(0.01)
    service.set_muted(True)
    assert service._alarm.muted and service.stats.muted
    service.set_muted(False)
    assert not service._alarm.muted
    service.stop(timeout=5)


def test_start_muted(env):
    service = service_for(env, mute=True)
    service.run(max_frames=2)
    assert service.stats.muted


def test_model_failure_is_reported(env):
    def broken(config):
        raise FileNotFoundError("pesos ausentes")

    service = service_for(env, detector=broken)
    assert service.run() == 2
    assert service.stats.status is SourceStatus.ERROR and "pesos" in service.stats.error
    assert service.finished


def test_missing_video_is_reported(env):
    settings, zones, audit = env
    settings = replace(settings, camera=CameraConfig(source="/nao/existe.mp4"))
    service = MonitoringService(settings, EventBus(), audit, NullNotifier(), zones,
                                detector_factory=ScriptedDetector)
    assert service.run() == 1
    assert service.stats.status is SourceStatus.ERROR


# ------------------------------------------------------------------ alarme
class CountingBackend:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def beep(self, frequency_hz, duration_ms):
        self.calls += 1
        time.sleep(duration_ms / 1000)


def test_muted_alarm_is_silent_and_resumes():
    backend = CountingBackend()
    config = SoundConfig(warning_tone=(880, 10, 10))
    with AlarmPlayer(backend, config) as alarm:
        alarm.set_muted(True)
        alarm.set_level(AlertState.WARNING)
        time.sleep(0.15)
        assert backend.calls == 0
        alarm.set_muted(False)
        time.sleep(0.15)
        assert backend.calls > 0


# ------------------------------------------------------------------ viewmodel
def stats(**changes) -> MonitorStats:
    return replace(MonitorStats(status=SourceStatus.LIVE), **changes)


@pytest.mark.parametrize(
    ("changes", "title", "color"),
    [
        ({"status": SourceStatus.LOADING_MODEL}, "INICIANDO", theme.NEUTRAL),
        ({"status": SourceStatus.NO_SIGNAL}, "SEM SINAL", theme.WARNING),
        ({"status": SourceStatus.ERROR, "error": "x"}, "ERRO", theme.DANGER),
        ({}, "ÁREA LIVRE", theme.NEUTRAL),
        ({"persons_in_zone": 2, "compliant": 2}, "CONFORME", theme.OK),
        ({"persons_in_zone": 1, "unverified": 1}, "NÃO VERIFICADO", theme.WARNING),
        ({"persons_in_zone": 1, "warnings": 1, "min_remaining_s": 2.4}, "ALERTA", theme.WARNING),
        ({"persons_in_zone": 1, "warnings": 1, "violations_active": 1}, "INFRAÇÃO", theme.DANGER),
    ],
)
def test_compliance_badge_priority(changes, title, color):
    badge = compliance_badge(stats(**changes))
    assert (badge.title, badge.color) == (title, color)


def test_badge_texts():
    assert compliance_badge(stats(persons_in_zone=1, warnings=1, min_remaining_s=2.4)).detail \
        == "1 operador sem EPI - confirmação em 2.4 s"
    assert compliance_badge(stats(persons_in_zone=3, compliant=3)).detail \
        == "3 operadores com EPI na zona"
    assert plural(2, "registro") == "2 registros"


def test_status_pill():
    assert status_pill(stats()).text == "● AO VIVO"
    assert status_pill(stats(status=SourceStatus.CONNECTING)).color == theme.INFO


def test_email_text():
    assert email_text(stats())[0] == "Desativado"
    assert email_text(stats(email_enabled=True, emails_sent=3))[0] == "Ativo · 3 enviado(s)"
    assert email_text(stats(email_enabled=True, emails_failed=1))[1] == theme.DANGER


def test_format_event():
    event = AlertEvent(EventType.VIOLATION_CONFIRMED, 4, AlertState.WARNING, AlertState.VIOLATION,
                       "z", "Zona", frozenset({PPE.HELMET, PPE.VEST}),
                       datetime(2026, 11, 8, 23, 1, 2), 5.0, 50, None, 5.0)
    text, color = format_event(event)
    assert text == "23:01:02  #4  INFRAÇÃO CONFIRMADA (Capacete, Colete)"
    assert color == theme.DANGER


@pytest.mark.parametrize(
    ("size", "box", "expected"),
    [((960, 540), (480, 480), (480, 270)), ((540, 960), (1000, 480), (270, 480)),
     ((960, 540), (1, 1), (960, 540)), ((960, 540), (1920, 1080), (1920, 1080))],
)
def test_fit_size(size, box, expected):
    assert fit_size(*size, *box) == expected


def test_audit_message():
    assert audit_message(3, [])[2] is True
    title, message, ok = audit_message(5, [f"p{n}" for n in range(12)])
    assert not ok and "COMPROMETIDA" in title and "mais 2" in message
