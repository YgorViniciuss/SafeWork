"""Testes do Módulo 2: máquina de estados, motor, eventos, overlay e alarme."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pytest

from safework.core.config import AlertConfig, ConfigError, Settings, SoundConfig
from safework.core.models import (
    PPE,
    BoundingBox,
    Detection,
    Frame,
    FrameAnalysis,
    ObjectClass,
    PersonAnalysis,
    PersonStatus,
)
from safework.logic import AlertEngine, AlertState, EventBus, EventType
from safework.notification import AlarmPlayer, AlertOverlay, create_backend

CONFIG = AlertConfig(warning_seconds=5.0, clear_after_s=1.0, lost_track_timeout_s=2.0)
BASE = datetime(2026, 11, 8, 14, 0, 0)
OK, NO, OUT, UNV = (PersonStatus.COMPLIANT, PersonStatus.NON_COMPLIANT,
                    PersonStatus.OUTSIDE_ZONE, PersonStatus.UNVERIFIED)


def person(track_id: int | None, status: PersonStatus) -> PersonAnalysis:
    box = BoundingBox(100 + 150 * (track_id or 0), 100, 200 + 150 * (track_id or 0), 400)
    in_zone = status is not OUT
    return PersonAnalysis(
        detection=Detection(ObjectClass.PERSON, 0.9, box, track_id),
        status=status,
        anchor=box.center,
        zone_id="z1" if in_zone else None,
        zone_name="Zona Crítica 1" if in_zone else None,
        missing_ppe=frozenset({PPE.HELMET}) if status is NO else frozenset(),
    )


def analysis(t: float, *persons: PersonAnalysis, index: int = 0) -> FrameAnalysis:
    image = np.zeros((480, 960, 3), np.uint8)
    frame = Frame(image, index, t, BASE + timedelta(seconds=t))
    return FrameAnalysis(frame, image, tuple(persons), (), 0.0)


def run(engine: AlertEngine, timeline):
    """timeline: lista de (t, [(track_id, status), ...]). Devolve snapshots."""
    return [
        engine.update(analysis(t, *(person(i, s) for i, s in people), index=n))
        for n, (t, people) in enumerate(timeline)
    ]


def at(start: float, stop: float, step: float, people):
    t = start
    while t <= stop + 1e-9:
        yield round(t, 3), people
        t += step


def types(snapshots):
    return [(e.type, e.track_id) for s in snapshots for e in s.events]


# ------------------------------------------------------------------- transições
def test_compliant_operator_never_creates_state():
    engine = AlertEngine(CONFIG)
    snaps = run(engine, at(0, 10, 0.5, [(1, OK)]))
    assert types(snaps) == [] and engine.active_tracks == 0


def test_violation_confirmed_after_exactly_five_seconds():
    engine = AlertEngine(CONFIG)
    snaps = run(engine, at(0, 7, 0.1, [(1, NO)]))
    assert types(snaps) == [(EventType.WARNING_STARTED, 1), (EventType.VIOLATION_CONFIRMED, 1)]
    confirmed = next(e for s in snaps for e in s.events if e.type is EventType.VIOLATION_CONFIRMED)
    assert confirmed.timestamp == pytest.approx(5.0)
    assert confirmed.warning_started_at == BASE
    assert confirmed.occurred_at == BASE + timedelta(seconds=5)
    assert confirmed.description == "Ausência de Capacete em Zona Crítica 1"
    assert snaps[-1].operators[1].state is AlertState.VIOLATION


def test_countdown_values():
    engine = AlertEngine(CONFIG)
    snaps = run(engine, [(0.0, [(1, NO)]), (2.0, [(1, NO)])])
    alert = snaps[-1].operators[1]
    assert alert.state is AlertState.WARNING
    assert alert.elapsed_s == pytest.approx(2.0) and alert.remaining_s == pytest.approx(3.0)


def test_putting_helmet_on_before_timeout_resolves_without_violation():
    engine = AlertEngine(CONFIG)
    timeline = list(at(0, 4.4, 0.2, [(1, NO)])) + list(at(4.6, 8, 0.2, [(1, OK)]))
    snaps = run(engine, timeline)
    assert types(snaps) == [(EventType.WARNING_STARTED, 1), (EventType.RESOLVED, 1)]
    assert engine.active_tracks == 0


def test_single_frame_of_helmet_does_not_reset_timer():
    engine = AlertEngine(CONFIG)
    timeline = [(t, [(1, OK if t == 3.0 else NO)]) for t, _ in at(0, 6, 0.5, None)]
    snaps = run(engine, timeline)
    assert (EventType.VIOLATION_CONFIRMED, 1) in types(snaps)
    confirmed = next(e for s in snaps for e in s.events if e.type is EventType.VIOLATION_CONFIRMED)
    assert confirmed.elapsed_s == pytest.approx(5.0)  # o timer não reiniciou em t=3


def test_no_escalation_while_operator_appears_compliant():
    """Colocou o capacete aos 4,5 s: a confirmação não pode chegar durante a histerese."""
    engine = AlertEngine(CONFIG)
    timeline = list(at(0, 4.5, 0.5, [(1, NO)])) + list(at(5.0, 6.0, 0.25, [(1, OK)]))
    assert EventType.VIOLATION_CONFIRMED not in [kind for kind, _ in types(run(engine, timeline))]


def test_leaving_the_zone_resolves_after_hysteresis():
    engine = AlertEngine(CONFIG)
    snaps = run(engine, list(at(0, 2, 0.5, [(1, NO)])) + list(at(2.5, 4, 0.5, [(1, OUT)])))
    assert types(snaps)[-1] == (EventType.RESOLVED, 1)


def test_unverified_status_never_starts_a_warning():
    engine = AlertEngine(CONFIG)
    assert types(run(engine, at(0, 10, 0.5, [(1, UNV)]))) == []


def test_violation_resolves_and_reports_previous_state():
    engine = AlertEngine(CONFIG)
    snaps = run(engine, list(at(0, 6, 0.5, [(1, NO)])) + list(at(6.5, 8, 0.5, [(1, OK)])))
    resolved = snaps[-1].events or next(
        s.events for s in reversed(snaps) if s.events
    )
    assert resolved[0].type is EventType.RESOLVED
    assert resolved[0].previous_state is AlertState.VIOLATION


def test_violation_reported_once_per_episode_by_default():
    engine = AlertEngine(CONFIG)
    snaps = run(engine, at(0, 30, 0.5, [(1, NO)]))
    assert [k for k, _ in types(snaps)].count(EventType.VIOLATION_CONFIRMED) == 1


def test_optional_reminder_repeats_violation():
    engine = AlertEngine(AlertConfig(violation_repeat_s=10.0))
    kinds = [k for k, _ in types(run(engine, at(0, 26, 0.5, [(1, NO)])))]
    assert kinds.count(EventType.VIOLATION_REMINDER) == 2  # em t=15 e t=25


def test_zero_seconds_confirms_immediately():
    engine = AlertEngine(AlertConfig(warning_seconds=0.0))
    assert [k for k, _ in types(run(engine, [(0.0, [(1, NO)])]))] == [
        EventType.WARNING_STARTED, EventType.VIOLATION_CONFIRMED]


# ---------------------------------------------------------- vários operadores
def test_each_operator_has_an_independent_timer():
    engine = AlertEngine(CONFIG)
    timeline = [(t, [(1, NO)] + ([(2, NO)] if t >= 3 else [])) for t, _ in at(0, 8.5, 0.5, None)]
    snaps = run(engine, timeline)
    confirmed = {e.track_id: e.timestamp for s in snaps for e in s.events
                 if e.type is EventType.VIOLATION_CONFIRMED}
    assert confirmed == {1: pytest.approx(5.0), 2: pytest.approx(8.0)}
    assert snaps[-1].highest_state is AlertState.VIOLATION


def test_short_occlusion_keeps_the_timer():
    engine = AlertEngine(CONFIG)
    timeline = [(t, [] if 2 <= t < 3.5 else [(1, NO)]) for t, _ in at(0, 6, 0.5, None)]
    confirmed = [e for s in run(engine, timeline) for e in s.events
                 if e.type is EventType.VIOLATION_CONFIRMED]
    assert confirmed and confirmed[0].timestamp == pytest.approx(5.0)


def test_track_lost_after_timeout_without_violation():
    engine = AlertEngine(CONFIG)
    timeline = list(at(0, 1, 0.5, [(1, NO)])) + list(at(1.5, 6, 0.5, []))
    assert types(run(engine, timeline)) == [
        (EventType.WARNING_STARTED, 1), (EventType.TRACK_LOST, 1)]
    assert engine.active_tracks == 0


def test_untracked_persons_are_ignored(caplog):
    engine = AlertEngine(CONFIG)
    assert types(run(engine, at(0, 6, 0.5, [(None, NO)]))) == []
    assert "tracking" in caplog.text


def test_expire_all_closes_open_episodes():
    engine = AlertEngine(CONFIG)
    run(engine, [(0.0, [(1, NO), (2, NO)])])
    snapshot = engine.expire_all()
    assert sorted(e.track_id for e in snapshot.events) == [1, 2]
    assert engine.active_tracks == 0 and snapshot.highest_state is AlertState.OK


# -------------------------------------------------------------------- eventos
def test_event_bus_filters_and_isolates_failures():
    received = []

    class Broken:
        def handle(self, event, evidence):
            raise RuntimeError("falha simulada")

    class Recorder:
        def handle(self, event, evidence):
            received.append((event.type, evidence.shape))

    bus = EventBus()
    bus.subscribe(Broken())
    bus.subscribe(Recorder(), [EventType.VIOLATION_CONFIRMED])
    engine = AlertEngine(CONFIG)
    for snapshot in run(engine, at(0, 6, 0.5, [(1, NO)])):
        for event in snapshot.events:
            bus.publish(event, np.zeros((4, 4, 3), np.uint8))
    assert received == [(EventType.VIOLATION_CONFIRMED, (4, 4, 3))]


def test_description_lists_multiple_items():
    engine = AlertEngine(CONFIG)
    p = person(1, NO)
    p = PersonAnalysis(p.detection, NO, p.anchor, "z1", "Zona Crítica 1",
                       missing_ppe=frozenset({PPE.HELMET, PPE.VEST}))
    event = engine.update(analysis(0.0, p)).events[0]
    assert event.description == "Ausência de Capacete e Colete em Zona Crítica 1"


# -------------------------------------------------------------------- overlay
def test_overlay_blinks_and_keeps_original():
    engine = AlertEngine(CONFIG)
    overlay = AlertOverlay(CONFIG)
    images = []
    for t in (0.0, 0.25):  # fases opostas do pisca a 2 Hz
        result = analysis(t, person(1, NO))
        snapshot = engine.update(result)
        images.append(overlay.draw(result.annotated_image, result.persons, snapshot))
        assert result.annotated_image.sum() == 0
    assert not np.array_equal(images[0], images[1])


def test_overlay_without_alerts_is_identity():
    result = analysis(0.0, person(1, OK))
    snapshot = AlertEngine(CONFIG).update(result)
    image = AlertOverlay(CONFIG).draw(result.annotated_image, result.persons, snapshot)
    assert np.array_equal(image, result.annotated_image)


def test_violation_overlay_is_stable_red():
    engine = AlertEngine(CONFIG)
    overlay = AlertOverlay(CONFIG)
    run(engine, at(0, 5.5, 0.5, [(1, NO)]))
    frames = []
    for t in (6.0, 6.25):
        result = analysis(t, person(1, NO))
        frames.append(overlay.draw(result.annotated_image, result.persons, engine.update(result)))
    assert np.array_equal(frames[0], frames[1])
    assert frames[0][476, 480].tolist() == [0, 0, 255]  # borda vermelha


# --------------------------------------------------------------------- alarme
class FakeBackend:
    name = "fake"

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def beep(self, frequency_hz, duration_ms):
        with self.lock:
            self.calls.append(frequency_hz)
        time.sleep(duration_ms / 1000)


def test_alarm_follows_state_without_blocking():
    backend = FakeBackend()
    config = SoundConfig(warning_tone=(880, 20, 20), violation_tone=(1320, 20, 5))
    with AlarmPlayer(backend, config) as alarm:
        time.sleep(0.1)
        assert backend.calls == []  # silêncio no Estado 0
        started = time.perf_counter()
        alarm.set_level(AlertState.WARNING)
        assert time.perf_counter() - started < 0.01  # não bloqueia o laço de vídeo
        time.sleep(0.15)
        alarm.set_level(AlertState.VIOLATION)
        time.sleep(0.15)
        alarm.set_level(AlertState.OK)
        time.sleep(0.1)
        count = len(backend.calls)
        time.sleep(0.15)
    assert 880 in backend.calls and 1320 in backend.calls
    assert len(backend.calls) == count  # parou ao voltar ao Estado 0


def test_alarm_survives_broken_audio_device():
    class Broken:
        name = "broken"

        def beep(self, *_):
            raise OSError("sem dispositivo")

    with AlarmPlayer(Broken(), SoundConfig(warning_tone=(880, 10, 10))) as alarm:
        alarm.set_level(AlertState.WARNING)
        time.sleep(0.1)
        assert alarm.backend_name == "none"


def test_backend_factory_falls_back(monkeypatch):
    assert create_backend("none").name == "none"
    assert create_backend("bell").name == "bell"
    assert create_backend("winsound").name in ("winsound", "bell")


# --------------------------------------------------------------------- config
def test_project_settings_include_module_2():
    settings = Settings.load()
    assert settings.alerts.warning_seconds == 5.0
    assert settings.sound.warning_tone == (880, 200, 300)


@pytest.mark.parametrize("kwargs", [{"warning_seconds": -1}, {"blink_hz": 0}])
def test_invalid_alert_config(kwargs):
    with pytest.raises(ConfigError):
        AlertConfig(**kwargs)


def test_invalid_sound_config():
    with pytest.raises(ConfigError):
        SoundConfig(backend="sirene")
    with pytest.raises(ConfigError):
        SoundConfig(warning_tone=(10, 100, 0))
