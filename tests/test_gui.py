"""Testes da janela (Módulo 4). São pulados se não houver Tkinter ou tela disponível."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")
ctk = pytest.importorskip("customtkinter")

try:
    _probe = tk.Tk()
    _probe.destroy()
except tk.TclError:  # pragma: no cover - depende do ambiente
    pytest.skip("sem display para abrir janelas", allow_module_level=True)

from safework.core.config import CameraConfig  # noqa: E402
from safework.core.models import PPE  # noqa: E402
from safework.gui.dashboard import Dashboard  # noqa: E402
from safework.gui.zone_editor import ZoneEditorDialog  # noqa: E402
from safework.monitoring import FrameUpdate, MonitorStats, SourceStatus  # noqa: E402
from safework.vision.roi import RiskZone, ZoneRepository  # noqa: E402

ZONE = RiskZone("zona-1", "Zona Crítica 1", ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))


class FakeService:
    """Dublê do MonitoringService: o teste controla frames e métricas."""

    def __init__(self):
        self.latest = None
        self.stats = MonitorStats()
        self.events = []
        self.muted = False
        self.started = False
        self.stopped = False
        self.finished = False
        self.reloads = 0

    def recent_events(self):
        return list(self.events)

    def set_muted(self, muted):
        self.muted = muted

    def reload_zones(self):
        self.reloads += 1
        return 2

    def start(self):
        self.started = True
        return self

    def stop(self, timeout=None):
        self.stopped = True
        self.finished = True

    def push(self, sequence, **stats):
        image = np.zeros((540, 960, 3), np.uint8)
        image[:, :, 2] = 200
        self.stats = replace(MonitorStats(status=SourceStatus.LIVE), **stats)
        self.latest = FrameUpdate(sequence, image, np.zeros_like(image), self.stats)


def pump(window, seconds=0.2):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        window.update()
        time.sleep(0.01)


@pytest.fixture
def dashboard(tmp_path):
    zones = ZoneRepository(tmp_path / "zones.json")
    zones.save([ZONE])
    service = FakeService()
    window = Dashboard(service, zones, audit_check=lambda: (3, []),
                       camera=CameraConfig(id="CAM-07", area="Solda"), poll_ms=20)
    window.messages = []
    window.show_message = lambda title, message, error=False: window.messages.append(
        (title, error))
    window.geometry("1280x760")
    window.start()
    pump(window)
    yield window, service, zones
    try:
        window.destroy()
    except tk.TclError:
        pass


def test_starts_service_and_shows_waiting_state(dashboard):
    window, service, _ = dashboard
    assert service.started
    assert window.metrics.badge_title.cget("text") == "INICIANDO"
    assert window.video.frames_shown == 0


def test_renders_frame_and_metrics(dashboard):
    window, service, _ = dashboard
    service.push(1, persons_in_zone=2, persons_detected=3, warnings=1, min_remaining_s=3.2,
                 session_violations=4, total_violations=9, fps=14.5, inference_ms=61)
    pump(window)
    assert window.video.frames_shown == 1
    width, height = window.video.displayed_size
    assert abs(width / height - 960 / 540) < 0.02  # proporção preservada
    cards = {key: card.value.cget("text") for key, card in window.metrics.cards.items()}
    assert cards == {"in_zone": "2", "warnings": "1", "session": "4", "total": "9",
                     "fps": "14.5", "inference": "61"}
    assert window.metrics.badge_title.cget("text") == "ALERTA"
    assert window.status_label.cget("text") == "● AO VIVO"


def test_same_frame_is_not_redrawn(dashboard):
    window, service, _ = dashboard
    service.push(1)
    pump(window, 0.3)
    assert window.video.frames_shown == 1
    service.push(2)
    pump(window)
    assert window.video.frames_shown == 2


def test_event_history(dashboard):
    from datetime import datetime

    from safework.logic import AlertEvent, AlertState, EventType

    window, service, _ = dashboard
    service.events = [AlertEvent(EventType.VIOLATION_CONFIRMED, 1, AlertState.WARNING,
                                 AlertState.VIOLATION, "z", "Z", frozenset({PPE.HELMET}),
                                 datetime(2026, 11, 8, 23, 0, 5), 5.0, 50, None, 5.0)]
    pump(window)
    assert window.metrics.event_lines() == ["23:00:05  #1  INFRAÇÃO CONFIRMADA (Capacete)"]


def test_mute_button_and_shortcut(dashboard):
    window, service, _ = dashboard
    window.mute_button.invoke()
    assert service.muted and "Reativar" in window.mute_button.cget("text")
    window.focus_force()
    window.event_generate("<KeyPress-m>")
    pump(window)
    assert not service.muted and "Silenciar" in window.mute_button.cget("text")


def test_verify_audit_runs_in_background(dashboard):
    window, service, _ = dashboard
    window.verify_audit()
    pump(window, 0.3)
    assert window.messages == [("Auditoria íntegra", False)]
    assert window.audit_button.cget("state") == "normal"


def test_verify_audit_reports_problems(tmp_path):
    zones = ZoneRepository(tmp_path / "zones.json")
    window = Dashboard(FakeService(), zones, lambda: (2, ["Registro 2: conteúdo alterado"]),
                       CameraConfig(), poll_ms=20)
    window.messages = []
    window.show_message = lambda title, message, error=False: window.messages.append(
        (title, error))
    window.start()
    window.verify_audit()
    pump(window, 0.3)
    assert window.messages == [("Auditoria COMPROMETIDA", True)]
    window.destroy()


def test_zone_editor_needs_a_frame(dashboard):
    window, service, _ = dashboard
    window.open_zone_editor()
    assert window.editor is None and "Aguarde" in window.toast.cget("text")


def test_zone_editor_creates_zone_and_reloads(dashboard):
    window, service, zones = dashboard
    service.push(1)
    pump(window)
    window.open_zone_editor()
    editor = window.editor
    pump(window)
    scale = editor._scale
    for x, y in ((96, 54), (480, 54), (480, 270)):  # pixels da imagem original
        editor.click(round(x * scale), round(y * scale))
    editor.name_entry.insert(0, "Prensa 03")
    editor.vest_var.set(False)
    assert editor.save()
    pump(window)
    saved = zones.load()
    assert [zone.name for zone in saved] == ["Zona Crítica 1", "Prensa 03"]
    assert saved[1].required_ppe == {PPE.HELMET}
    assert saved[1].points[0] == pytest.approx((0.1, 0.1), abs=0.01)
    assert service.reloads == 1 and "aplicadas" in window.toast.cget("text")


def test_zone_editor_requires_ppe_and_three_vertices(dashboard):
    window, service, zones = dashboard
    service.push(1)
    pump(window)
    window.open_zone_editor()
    editor = window.editor
    editor.click(10, 10)
    editor.click(100, 10)
    assert editor.close_polygon() is None and "3 vértices" in editor.hint.cget("text")
    editor.click(100, 100)
    editor.helmet_var.set(False)
    editor.vest_var.set(False)
    assert editor.close_polygon() is None and "EPI" in editor.hint.cget("text")
    editor.cancel()
    assert zones.load() == [ZONE] and service.reloads == 0


def test_shortcuts_ignored_while_editor_open(dashboard):
    window, service, _ = dashboard
    service.push(1)
    pump(window)
    window.open_zone_editor()
    window._shortcut(window.toggle_mute)
    assert not service.muted
    window.editor.cancel()
    window._shortcut(window.toggle_mute)
    assert service.muted


def test_exit_stops_service_then_closes(dashboard):
    window, service, _ = dashboard
    window.request_exit()
    assert window.exit_button.cget("state") == "disabled"
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            window.update()
        except tk.TclError:
            break  # janela destruída
        time.sleep(0.01)
    assert service.stopped
    with pytest.raises(tk.TclError):
        window.winfo_exists()


def test_editor_maps_clicks_to_image_pixels(tmp_path):
    root = ctk.CTk()
    zones = ZoneRepository(tmp_path / "zones.json")
    editor = ZoneEditorDialog(root, np.zeros((1080, 1920, 3), np.uint8), zones)
    assert editor._display_size == (1000, 562)
    editor.click(500, 281)
    x, y = editor.editor_state.draft[0]
    assert x == pytest.approx(0.5, abs=0.002) and y == pytest.approx(0.5, abs=0.002)
    root.destroy()
