"""Testes do Módulo 3: evidência, auditoria SQLite e e-mail assíncrono."""

from __future__ import annotations

import email
import smtplib
import sqlite3
import threading
import time
from dataclasses import replace
from datetime import datetime
from email.header import decode_header, make_header

import cv2
import numpy as np
import pytest

from safework.app import SafeWorkApp
from safework.audit import (
    AuditRepository,
    EmailStatus,
    EvidenceStore,
    ViolationAuditor,
    ViolationRecord,
)
from safework.core.config import (
    AuditConfig,
    CameraConfig,
    ConfigError,
    EmailConfig,
    EvidenceConfig,
    Settings,
)
from safework.core.models import PPE
from safework.logic import AlertEvent, AlertState, EventType
from safework.notification import EmailNotifier, NullNotifier, build_message

CAMERA = CameraConfig(source="rtsp://admin:s3nh4@10.0.0.5/live", id="CAM-01",
                      name="Câmera Prensa 01", area="Setor de Prensas")
WHEN = datetime(2026, 11, 8, 22, 0, 0)


def event(track_id: int = 1, when: datetime = WHEN) -> AlertEvent:
    return AlertEvent(
        type=EventType.VIOLATION_CONFIRMED, track_id=track_id,
        previous_state=AlertState.WARNING, state=AlertState.VIOLATION,
        zone_id="z1", zone_name="Zona Crítica 1", missing_ppe=frozenset({PPE.HELMET}),
        occurred_at=when, timestamp=5.0, frame_index=50,
        warning_started_at=datetime(2026, 11, 8, 21, 59, 55), elapsed_s=5.0,
    )


def image() -> np.ndarray:
    frame = np.zeros((540, 960, 3), np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 400), (0, 0, 255), 4)
    return frame


def email_config(**overrides) -> EmailConfig:
    base = dict(enabled=True, host="smtp.teste", port=2525, security="none",
                sender="safework@teste.com", recipients=("tecnico@teste.com",),
                retry_delay_s=0.01, shutdown_timeout_s=3.0, password_env="")
    return EmailConfig(**{**base, **overrides})


@pytest.fixture
def repository(tmp_path):
    return AuditRepository(tmp_path / "data" / "audit.db")


@pytest.fixture
def store(tmp_path):
    return EvidenceStore(EvidenceConfig(directory=str(tmp_path / "evidences")), "CAM-01")


def record_for(repo: AuditRepository, track_id: int = 1, evidence=None) -> ViolationRecord:
    path, digest = evidence or (None, None)
    return repo.append(ViolationRecord.from_event(
        event(track_id), CAMERA, str(path) if path else None, digest, EmailStatus.PENDING))


# ------------------------------------------------------------------ evidência
def test_evidence_name_matches_required_pattern(store):
    path, digest = store.save(image(), 1, WHEN)
    assert path.name == "evidencia_CAM01_OP1_20261108_220000.jpg"
    assert cv2.imread(str(path)).shape == (540, 960, 3)
    import hashlib
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_evidence_never_overwrites(store):
    first, _ = store.save(image(), 1, WHEN)
    second, _ = store.save(image(), 1, WHEN)
    assert first.exists() and second.name == "evidencia_CAM01_OP1_20261108_220000_2.jpg"


def test_evidence_keeps_high_quality(store):
    """Cena com gradiente, bordas e texto, parecida com um frame anotado."""
    x = np.linspace(0, 255, 960, dtype=np.uint8)
    frame = np.dstack([np.tile(x, (540, 1)), np.tile(x[::-1], (540, 1)),
                       np.full((540, 960), 90, np.uint8)])
    cv2.rectangle(frame, (100, 100), (300, 400), (0, 0, 255), 3)
    cv2.putText(frame, "ALERTA 3.2s", (110, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    path, _ = store.save(frame, 1, WHEN)
    assert cv2.PSNR(frame, cv2.imread(str(path))) > 45


# ------------------------------------------------------------------ registro
def test_record_metadata_and_masked_camera_address():
    record = ViolationRecord.from_event(event(), CAMERA, None, None, EmailStatus.PENDING)
    assert record.occurred_at == "2026-11-08 22:00:00"
    assert record.warning_started_at == "2026-11-08 21:59:55"
    assert record.camera_source == "rtsp://***@10.0.0.5/live"
    assert record.infraction_type == "Ausência de Capacete em Zona Crítica 1"
    assert record.missing_ppe == "capacete"


def test_repository_chains_records(repository):
    first = record_for(repository, 1)
    second = record_for(repository, 2)
    assert first.previous_hash == "0" * 64
    assert second.previous_hash == first.record_hash
    assert repository.get(second.id) == second
    assert repository.verify() == []


def test_email_status_update_keeps_chain_valid(repository):
    record = record_for(repository)
    repository.update_email_status(record.id, EmailStatus.SENT)
    stored = repository.get(record.id)
    assert stored.email_status == "enviado" and stored.email_sent_at
    assert repository.verify() == []


def test_verify_detects_edited_record(repository):
    record_for(repository)
    with sqlite3.connect(repository.path) as connection:
        connection.execute("UPDATE violations SET infraction_type = 'Nada' WHERE id = 1")
    assert any("alterado" in item for item in repository.verify())


def test_verify_detects_deleted_record(repository):
    for track_id in (1, 2, 3):
        record_for(repository, track_id)
    with sqlite3.connect(repository.path) as connection:
        connection.execute("DELETE FROM violations WHERE id = 2")
    assert any("Registro 3: elo quebrado" in item for item in repository.verify())


def test_verify_detects_modified_or_missing_image(repository, store):
    path, digest = store.save(image(), 1, WHEN)
    record_for(repository, 1, (path, digest))
    path.write_bytes(path.read_bytes() + b"x")
    assert any("modificada" in item for item in repository.verify())
    path.unlink()
    assert any("ausente" in item for item in repository.verify())


def test_repository_is_thread_safe(repository):
    threads = [threading.Thread(target=record_for, args=(repository, n)) for n in range(20)]
    for item in threads:
        item.start()
    for item in threads:
        item.join()
    assert repository.count() == 20 and repository.verify() == []


# ------------------------------------------------------------------ auditor
class RecordingNotifier:
    initial_status = EmailStatus.PENDING

    def __init__(self):
        self.jobs = []

    def submit(self, record, attachment):
        self.jobs.append((record, attachment))


def test_auditor_saves_evidence_records_and_notifies(repository, store):
    notifier = RecordingNotifier()
    record = ViolationAuditor(store, repository, notifier, CAMERA).handle(event(), image())
    assert record.evidence_sha256 and record.evidence_path.endswith(
        "evidencia_CAM01_OP1_20261108_220000.jpg")
    assert repository.count() == 1
    assert notifier.jobs[0][1].exists()


def test_auditor_records_even_when_disk_fails(repository, store, monkeypatch):
    monkeypatch.setattr(store, "save", lambda *_: (_ for _ in ()).throw(OSError("disco cheio")))
    record = ViolationAuditor(store, repository, NullNotifier(), CAMERA).handle(event(), image())
    assert record.evidence_path is None and record.email_status == "desativado"
    assert repository.count() == 1


# ------------------------------------------------------------------ e-mail
class FakeSmtp:
    """Servidor simulado. `script` define o resultado de cada conexão."""

    def __init__(self, script=None, delay=0.0):
        self.script = list(script or [])
        self.delay = delay
        self.messages = []
        self.logins = []

    def __call__(self, config):
        outcome = self.script.pop(0) if self.script else None
        if isinstance(outcome, Exception):
            raise outcome
        return _Connection(self)


class _Connection:
    def __init__(self, server):
        self.server = server

    def login(self, user, password):
        self.server.logins.append((user, password))

    def send_message(self, message, sender, recipients):
        """Serializa como o smtplib real: política com CRLF."""
        from email.generator import BytesGenerator
        from io import BytesIO

        time.sleep(self.server.delay)
        buffer = BytesIO()
        BytesGenerator(buffer, policy=message.policy.clone(linesep="\r\n")).flatten(message)
        data = buffer.getvalue()
        assert max(len(line) for line in data.split(b"\r\n")) <= 998  # limite do RFC 5321
        self.server.messages.append((sender, recipients, data))

    def quit(self):
        pass


def wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_email_sent_with_attachment_and_status_recorded(repository, store):
    path, digest = store.save(image(), 1, WHEN)
    record = record_for(repository, 1, (path, digest))
    server = FakeSmtp()
    with EmailNotifier(email_config(), repository.update_email_status, server) as notifier:
        notifier.submit(record, path)
        assert wait_for(lambda: repository.get(record.id).email_status == "enviado")
    message = email.message_from_bytes(server.messages[0][2])
    subject = str(make_header(decode_header(message["Subject"])))
    assert subject == ("[SafeWork] Infração confirmada: Ausência de Capacete em Zona Crítica 1 "
                       "- CAM-01")
    attachments = [part for part in message.walk() if part.get_filename()]
    assert attachments[0].get_filename() == path.name
    assert attachments[0].get_payload(decode=True) == path.read_bytes()


def test_submit_never_blocks_the_video_loop(repository):
    server = FakeSmtp(delay=1.0)  # servidor muito lento
    with EmailNotifier(email_config(), None, server) as notifier:
        started = time.perf_counter()
        for track_id in range(5):
            notifier.submit(record_for(repository, track_id), None)
        assert time.perf_counter() - started < 0.05
        notifier._stopping.set()  # não espera os 5 s de envio no teste


def test_temporary_failure_is_retried(repository):
    record = record_for(repository)
    server = FakeSmtp([OSError("rede fora"), smtplib.SMTPServerDisconnected("caiu"), None])
    with EmailNotifier(email_config(max_retries=3), repository.update_email_status,
                       server) as notifier:
        notifier.submit(record, None)
        assert wait_for(lambda: repository.get(record.id).email_status == "enviado")
    assert len(server.messages) == 1


def test_authentication_error_is_not_retried(repository):
    record = record_for(repository)
    server = FakeSmtp([smtplib.SMTPAuthenticationError(535, b"senha incorreta"), None])
    with EmailNotifier(email_config(), repository.update_email_status, server) as notifier:
        notifier.submit(record, None)
        assert wait_for(lambda: repository.get(record.id).email_status == "falhou")
    assert server.messages == [] and "535" in repository.get(record.id).email_error


def test_gives_up_after_max_retries(repository):
    record = record_for(repository)
    server = FakeSmtp([OSError("x")] * 5)
    with EmailNotifier(email_config(max_retries=2), repository.update_email_status, server) as n:
        n.submit(record, None)
        assert wait_for(lambda: repository.get(record.id).email_status == "falhou")
    assert len(server.script) == 3  # só 2 tentativas consumidas


def test_full_queue_marks_record_as_dropped(repository):
    gate = threading.Event()

    class Blocking(FakeSmtp):
        def __call__(self, config):
            gate.wait(2)
            return super().__call__(config)

    notifier = EmailNotifier(email_config(queue_size=1), repository.update_email_status,
                             Blocking()).start()
    records = [record_for(repository, n) for n in range(3)]
    notifier.submit(records[0], None)
    wait_for(lambda: notifier.pending == 0)  # o primeiro já está em envio
    notifier.submit(records[1], None)
    notifier.submit(records[2], None)  # fila cheia
    assert repository.get(records[2].id).email_status == "descartado"
    gate.set()
    notifier.stop()


def test_stop_flushes_pending_emails(repository):
    server = FakeSmtp(delay=0.05)
    notifier = EmailNotifier(email_config(), repository.update_email_status, server).start()
    for track_id in range(4):
        notifier.submit(record_for(repository, track_id), None)
    notifier.stop()
    assert len(server.messages) == 4


def test_password_from_environment_has_priority(monkeypatch, repository):
    monkeypatch.setenv("SENHA_TESTE", "da-variavel")
    config = email_config(username="u@teste.com", password="do-arquivo",
                          password_env="SENHA_TESTE")
    assert config.resolved_password() == "da-variavel"
    server = FakeSmtp()
    EmailNotifier(config, None, server).send_test()
    assert server.logins == [("u@teste.com", "da-variavel")]


def test_message_escapes_html():
    record = replace(ViolationRecord.from_event(event(), CAMERA, None, None, "pendente"),
                     zone_name="<script>")
    raw = build_message(record, email_config(), None).as_string()
    assert "<script>" not in raw


# ------------------------------------------------------------------ configuração
def test_email_config_validation():
    with pytest.raises(ConfigError):
        EmailConfig(enabled=True, sender="", recipients=("a@b.com",))
    with pytest.raises(ConfigError):
        EmailConfig(enabled=True, sender="a@b.com", recipients=("sem-arroba",))
    with pytest.raises(ConfigError):
        EmailConfig(security="tls")
    EmailConfig(enabled=False)  # desativado não exige dados


def test_single_recipient_string_is_accepted(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("email:\n  recipients: tecnico@empresa.com\n", encoding="utf-8")
    assert Settings.load(path).email.recipients == ("tecnico@empresa.com",)


def test_project_settings_have_smtp_section():
    settings = Settings.load()
    assert settings.email.enabled is False
    assert settings.email.port == 587 and settings.email.security == "starttls"


# ------------------------------------------------------------------ integração
def test_app_records_confirmed_violation_only(tmp_path):
    settings = replace(
        Settings.load(),
        camera=CAMERA,
        evidence=EvidenceConfig(directory=str(tmp_path / "evidences")),
        audit=AuditConfig(database=str(tmp_path / "audit.db")),
    )
    app = SafeWorkApp(settings)
    app.events.publish(replace(event(), type=EventType.WARNING_STARTED), image())
    assert app.audit.count() == 0
    app.events.publish(event(), image())
    assert app.audit.count() == 1
    assert (tmp_path / "evidences" / "evidencia_CAM01_OP1_20261108_220000.jpg").exists()
    assert app.verify_audit() == 0
