"""SMTP Notification Engine: e-mail assíncrono com o print de evidência em anexo.

O laço de vídeo só chama `submit()`, que coloca o pedido numa fila e retorna em
microssegundos. Uma thread dedicada consome a fila e conversa com o servidor
SMTP, com novas tentativas em caso de falha temporária da rede. Assim, um
servidor lento ou fora do ar nunca reduz o FPS da câmera.
"""

from __future__ import annotations

import html
import logging
import queue
import smtplib
import ssl
import threading
from collections.abc import Callable
from dataclasses import dataclass
from email.header import Header
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid
from pathlib import Path
from typing import Any

from safework.audit.records import EmailStatus, ViolationRecord
from safework.core.config import EmailConfig

logger = logging.getLogger(__name__)

ResultCallback = Callable[[int, str, str | None], None]
SmtpFactory = Callable[[EmailConfig], Any]

# Erros que não melhoram com nova tentativa: credencial ou destinatário inválidos.
_PERMANENT_ERRORS = (
    smtplib.SMTPAuthenticationError,
    smtplib.SMTPRecipientsRefused,
    smtplib.SMTPSenderRefused,
    smtplib.SMTPNotSupportedError,
)


def default_smtp_factory(config: EmailConfig) -> smtplib.SMTP:
    """Abre a conexão conforme `security` (SSL direto, STARTTLS ou texto puro)."""
    context = ssl.create_default_context()
    if config.security == "ssl":
        return smtplib.SMTP_SSL(config.host, config.port, timeout=config.timeout_s,
                                context=context)
    connection = smtplib.SMTP(config.host, config.port, timeout=config.timeout_s)
    if config.security == "starttls":
        connection.starttls(context=context)
    return connection


def build_message(
    record: ViolationRecord,
    config: EmailConfig,
    attachment: Path | None,
) -> MIMEMultipart:
    """Monta o e-mail: texto simples, versão HTML e a imagem anexada."""
    message = MIMEMultipart("mixed")
    subject = f"[SafeWork] Infração confirmada: {record.infraction_type} - {record.camera_id}"
    message["Subject"] = Header(subject, "utf-8")
    message["From"] = formataddr((str(Header("SafeWork", "utf-8")), config.sender))
    message["To"] = ", ".join(config.recipients)
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain="safework.local")
    message["X-SafeWork-Event"] = record.event_uid

    rows = [
        ("Data e hora", record.occurred_at),
        ("Início da infração", record.warning_started_at or "-"),
        ("Infração", record.infraction_type),
        ("Câmera", f"{record.camera_id} - {record.camera_name}"),
        ("Origem da câmera", record.camera_source),
        ("Área / setor", record.camera_area),
        ("Zona de risco", record.zone_name or "-"),
        ("Operador (ID de rastreamento)", f"#{record.track_id}"),
        ("Tempo sem EPI na zona", f"{record.elapsed_s:.1f} s"),
        ("Registro de auditoria", f"#{record.id}"),
        ("Arquivo de evidência", record.evidence_path or "não gerado"),
        ("SHA-256 da evidência", record.evidence_sha256 or "-"),
    ]
    text = "SafeWork - Infração de EPI confirmada\n\n" + "\n".join(
        f"{label}: {value}" for label, value in rows
    ) + "\n\nMensagem automática. A imagem de evidência segue em anexo.\n"
    table = "".join(
        f"<tr><td style='padding:4px 12px 4px 0;color:#555'>{html.escape(label)}</td>"
        f"<td style='padding:4px 0'><b>{html.escape(str(value))}</b></td></tr>"
        for label, value in rows
    )
    body = (
        "<div style='font-family:Arial,sans-serif'>"
        "<h2 style='color:#c00000;margin:0 0 8px'>Infração de EPI confirmada</h2>"
        f"<table style='border-collapse:collapse;font-size:14px'>{table}</table>"
        "<p style='color:#777;font-size:12px'>Mensagem automática do SafeWork. "
        "A imagem de evidência segue em anexo.</p></div>"
    )
    alternative = MIMEMultipart("alternative")
    alternative.attach(MIMEText(text, "plain", "utf-8"))
    alternative.attach(MIMEText(body, "html", "utf-8"))
    message.attach(alternative)

    if attachment is not None and attachment.exists():
        image = MIMEImage(attachment.read_bytes(), _subtype="jpeg")
        image.add_header("Content-Disposition", "attachment", filename=attachment.name)
        message.attach(image)
    return message


@dataclass(frozen=True)
class _Job:
    record: ViolationRecord
    attachment: Path | None


class NullNotifier:
    """Usado quando o e-mail está desativado: registra o status e não envia nada."""

    initial_status = EmailStatus.DISABLED

    def submit(self, record: ViolationRecord, attachment: Path | None) -> None:
        pass

    def start(self) -> "NullNotifier":
        return self

    def stop(self) -> None:
        pass

    def __enter__(self) -> "NullNotifier":
        return self

    def __exit__(self, *_: object) -> None:
        pass


class EmailNotifier:
    """Fila + thread de envio SMTP, com novas tentativas e backoff exponencial."""

    initial_status = EmailStatus.PENDING

    def __init__(
        self,
        config: EmailConfig,
        on_result: ResultCallback | None = None,
        smtp_factory: SmtpFactory = default_smtp_factory,
    ) -> None:
        self._config = config
        self._on_result = on_result
        self._smtp_factory = smtp_factory
        self._queue: queue.Queue[_Job] = queue.Queue(maxsize=config.queue_size)
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self.sent = 0
        self.failed = 0

    # ------------------------------------------------------------------ ciclo de vida
    def start(self) -> "EmailNotifier":
        if self._thread is None:
            self._stopping.clear()
            self._thread = threading.Thread(target=self._worker, name="smtp-sender",
                                            daemon=True)
            self._thread.start()
            logger.info("Notificação por e-mail ativa: %s:%d -> %s", self._config.host,
                        self._config.port, ", ".join(self._config.recipients))
        return self

    def stop(self) -> None:
        """Pede o encerramento e espera até `shutdown_timeout_s` pelas pendências."""
        if self._thread is None:
            return
        self._stopping.set()
        self._thread.join(timeout=self._config.shutdown_timeout_s)
        if self._thread.is_alive():
            logger.warning("Encerrado com e-mails ainda pendentes; ficam como 'pendente' "
                           "na auditoria.")
        self._thread = None

    def __enter__(self) -> "EmailNotifier":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------ uso
    def submit(self, record: ViolationRecord, attachment: Path | None) -> None:
        """Enfileira o envio. Nunca bloqueia o chamador."""
        try:
            self._queue.put_nowait(_Job(record, attachment))
        except queue.Full:
            logger.error("Fila de e-mail cheia: notificação do registro #%s descartada.",
                         record.id)
            self._report(record, EmailStatus.DROPPED, "fila de envio cheia")

    def send_test(self) -> None:
        """Envia um e-mail de teste de forma síncrona (levanta exceção em caso de erro)."""
        message = MIMEMultipart()
        message["Subject"] = Header("[SafeWork] Teste de configuração SMTP", "utf-8")
        message["From"] = self._config.sender
        message["To"] = ", ".join(self._config.recipients)
        message["Date"] = formatdate(localtime=True)
        message.attach(MIMEText("Se você recebeu esta mensagem, o SMTP do SafeWork está "
                                "configurado corretamente.", "plain", "utf-8"))
        self._deliver(message)

    # ------------------------------------------------------------------ thread de envio
    def _worker(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stopping.is_set():
                    return
                continue
            try:
                self._process(job)
            finally:
                self._queue.task_done()

    def _process(self, job: _Job) -> None:
        record = job.record
        try:
            message = build_message(record, self._config, job.attachment)
        except Exception as exc:
            logger.exception("Falha ao montar o e-mail do registro #%s.", record.id)
            self._finish(record, EmailStatus.FAILED, f"montagem: {exc}")
            return

        delay = self._config.retry_delay_s
        for attempt in range(1, self._config.max_retries + 1):
            try:
                self._deliver(message)
                logger.info("E-mail do registro #%s enviado (tentativa %d).", record.id, attempt)
                self._finish(record, EmailStatus.SENT, None)
                return
            except _PERMANENT_ERRORS as exc:
                hint = (" Verifique usuário e senha (no Gmail, use uma senha de app)."
                        if isinstance(exc, smtplib.SMTPAuthenticationError) else "")
                logger.error("Envio do registro #%s recusado: %s.%s", record.id, exc, hint)
                self._finish(record, EmailStatus.FAILED, str(exc))
                return
            except (smtplib.SMTPException, OSError) as exc:  # rede, timeout, servidor ocupado
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Tentativa %d/%d do registro #%s falhou: %s", attempt,
                               self._config.max_retries, record.id, last_error)
                if attempt < self._config.max_retries:
                    # Durante o encerramento, tenta de novo sem esperar.
                    self._stopping.wait(delay)
                    delay *= 2
        self._finish(record, EmailStatus.FAILED, last_error)

    def _deliver(self, message: MIMEMultipart) -> None:
        connection = self._smtp_factory(self._config)
        try:
            password = self._config.resolved_password()
            if self._config.username and password:
                connection.login(self._config.username, password)
            # send_message gera as quebras de linha CRLF exigidas pelo SMTP (RFC 5321).
            # Com sendmail(message.as_bytes()), as linhas saem com LF e servidores
            # estritos recusam a mensagem ("Line too long").
            connection.send_message(message, self._config.sender,
                                    list(self._config.recipients))
        finally:
            try:
                connection.quit()
            except Exception:
                pass

    def _finish(self, record: ViolationRecord, status: str, error: str | None) -> None:
        if status == EmailStatus.SENT:
            self.sent += 1
        else:
            self.failed += 1
        self._report(record, status, error)

    def _report(self, record: ViolationRecord, status: str, error: str | None) -> None:
        if self._on_result is None or record.id is None:
            return
        try:
            self._on_result(record.id, status, error)
        except Exception:
            logger.exception("Falha ao registrar o status do e-mail #%s.", record.id)
