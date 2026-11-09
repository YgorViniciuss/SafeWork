"""Composição da aplicação: cria os objetos e conduz o laço de exibição."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from safework.audit import AuditRepository, EvidenceStore, ViolationAuditor
from safework.core.config import Settings, resolve_path
from safework.logic import EventBus, EventType, LoggingAlertListener
from safework.monitoring import DetectorFactory, FrameUpdate, MonitoringService
from safework.notification import EmailNotifier, NullNotifier
from safework.vision.capture import VideoSource
from safework.vision.detector import YoloDetector
from safework.vision.preprocessing import FramePreprocessor
from safework.vision.roi import ZoneRepository
from safework.vision.roi_editor import RoiEditor

logger = logging.getLogger(__name__)

_EXIT_KEYS = (ord("q"), ord("Q"), 27)


class GuiUnavailable(RuntimeError):
    """A interface gráfica não pode ser aberta neste ambiente."""


class SafeWorkApp:
    """Ponto de entrada orientado a objetos do sistema."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._zones = ZoneRepository(resolve_path(settings.zones_file))
        self.events = EventBus()
        self.events.subscribe(LoggingAlertListener(settings.camera))

        # Módulo 3: evidência, auditoria e e-mail, inscritos na infração confirmada.
        self.audit = AuditRepository(resolve_path(settings.audit.database))
        self.notifier = self._build_notifier()
        self.events.subscribe(
            ViolationAuditor(
                store=EvidenceStore(settings.evidence, settings.camera.id),
                repository=self.audit,
                notifier=self.notifier,
                camera=settings.camera,
            ),
            [EventType.VIOLATION_CONFIRMED, EventType.VIOLATION_REMINDER],
        )

    def _build_notifier(self) -> EmailNotifier | NullNotifier:
        config = self._settings.email
        if not config.enabled:
            logger.info("Notificação por e-mail desativada (email.enabled: false).")
            return NullNotifier()
        if config.username and not config.resolved_password():
            logger.warning("SMTP sem senha: defina a variável %s ou email.password.",
                           config.password_env)
        # O resultado do envio volta para a auditoria (status enviado/falhou).
        return EmailNotifier(config, on_result=self.audit.update_email_status)

    def audit_report(self) -> tuple[int, list[str]]:
        """(total de registros, problemas encontrados). Usado pela GUI e pela CLI."""
        return self.audit.count(), self.audit.verify()

    def verify_audit(self) -> int:
        """Confere a cadeia de hashes da auditoria e as imagens de evidência."""
        total, problems = self.audit_report()
        if problems:
            for problem in problems:
                logger.error("%s", problem)
            logger.error("Trilha de auditoria COMPROMETIDA: %d problema(s) em %d registro(s).",
                         len(problems), total)
            return 1
        logger.info("Trilha de auditoria íntegra: %d registro(s) em %s.", total, self.audit.path)
        return 0

    def test_email(self) -> int:
        """Envia um e-mail de teste e informa o resultado."""
        if not isinstance(self.notifier, EmailNotifier):
            logger.error("Ative email.enabled no settings.yaml antes de testar.")
            return 1
        try:
            self.notifier.send_test()
        except Exception as exc:
            logger.error("Falha no envio de teste: %s: %s", type(exc).__name__, exc)
            return 1
        logger.info("E-mail de teste enviado para %s.", ", ".join(self._settings.email.recipients))
        return 0

    def edit_zones(self) -> int:
        """Captura um frame da fonte e abre o editor de ROI."""
        preprocessor = FramePreprocessor(self._settings.preprocessing)
        with VideoSource(self._settings.camera) as source:
            frame = source.read(timeout=15.0)
        if frame is None:
            logger.error("Nenhum frame recebido da fonte: %s", source.error or "tempo esgotado")
            return 1
        RoiEditor(self._zones).run(preprocessor.process(frame.image))
        return 0

    # ------------------------------------------------------------------ monitoramento
    def create_service(
        self,
        mute: bool = False,
        detector_factory: DetectorFactory = YoloDetector,
    ) -> MonitoringService:
        return MonitoringService(
            settings=self._settings,
            events=self.events,
            audit=self.audit,
            notifier=self.notifier,
            zones=self._zones,
            detector_factory=detector_factory,
            mute=mute,
        )

    def run_gui(self, mute: bool = False) -> int:
        """Abre o dashboard industrial (Módulo 4)."""
        try:
            from safework.gui import Dashboard
        except ImportError as exc:  # Tkinter ou CustomTkinter ausentes
            raise GuiUnavailable(f"interface gráfica indisponível ({exc})") from exc

        service = self.create_service(mute)
        try:
            dashboard = Dashboard(
                service=service,
                zones=self._zones,
                audit_check=self.audit_report,
                camera=self._settings.camera,
                title=self._settings.display.window_name,
            )
        except Exception as exc:  # ex.: sem display (TclError)
            raise GuiUnavailable(f"não foi possível abrir a janela ({exc})") from exc
        dashboard.start()
        dashboard.mainloop()
        return service.exit_code

    def run(
        self,
        headless: bool = False,
        output: str | None = None,
        max_frames: int = 0,
        mute: bool = False,
    ) -> int:
        """Modo sem painel: janela OpenCV simples ou nenhuma janela (--headless)."""
        service = self.create_service(mute)
        writer: cv2.VideoWriter | None = None
        window = self._settings.display.window_name

        def on_frame(update: FrameUpdate) -> bool:
            nonlocal writer
            if output:
                writer = writer or self._create_writer(output, update.image)
                writer.write(update.image)
            if headless:
                return True
            cv2.imshow(window, update.image)
            return not self._user_requested_exit()

        def on_idle() -> bool:
            return headless or not self._user_requested_exit()

        try:
            return service.run(on_frame, on_idle, max_frames)
        finally:
            if writer is not None:
                writer.release()
            if not headless:
                cv2.destroyAllWindows()

    def _user_requested_exit(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        window = self._settings.display.window_name
        try:
            closed = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            closed = False  # a janela ainda não foi criada (sem frames até agora)
        return key in _EXIT_KEYS or closed

    @staticmethod
    def _create_writer(output: str, image: np.ndarray) -> cv2.VideoWriter:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = image.shape[:2]
        return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))
