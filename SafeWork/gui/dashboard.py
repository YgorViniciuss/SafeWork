"""Janela principal do SafeWork: vídeo, painel de métricas e barra de ações.

    ┌──────────────────────────────────────────────────────────────┐
    │ SAFEWORK  CAM-01 · Setor de Prensas        ● AO VIVO  22:41 │  cabeçalho
    ├───────────────────────────────────────────┬──────────────────┤
    │                                           │ CONFORME         │
    │      feed de vídeo com overlays           │ contadores       │
    │      (ROI + caixas + alertas de 5 s)      │ situação         │
    │                                           │ eventos          │
    ├───────────────────────────────────────────┴──────────────────┤
    │ [Editar zona E] [Silenciar M] [Verificar auditoria V] [Sair Q]│  ações
    └──────────────────────────────────────────────────────────────┘

O monitoramento roda numa thread (MonitoringService). A interface só lê o
frame e as métricas mais recentes a cada `poll_ms`, então nunca trava, mesmo
que a inferência esteja lenta. Toda chamada ao Tk acontece na thread principal.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from collections.abc import Callable
from datetime import datetime
from tkinter import messagebox
from typing import Protocol

import customtkinter as ctk

from safework.core.config import CameraConfig
from safework.gui import theme
from safework.gui.metrics_panel import MetricsPanel
from safework.gui.video_panel import VideoPanel
from safework.gui.viewmodel import audit_message, compliance_badge, status_pill
from safework.gui.zone_editor import ZoneEditorDialog
from safework.logic import AlertEvent
from safework.monitoring import FrameUpdate, MonitorStats
from safework.vision.roi import RiskZone, ZoneRepository

logger = logging.getLogger(__name__)

AuditCheck = Callable[[], tuple[int, list[str]]]


class MonitorService(Protocol):
    """O que o dashboard precisa do serviço (permite um dublê nos testes)."""

    latest: FrameUpdate | None
    stats: MonitorStats
    finished: bool
    muted: bool

    def recent_events(self) -> list[AlertEvent]: ...
    def set_muted(self, muted: bool) -> None: ...
    def reload_zones(self) -> int: ...
    def start(self) -> object: ...
    def stop(self, timeout: float | None = None) -> None: ...


class Dashboard(ctk.CTk):
    """Dashboard de monitoramento industrial."""

    def __init__(
        self,
        service: MonitorService,
        zones: ZoneRepository,
        audit_check: AuditCheck,
        camera: CameraConfig,
        title: str = "SafeWork - Monitoramento",
        poll_ms: int = 30,
    ) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__(fg_color=theme.BACKGROUND)
        self.title(title)
        self.geometry("1366x800")
        self.minsize(1100, 680)

        self._service = service
        self._zones = zones
        self._audit_check = audit_check
        self._camera = camera
        self._poll_ms = poll_ms
        self._last_sequence = -1
        self._closing = False
        self._audit_thread: threading.Thread | None = None
        self._audit_result: tuple[int, list[str]] | Exception | None = None
        self._toast_job: str | None = None
        self.editor: ZoneEditorDialog | None = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self._build_body()
        self._build_actions()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self.request_exit)

    # ------------------------------------------------------------------ montagem
    def _build_header(self) -> None:
        header = ctk.CTkFrame(self, fg_color=theme.SURFACE, corner_radius=0, height=56)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(header, text="SAFEWORK", text_color=theme.TEXT,
                     font=ctk.CTkFont(theme.FONT_FAMILY, 22, "bold")).grid(
            row=0, column=0, padx=(18, 12), pady=10)
        camera = self._camera
        ctk.CTkLabel(header, text=f"{camera.id}  ·  {camera.name}  ·  {camera.area}",
                     text_color=theme.TEXT_MUTED,
                     font=ctk.CTkFont(theme.FONT_FAMILY, 14)).grid(row=0, column=1, sticky="w")
        self.status_label = ctk.CTkLabel(header, text="● CONECTANDO", text_color=theme.INFO,
                                         font=ctk.CTkFont(theme.FONT_FAMILY, 14, "bold"))
        self.status_label.grid(row=0, column=2, padx=12)
        self.clock_label = ctk.CTkLabel(header, text="", text_color=theme.TEXT,
                                        font=ctk.CTkFont(theme.MONO_FAMILY, 14))
        self.clock_label.grid(row=0, column=3, padx=(0, 18))

    def _build_body(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)
        self.video = VideoPanel(body)
        self.video.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.metrics = MetricsPanel(body)
        self.metrics.grid(row=0, column=1, sticky="ns")

    def _build_actions(self) -> None:
        bar = ctk.CTkFrame(self, fg_color=theme.SURFACE, corner_radius=0, height=64)
        bar.grid(row=2, column=0, sticky="ew")
        style = {"height": 42, "width": 210, "corner_radius": 6,
                 "font": ctk.CTkFont(theme.FONT_FAMILY, 14, "bold")}
        self.edit_button = ctk.CTkButton(bar, text="Editar zona de risco  [E]",
                                         command=self.open_zone_editor, fg_color=theme.INFO,
                                         **style)
        self.mute_button = ctk.CTkButton(bar, text="Silenciar alarme  [M]",
                                         command=self.toggle_mute, fg_color=theme.CARD,
                                         hover_color=theme.BORDER, **style)
        self.audit_button = ctk.CTkButton(bar, text="Verificar auditoria  [V]",
                                          command=self.verify_audit, fg_color=theme.CARD,
                                          hover_color=theme.BORDER, **style)
        self.exit_button = ctk.CTkButton(bar, text="Sair  [Q]", command=self.request_exit,
                                         fg_color=theme.DANGER, hover_color="#a92c2c",
                                         **{**style, "width": 130})
        for index, button in enumerate((self.edit_button, self.mute_button, self.audit_button)):
            button.grid(row=0, column=index, padx=(12 if index == 0 else 6, 6), pady=11)
        self.toast = ctk.CTkLabel(bar, text="", text_color=theme.TEXT_MUTED,
                                  font=ctk.CTkFont(theme.FONT_FAMILY, 13))
        self.toast.grid(row=0, column=3, sticky="w", padx=12)
        bar.grid_columnconfigure(3, weight=1)
        self.exit_button.grid(row=0, column=4, padx=12, pady=11)

    def _bind_shortcuts(self) -> None:
        shortcuts = {
            "e": self.open_zone_editor,
            "m": self.toggle_mute,
            "v": self.verify_audit,
            "q": self.request_exit,
        }
        for key, action in shortcuts.items():
            for variant in (key, key.upper()):
                self.bind(f"<KeyPress-{variant}>", lambda _e, a=action: self._shortcut(a))
        self.bind("<Escape>", lambda _e: self._shortcut(self.request_exit))
        self.bind("<F11>", lambda _e: self.attributes(
            "-fullscreen", not self.attributes("-fullscreen")))

    def _shortcut(self, action: Callable[[], object]) -> None:
        # Atalhos valem só com o foco no dashboard (não enquanto o editor está aberto).
        if self.editor is None or not self.editor.winfo_exists():
            action()

    # ------------------------------------------------------------------ ciclo de vida
    def start(self) -> "Dashboard":
        self._service.start()
        self._poll_job = self.after(self._poll_ms, self._poll)
        return self

    def _poll(self) -> None:
        """Atualiza a tela com o estado mais recente do serviço."""
        try:
            self.refresh()
        except tk.TclError:
            return  # janela destruída durante a atualização
        if self._closing and self._service.finished:
            self.destroy()
            return
        self._poll_job = self.after(self._poll_ms, self._poll)

    def refresh(self) -> None:
        update = self._service.latest
        if update is not None and update.sequence != self._last_sequence:
            self.video.show(update.image)
            self._last_sequence = update.sequence
        stats = self._service.stats
        if update is None:
            self.video.show_message(compliance_badge(stats).detail)
        self.metrics.update_view(stats, self._service.recent_events())
        pill = status_pill(stats)
        self.status_label.configure(text=pill.text, text_color=pill.color)
        self.clock_label.configure(text=datetime.now().strftime("%d/%m/%Y  %H:%M:%S"))
        self._check_audit_result()

    # ------------------------------------------------------------------ ações
    def open_zone_editor(self) -> None:
        if self._closing:
            return
        update = self._service.latest
        if update is None:
            self.show_toast("Aguarde o primeiro frame da câmera para editar as zonas.",
                            theme.WARNING)
            return
        self.editor = ZoneEditorDialog(self, update.raw, self._zones, self._on_zones_saved)

    def _on_zones_saved(self, zones: list[RiskZone]) -> None:
        count = self._service.reload_zones()
        self.show_toast(f"Zonas salvas e aplicadas: {count}.", theme.OK)

    def toggle_mute(self) -> None:
        muted = not self._service.muted
        self._service.set_muted(muted)
        if muted:
            self.mute_button.configure(text="Reativar alarme  [M]", fg_color=theme.WARNING,
                                       text_color=theme.BACKGROUND)
            self.show_toast("Alarme sonoro silenciado. Alertas visuais e auditoria continuam.",
                            theme.WARNING)
        else:
            self.mute_button.configure(text="Silenciar alarme  [M]", fg_color=theme.CARD,
                                       text_color=theme.TEXT)
            self.show_toast("Alarme sonoro reativado.", theme.OK)

    def verify_audit(self) -> None:
        """Roda a verificação numa thread: com muitas imagens, pode levar segundos."""
        if self._audit_thread is not None and self._audit_thread.is_alive():
            return
        self.audit_button.configure(state="disabled", text="Verificando...")
        self._audit_result = None

        def work() -> None:
            try:
                self._audit_result = self._audit_check()
            except Exception as exc:
                self._audit_result = exc

        self._audit_thread = threading.Thread(target=work, name="audit-check", daemon=True)
        self._audit_thread.start()

    def _check_audit_result(self) -> None:
        result = self._audit_result
        if result is None:
            return
        self._audit_result = None
        self.audit_button.configure(state="normal", text="Verificar auditoria  [V]")
        if isinstance(result, Exception):
            self.show_message("Erro na verificação", str(result), error=True)
            return
        title, message, ok = audit_message(*result)
        self.show_toast(title, theme.OK if ok else theme.DANGER)
        self.show_message(title, message, error=not ok)

    def request_exit(self) -> None:
        """Encerra em segundo plano (câmera, alarme e e-mails pendentes) e fecha."""
        if self._closing:
            return
        self._closing = True
        for button in (self.edit_button, self.mute_button, self.audit_button, self.exit_button):
            button.configure(state="disabled")
        self.show_toast("Encerrando... finalizando câmera e envios pendentes.", theme.TEXT_MUTED,
                        sticky=True)
        threading.Thread(target=self._service.stop, name="shutdown", daemon=True).start()

    # ------------------------------------------------------------------ feedback
    def show_toast(self, text: str, color: str = theme.TEXT_MUTED, sticky: bool = False) -> None:
        self.toast.configure(text=text, text_color=color)
        if self._toast_job is not None:
            self.after_cancel(self._toast_job)
            self._toast_job = None
        if not sticky:
            self._toast_job = self.after(5000, lambda: self.toast.configure(text=""))

    def show_message(self, title: str, message: str, error: bool = False) -> None:
        (messagebox.showerror if error else messagebox.showinfo)(title, message, parent=self)
