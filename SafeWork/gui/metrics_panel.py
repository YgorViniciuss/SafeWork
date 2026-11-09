"""Painel lateral: conformidade, contadores, desempenho e histórico de eventos."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence

import customtkinter as ctk

from safework.gui import theme
from safework.gui.viewmodel import alarm_text, compliance_badge, email_text, format_event
from safework.logic import AlertEvent
from safework.monitoring import MonitorStats


class _MetricCard(ctk.CTkFrame):
    """Número grande com rótulo."""

    def __init__(self, master: tk.Misc, label: str) -> None:
        super().__init__(master, fg_color=theme.CARD, corner_radius=8)
        self.value = ctk.CTkLabel(self, text="0", text_color=theme.TEXT,
                                  font=ctk.CTkFont(theme.FONT_FAMILY, 28, "bold"))
        self.value.pack(anchor="w", padx=12, pady=(8, 0))
        ctk.CTkLabel(self, text=label, text_color=theme.TEXT_MUTED,
                     font=ctk.CTkFont(theme.FONT_FAMILY, 12)).pack(anchor="w", padx=12,
                                                                    pady=(0, 8))

    def set(self, text: str, color: str = theme.TEXT) -> None:
        if self.value.cget("text") != text or self.value.cget("text_color") != color:
            self.value.configure(text=text, text_color=color)


class _InfoRow(ctk.CTkFrame):
    """Linha 'rótulo ......... valor'."""

    def __init__(self, master: tk.Misc, label: str) -> None:
        super().__init__(master, fg_color="transparent")
        ctk.CTkLabel(self, text=label, text_color=theme.TEXT_MUTED,
                     font=ctk.CTkFont(theme.FONT_FAMILY, 13)).pack(side="left")
        self.value = ctk.CTkLabel(self, text="-", text_color=theme.TEXT,
                                  font=ctk.CTkFont(theme.FONT_FAMILY, 13, "bold"))
        self.value.pack(side="right")

    def set(self, text: str, color: str = theme.TEXT) -> None:
        if self.value.cget("text") != text or self.value.cget("text_color") != color:
            self.value.configure(text=text, text_color=color)


class MetricsPanel(ctk.CTkScrollableFrame):
    """Barra lateral do dashboard."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, width=theme.SIDEBAR_WIDTH, fg_color=theme.SURFACE,
                         corner_radius=8)

        # Indicador principal de conformidade
        self.badge = ctk.CTkFrame(self, fg_color=theme.NEUTRAL, corner_radius=8)
        self.badge.pack(fill="x", padx=8, pady=(8, 6))
        self.badge_title = ctk.CTkLabel(self.badge, text="INICIANDO", text_color=theme.TEXT,
                                        font=ctk.CTkFont(theme.FONT_FAMILY, 26, "bold"),
                                        anchor="w")
        self.badge_title.pack(anchor="w", padx=14, pady=(10, 0))
        self.badge_detail = ctk.CTkLabel(self.badge, text="", text_color=theme.TEXT,
                                         font=ctk.CTkFont(theme.FONT_FAMILY, 13),
                                         wraplength=theme.SIDEBAR_WIDTH - 90, justify="left",
                                         anchor="w")
        self.badge_detail.pack(anchor="w", padx=14, pady=(0, 10))

        # Contadores
        grid = ctk.CTkFrame(self, fg_color="transparent")
        grid.pack(fill="x", padx=4)
        grid.grid_columnconfigure((0, 1), weight=1, uniform="cards")
        self.cards: dict[str, _MetricCard] = {}
        cards = [
            ("in_zone", "Operadores na zona"),
            ("warnings", "Em alerta (5 s)"),
            ("session", "Infrações na sessão"),
            ("total", "Total na auditoria"),
            ("fps", "FPS"),
            ("inference", "Inferência (ms)"),
        ]
        for index, (key, label) in enumerate(cards):
            card = _MetricCard(grid, label)
            card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=4, pady=4)
            self.cards[key] = card

        # Situação do sistema
        info = ctk.CTkFrame(self, fg_color=theme.CARD, corner_radius=8)
        info.pack(fill="x", padx=8, pady=6)
        self.rows: dict[str, _InfoRow] = {}
        for key, label in (("detected", "Pessoas detectadas"), ("compliance", "Conformes / sem EPI"),
                           ("zones", "Zonas de risco"), ("alarm", "Alarme sonoro"),
                           ("email", "E-mail"), ("ppe", "Detecção de EPI")):
            row = _InfoRow(info, label)
            row.pack(fill="x", padx=12, pady=3)
            self.rows[key] = row

        # Histórico de eventos
        ctk.CTkLabel(self, text="Eventos recentes", text_color=theme.TEXT_MUTED,
                     font=ctk.CTkFont(theme.FONT_FAMILY, 13, "bold")).pack(anchor="w", padx=12,
                                                                          pady=(8, 2))
        self.events = tk.Text(self, height=10, bg=theme.CARD, fg=theme.TEXT, relief="flat",
                              font=(theme.MONO_FAMILY, 10), wrap="word", borderwidth=0,
                              highlightthickness=0, padx=8, pady=6)
        self.events.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.events.configure(state="disabled")
        self._events_key: tuple = ()

    def update_view(self, stats: MonitorStats, events: Sequence[AlertEvent]) -> None:
        badge = compliance_badge(stats)
        if self.badge.cget("fg_color") != badge.color:
            self.badge.configure(fg_color=badge.color)
        for label, text in ((self.badge_title, badge.title), (self.badge_detail, badge.detail)):
            if label.cget("text") != text or label.cget("text_color") != badge.text_color:
                label.configure(text=text, text_color=badge.text_color)

        self.cards["in_zone"].set(str(stats.persons_in_zone))
        self.cards["warnings"].set(str(stats.warnings),
                                   theme.WARNING if stats.warnings else theme.TEXT)
        self.cards["session"].set(str(stats.session_violations),
                                  theme.DANGER if stats.session_violations else theme.TEXT)
        self.cards["total"].set(str(stats.total_violations))
        self.cards["fps"].set(f"{stats.fps:.1f}")
        self.cards["inference"].set(f"{stats.inference_ms:.0f}")

        self.rows["detected"].set(str(stats.persons_detected))
        self.rows["compliance"].set(f"{stats.compliant} / {stats.non_compliant}",
                                    theme.DANGER if stats.non_compliant else theme.TEXT)
        self.rows["zones"].set(str(stats.zones), theme.TEXT if stats.zones else theme.WARNING)
        self.rows["alarm"].set(*alarm_text(stats))
        self.rows["email"].set(*email_text(stats))
        self.rows["ppe"].set(*(("Ativa", theme.OK) if stats.ppe_verifiable
                               else ("Indisponível", theme.WARNING)))
        self._update_events(events)

    def _update_events(self, events: Sequence[AlertEvent]) -> None:
        shown = list(events[:30])
        key = tuple(id(event) for event in shown)
        if key == self._events_key:
            return  # evita redesenhar o texto a cada frame
        self._events_key = key
        self.events.configure(state="normal")
        self.events.delete("1.0", "end")
        if not shown:
            self.events.insert("end", "Nenhum evento nesta sessão.", ("muted",))
            self.events.tag_configure("muted", foreground=theme.TEXT_MUTED)
        for index, event in enumerate(shown):
            text, color = format_event(event)
            tag = f"color{index}"
            self.events.tag_configure(tag, foreground=color)
            self.events.insert("end", text + "\n", (tag,))
        self.events.configure(state="disabled")

    def event_lines(self) -> list[str]:
        """Linhas exibidas no histórico (usado pelos testes)."""
        return [line for line in self.events.get("1.0", "end").splitlines() if line]
