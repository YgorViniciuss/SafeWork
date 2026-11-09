"""Editor de zonas de risco (ROI) dentro do dashboard.

Usa o frame mais recente da câmera como fundo e a mesma lógica do editor
OpenCV do Módulo 1 (`ZoneEditorState`), agora com botões, nome da zona e
escolha dos EPIs exigidos.

    Clique esquerdo  adiciona vértice      Clique direito / Ctrl+Z  desfaz
    Enter            fecha o polígono      Ctrl+S                   salva
    Esc              cancela
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from safework.core.models import PPE
from safework.gui import theme
from safework.gui.viewmodel import fit_size
from safework.vision.roi import RiskZone, ZoneRepository
from safework.vision.roi_editor import ZoneEditorState

_MAX_CANVAS = (1000, 620)


class ZoneEditorDialog(ctk.CTkToplevel):
    """Janela modal de edição. Chama `on_saved(zonas)` após salvar."""

    def __init__(
        self,
        master: tk.Misc,
        background_bgr: np.ndarray,
        repository: ZoneRepository,
        on_saved: Callable[[list[RiskZone]], None] | None = None,
    ) -> None:
        super().__init__(master, fg_color=theme.BACKGROUND)
        self.title("SafeWork - Editar zonas de risco")
        self.resizable(False, False)
        self._repository = repository
        self._on_saved = on_saved
        self.saved = False

        height, width = background_bgr.shape[:2]
        self.editor_state = ZoneEditorState(width, height, repository.load())
        self._display_size = fit_size(width, height, *_MAX_CANVAS)
        self._scale = self._display_size[0] / width
        resized = cv2.resize(background_bgr, self._display_size, interpolation=cv2.INTER_AREA)
        self._photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        )

        self._build()
        self._redraw()

        self.bind("<Return>", lambda _e: self.close_polygon())
        self.bind("<KP_Enter>", lambda _e: self.close_polygon())
        self.bind("<Control-z>", lambda _e: self.undo())
        self.bind("<Control-s>", lambda _e: self.save())
        self.bind("<Escape>", lambda _e: self.cancel())
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.after(50, self._make_modal)

    # ------------------------------------------------------------------ montagem
    def _build(self) -> None:
        self.canvas = tk.Canvas(self, width=self._display_size[0], height=self._display_size[1],
                                highlightthickness=0, bg="#000000", cursor="crosshair")
        self.canvas.grid(row=0, column=0, padx=12, pady=12)
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self.canvas.bind("<Button-1>", lambda e: self.click(e.x, e.y))
        self.canvas.bind("<Button-3>", lambda _e: self.undo())

        side = ctk.CTkFrame(self, fg_color=theme.SURFACE, width=260)
        side.grid(row=0, column=1, sticky="ns", padx=(0, 12), pady=12)

        def heading(text: str) -> None:
            ctk.CTkLabel(side, text=text, text_color=theme.TEXT_MUTED,
                         font=ctk.CTkFont(theme.FONT_FAMILY, 12, "bold")).pack(
                anchor="w", padx=14, pady=(12, 2))

        heading("NOVA ZONA")
        self.name_entry = ctk.CTkEntry(side, placeholder_text="Nome (ex.: Prensa 03)", width=230)
        self.name_entry.pack(padx=14, pady=4)
        self.helmet_var = tk.BooleanVar(value=True)
        self.vest_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(side, text="Exige capacete", variable=self.helmet_var).pack(
            anchor="w", padx=14, pady=4)
        ctk.CTkCheckBox(side, text="Exige colete", variable=self.vest_var).pack(
            anchor="w", padx=14, pady=4)

        heading("DESENHO")
        buttons = [
            ("Fechar polígono  (Enter)", self.close_polygon, theme.INFO),
            ("Desfazer vértice  (Ctrl+Z)", self.undo, theme.CARD),
            ("Limpar desenho", self.clear, theme.CARD),
            ("Apagar última zona", self.delete_last, theme.CARD),
        ]
        for text, command, color in buttons:
            ctk.CTkButton(side, text=text, command=command, fg_color=color, width=230,
                          hover_color=theme.BORDER).pack(padx=14, pady=4)

        heading("ZONAS")
        self.zone_list = ctk.CTkLabel(side, text="", justify="left", anchor="w",
                                      text_color=theme.TEXT, wraplength=230)
        self.zone_list.pack(anchor="w", padx=14, pady=4)

        self.hint = ctk.CTkLabel(side, text="Clique na imagem para marcar os vértices.",
                                 text_color=theme.TEXT_MUTED, wraplength=230, justify="left")
        self.hint.pack(anchor="w", padx=14, pady=(12, 4))

        actions = ctk.CTkFrame(side, fg_color="transparent")
        actions.pack(side="bottom", fill="x", padx=14, pady=14)
        ctk.CTkButton(actions, text="Salvar  (Ctrl+S)", command=self.save, fg_color=theme.OK,
                      hover_color="#1b7f47").pack(fill="x", pady=4)
        ctk.CTkButton(actions, text="Cancelar  (Esc)", command=self.cancel,
                      fg_color=theme.CARD, hover_color=theme.BORDER).pack(fill="x", pady=4)

    def _make_modal(self) -> None:
        try:
            self.grab_set()
            self.focus_force()
        except tk.TclError:  # janela ainda não visível
            pass

    # ------------------------------------------------------------------ ações
    def click(self, canvas_x: int, canvas_y: int) -> None:
        """Converte o clique na tela para pixels da imagem original."""
        self.editor_state.add_vertex(round(canvas_x / self._scale), round(canvas_y / self._scale))
        self._redraw()

    def undo(self) -> None:
        self.editor_state.undo_vertex()
        self._redraw()

    def clear(self) -> None:
        self.editor_state.clear_draft()
        self._redraw()

    def delete_last(self) -> None:
        self.editor_state.delete_last_zone()
        self._redraw()

    def required_ppe(self) -> frozenset[PPE]:
        items = set()
        if self.helmet_var.get():
            items.add(PPE.HELMET)
        if self.vest_var.get():
            items.add(PPE.VEST)
        return frozenset(items)

    def close_polygon(self) -> RiskZone | None:
        required = self.required_ppe()
        if not required:
            self._set_hint("Marque ao menos um EPI exigido.", theme.WARNING)
            return None
        zone = self.editor_state.close_polygon(required, self.name_entry.get())
        if zone is None:
            self._set_hint("São necessários ao menos 3 vértices.", theme.WARNING)
            return None
        self.name_entry.delete(0, "end")
        self._set_hint(f"'{zone.name}' criada. Salve para aplicar.", theme.OK)
        self._redraw()
        return zone

    def save(self) -> bool:
        if len(self.editor_state.draft) >= 3 and self.close_polygon() is None:
            return False  # rascunho não fechado: fecha automaticamente se possível
        if self.editor_state.draft:
            if not self._confirm("Desenho incompleto",
                                 "Há vértices não fechados em zona. Descartar e salvar?"):
                return False
            self.editor_state.clear_draft()
        self._repository.save(self.editor_state.zones)
        self.saved = True
        if self._on_saved is not None:
            self._on_saved(list(self.editor_state.zones))
        self.destroy()
        return True

    def cancel(self) -> None:
        self.destroy()

    # ------------------------------------------------------------------ desenho
    def _redraw(self) -> None:
        canvas = self.canvas
        canvas.delete("overlay")
        width, height = self._display_size
        for zone in self.editor_state.zones:
            points = [(x * (width - 1), y * (height - 1)) for x, y in zone.points]
            flat = [value for point in points for value in point]
            # Só contorno: "stipple" (transparência) não funciona em todos os sistemas.
            canvas.create_polygon(*flat, outline="#ff9f1a", fill="", width=3, tags="overlay")
            top = min(points, key=lambda point: point[1])
            canvas.create_text(top[0] + 4, top[1] - 10, text=zone.name, fill="#ffffff",
                               anchor="w", font=(theme.FONT_FAMILY, 11, "bold"), tags="overlay")

        draft = [(x * (width - 1), y * (height - 1)) for x, y in self.editor_state.draft]
        if len(draft) > 1:
            canvas.create_line(*[v for p in draft for v in p], fill="#00e5ff", width=2,
                               tags="overlay")
        for x, y in draft:
            canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#00e5ff", outline="#000000",
                               tags="overlay")

        lines = []
        for zone in self.editor_state.zones:
            ppe = " + ".join(sorted(item.label for item in zone.required_ppe))
            lines.append(f"• {zone.name}  ({ppe})")
        self.zone_list.configure(text="\n".join(lines) or "Nenhuma zona.")

    def _set_hint(self, text: str, color: str) -> None:
        self.hint.configure(text=text, text_color=color)

    def _confirm(self, title: str, message: str) -> bool:
        return messagebox.askyesno(title, message, parent=self)
