"""Exibição do feed de vídeo com os overlays já desenhados pelo pipeline."""

from __future__ import annotations

import tkinter as tk

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from safework.gui import theme
from safework.gui.viewmodel import fit_size


class VideoPanel(ctk.CTkFrame):
    """Mostra o frame mais recente ajustado ao espaço disponível."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, fg_color="#000000", corner_radius=8)
        # Um tk.Label com PhotoImage é mais rápido que CTkImage para atualizar ~30x/s.
        self._label = tk.Label(
            self, bg="#000000", fg=theme.TEXT_MUTED, text="Aguardando vídeo...",
            font=(theme.FONT_FAMILY, 14), compound="center", borderwidth=0,
        )
        self._label.pack(fill="both", expand=True, padx=2, pady=2)
        self._photo: ImageTk.PhotoImage | None = None
        self.frames_shown = 0
        self.displayed_size: tuple[int, int] = (0, 0)

    def show(self, image_bgr: np.ndarray) -> None:
        height, width = image_bgr.shape[:2]
        box_width = self._label.winfo_width()
        box_height = self._label.winfo_height()
        target = fit_size(width, height, box_width, box_height)
        if target != (width, height):
            interpolation = cv2.INTER_AREA if target[0] < width else cv2.INTER_LINEAR
            image_bgr = cv2.resize(image_bgr, target, interpolation=interpolation)
        # Aqui a conversão BGR -> RGB é necessária: o Tk/Pillow exibem RGB.
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._label.configure(image=self._photo, text="")
        self.frames_shown += 1
        self.displayed_size = target

    def show_message(self, text: str) -> None:
        """Texto sobre fundo preto (antes do primeiro frame ou em caso de erro)."""
        if self._photo is None:
            self._label.configure(text=text)
