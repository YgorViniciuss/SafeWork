"""Módulo 4: dashboard de monitoramento industrial (CustomTkinter).

`Dashboard` é importado sob demanda: assim `safework.gui.viewmodel` funciona
(e é testável) mesmo numa instalação Python sem Tkinter.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Dashboard", "ZoneEditorDialog"]


def __getattr__(name: str) -> Any:
    if name == "Dashboard":
        from safework.gui.dashboard import Dashboard

        return Dashboard
    if name == "ZoneEditorDialog":
        from safework.gui.zone_editor import ZoneEditorDialog

        return ZoneEditorDialog
    raise AttributeError(name)
