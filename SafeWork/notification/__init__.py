"""Notificações: alertas locais (Módulo 2). O e-mail SMTP entra aqui no Módulo 4."""

from safework.notification.sound import AlarmPlayer, SilentBackend, create_backend
from safework.notification.visual import AlertOverlay

__all__ = ["AlarmPlayer", "AlertOverlay", "SilentBackend", "create_backend"]
