"""Notificações: alertas locais (Módulo 2) e e-mail SMTP assíncrono (Módulo 3)."""

from safework.notification.email import EmailNotifier, NullNotifier, build_message
from safework.notification.sound import AlarmPlayer, SilentBackend, create_backend
from safework.notification.visual import AlertOverlay

__all__ = [
    "AlarmPlayer",
    "AlertOverlay",
    "EmailNotifier",
    "NullNotifier",
    "SilentBackend",
    "build_message",
    "create_backend",
]
