"""Lógica de apresentação do dashboard, sem dependência de Tkinter.

Tudo o que decide *o que* mostrar (textos, cores, escala da imagem, conversão
de coordenadas) fica aqui como funções puras, testáveis sem tela.
"""

from __future__ import annotations

from dataclasses import dataclass

from safework.gui import theme
from safework.logic import AlertEvent, EventType
from safework.monitoring import MonitorStats, SourceStatus


@dataclass(frozen=True)
class Badge:
    """Indicador principal de conformidade."""

    title: str
    detail: str
    color: str
    text_color: str = theme.TEXT


@dataclass(frozen=True)
class Pill:
    text: str
    color: str


_EVENT_LABELS = {
    EventType.WARNING_STARTED: "Alerta iniciado",
    EventType.VIOLATION_CONFIRMED: "INFRAÇÃO CONFIRMADA",
    EventType.VIOLATION_REMINDER: "Infração persistente",
    EventType.RESOLVED: "Regularizado",
    EventType.TRACK_LOST: "Saiu de cena",
}

_EVENT_COLORS = {
    EventType.WARNING_STARTED: theme.WARNING,
    EventType.VIOLATION_CONFIRMED: theme.DANGER,
    EventType.VIOLATION_REMINDER: theme.DANGER,
    EventType.RESOLVED: theme.OK,
    EventType.TRACK_LOST: theme.TEXT_MUTED,
}


def compliance_badge(stats: MonitorStats) -> Badge:
    """Situação geral da área, do mais grave para o menos grave."""
    status = stats.status
    if status is SourceStatus.ERROR:
        return Badge("ERRO", stats.error or "Falha no monitoramento", theme.DANGER)
    if status is SourceStatus.LOADING_MODEL:
        return Badge("INICIANDO", "Carregando o modelo de detecção...", theme.NEUTRAL)
    if status in (SourceStatus.STARTING, SourceStatus.CONNECTING):
        return Badge("INICIANDO", "Conectando à câmera...", theme.NEUTRAL)
    if status is SourceStatus.NO_SIGNAL:
        return Badge("SEM SINAL", "Reconectando à câmera; nada está sendo verificado",
                     theme.WARNING, theme.BACKGROUND)
    if status in (SourceStatus.FINISHED, SourceStatus.STOPPED):
        return Badge("ENCERRADO", "Monitoramento finalizado", theme.NEUTRAL)
    if stats.violations_active:
        return Badge("INFRAÇÃO", f"{plural(stats.violations_active, 'operador')} sem EPI "
                     "após 5 s - evidência registrada", theme.DANGER)
    if stats.warnings:
        remaining = stats.min_remaining_s or 0.0
        return Badge("ALERTA", f"{plural(stats.warnings, 'operador')} sem EPI - "
                     f"confirmação em {remaining:.1f} s", theme.WARNING, theme.BACKGROUND)
    if stats.persons_in_zone == 0:
        return Badge("ÁREA LIVRE", "Nenhum operador na zona de risco", theme.NEUTRAL)
    if stats.unverified:
        return Badge("NÃO VERIFICADO", "O modelo carregado não detecta EPI",
                     theme.WARNING, theme.BACKGROUND)
    return Badge("CONFORME", f"{plural(stats.compliant, 'operador')} com EPI na zona",
                 theme.OK)


def status_pill(stats: MonitorStats) -> Pill:
    """Estado da conexão com a câmera, mostrado no cabeçalho."""
    mapping = {
        SourceStatus.LIVE: Pill("● AO VIVO", theme.OK),
        SourceStatus.NO_SIGNAL: Pill("● SEM SINAL", theme.WARNING),
        SourceStatus.ERROR: Pill("● ERRO", theme.DANGER),
        SourceStatus.FINISHED: Pill("● ENCERRADO", theme.NEUTRAL),
        SourceStatus.STOPPED: Pill("● PARADO", theme.NEUTRAL),
    }
    return mapping.get(stats.status, Pill("● CONECTANDO", theme.INFO))


def plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "es" if word.endswith("r") else "s")


def format_event(event: AlertEvent) -> tuple[str, str]:
    """(texto, cor) de uma linha do histórico de eventos."""
    label = _EVENT_LABELS[event.type]
    text = f"{event.occurred_at:%H:%M:%S}  #{event.track_id}  {label}"
    if event.type in (EventType.WARNING_STARTED, EventType.VIOLATION_CONFIRMED,
                      EventType.VIOLATION_REMINDER):
        missing = ", ".join(sorted(item.label for item in event.missing_ppe))
        if missing:
            text += f" ({missing})"
    return text, _EVENT_COLORS[event.type]


def alarm_text(stats: MonitorStats) -> tuple[str, str]:
    return ("Silenciado", theme.WARNING) if stats.muted else ("Ativo", theme.OK)


def email_text(stats: MonitorStats) -> tuple[str, str]:
    if not stats.email_enabled:
        return "Desativado", theme.TEXT_MUTED
    text = f"Ativo · {stats.emails_sent} enviado(s)"
    if stats.emails_failed:
        return f"{text} · {stats.emails_failed} falha(s)", theme.DANGER
    return text, theme.OK


def fit_size(width: int, height: int, box_width: int, box_height: int) -> tuple[int, int]:
    """Maior tamanho que cabe na caixa preservando a proporção da imagem."""
    if width <= 0 or height <= 0 or box_width <= 1 or box_height <= 1:
        return max(width, 1), max(height, 1)
    scale = min(box_width / width, box_height / height)
    return max(int(width * scale), 1), max(int(height * scale), 1)


def audit_message(total: int, problems: list[str]) -> tuple[str, str, bool]:
    """(título, mensagem, íntegro) do resultado da verificação de auditoria."""
    if not problems:
        return ("Auditoria íntegra",
                f"{plural(total, 'registro')} verificado(s).\n\n"
                "A cadeia de hashes está intacta e todas as imagens de evidência "
                "conferem com o SHA-256 registrado.", True)
    shown = "\n".join(f"• {item}" for item in problems[:10])
    more = f"\n... e mais {len(problems) - 10}." if len(problems) > 10 else ""
    return ("Auditoria COMPROMETIDA",
            f"{len(problems)} problema(s) em {plural(total, 'registro')}:\n\n{shown}{more}",
            False)
