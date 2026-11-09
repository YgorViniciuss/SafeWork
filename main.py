"""SafeWork - ponto de entrada pela linha de comando.

Exemplos:
    python main.py                               abre o dashboard (GUI) com a fonte do settings.yaml
    python main.py --cv-window                   janela OpenCV simples, sem painel
    python main.py --edit-zones                  desenha as zonas de risco (editor OpenCV)
    python main.py --source data/teste.mp4       usa um vídeo
    python main.py --source data/teste.mp4 --headless --output logs/saida.mp4
    python main.py --mute                        monitora sem alarme sonoro
    python main.py --test-email                  envia um e-mail de teste
    python main.py --verify-audit                confere a integridade da auditoria
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from safework.app import GuiUnavailable, SafeWorkApp
from safework.core.config import ConfigError, Settings
from safework.utils.logger import setup_logging

logger = logging.getLogger("safework.main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafeWork - monitoramento de EPIs")
    parser.add_argument("--config", default="config/settings.yaml", help="arquivo de configuração")
    parser.add_argument("--source", help="substitui a fonte: índice, URL RTSP ou arquivo")
    parser.add_argument("--edit-zones", action="store_true", help="abre o editor de zonas")
    parser.add_argument("--headless", action="store_true", help="executa sem janela")
    parser.add_argument("--cv-window", action="store_true",
                        help="usa a janela OpenCV simples em vez do dashboard")
    parser.add_argument("--output",
                        help="grava o vídeo anotado em .mp4 (modo --cv-window ou --headless)")
    parser.add_argument("--max-frames", type=int, default=0, help="encerra após N frames")
    parser.add_argument("--mute", action="store_true",
                        help="inicia com o alarme sonoro silenciado")
    parser.add_argument("--warning-seconds", type=float,
                        help="substitui alerts.warning_seconds (padrão: 5 s)")
    parser.add_argument("--test-email", action="store_true",
                        help="envia um e-mail de teste com a configuração SMTP e sai")
    parser.add_argument("--verify-audit", action="store_true",
                        help="confere a cadeia de hashes e as evidências e sai")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    try:
        settings = Settings.load(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2

    if args.source is not None:
        settings = replace(settings, camera=replace(settings.camera, source=args.source))
    if args.warning_seconds is not None:
        try:
            alerts = replace(settings.alerts, warning_seconds=args.warning_seconds)
        except ConfigError as exc:
            logger.error("%s", exc)
            return 2
        settings = replace(settings, alerts=alerts)

    app = SafeWorkApp(settings)
    if args.verify_audit:
        return app.verify_audit()
    if args.test_email:
        return app.test_email()
    if args.edit_zones:
        return app.edit_zones()

    # Gravação de vídeo e limite de frames são recursos do modo sem dashboard.
    use_dashboard = not (args.headless or args.cv_window or args.output or args.max_frames)
    if use_dashboard:
        try:
            return app.run_gui(mute=args.mute)
        except GuiUnavailable as exc:
            logger.warning("Dashboard indisponível: %s. Usando a janela OpenCV.", exc)
    return app.run(
        headless=args.headless,
        output=args.output,
        max_frames=args.max_frames,
        mute=args.mute,
    )


if __name__ == "__main__":
    sys.exit(main())
