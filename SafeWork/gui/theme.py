"""Paleta e tipografia do dashboard.

Tema escuro de sala de controle: fundo neutro, e cor reservada para o que
exige atenção. Verde, âmbar e vermelho seguem o mesmo código dos overlays
do vídeo, então o painel e a imagem "falam a mesma língua".
"""

import sys

BACKGROUND = "#0f1216"
SURFACE = "#171b21"
CARD = "#1f252d"
BORDER = "#2b323c"
TEXT = "#e7eaee"
TEXT_MUTED = "#8a94a3"

OK = "#22a05a"
WARNING = "#e0a400"
DANGER = "#d93b3b"
NEUTRAL = "#4b5563"
INFO = "#2f7fd8"

# Fonte nativa de cada sistema: o Tk troca uma família inexistente por uma
# fonte bitmap genérica, que ignora tamanho e negrito.
if sys.platform == "win32":
    FONT_FAMILY, MONO_FAMILY = "Segoe UI", "Consolas"
elif sys.platform == "darwin":
    FONT_FAMILY, MONO_FAMILY = "Helvetica Neue", "Menlo"
else:
    FONT_FAMILY, MONO_FAMILY = "DejaVu Sans", "DejaVu Sans Mono"

SIDEBAR_WIDTH = 340
