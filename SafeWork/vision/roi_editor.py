"""Editor gráfico para desenhar as zonas de risco sobre um frame da câmera.

Controles:
    Botão esquerdo   adiciona um vértice
    Botão direito    remove o último vértice
    ENTER            fecha o polígono atual e cria a zona
    D                apaga a última zona
    C                limpa o desenho em andamento
    S                salva e sai
    ESC              sai sem salvar
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from safework.core.models import PPE, Point
from safework.vision.renderer import ascii_text
from safework.vision.roi import RiskZone, ZoneRepository

logger = logging.getLogger(__name__)

_DRAFT_COLOR = (0, 255, 255)
_ZONE_COLOR = (0, 165, 255)
_HELP = "Clique: vertice | Dir: desfaz | ENTER: fecha zona | D: apaga zona | S: salva | ESC: sai"


class ZoneEditorState:
    """Lógica do editor, separada da janela para poder ser testada."""

    def __init__(self, width: int, height: int, zones: list[RiskZone] | None = None) -> None:
        self.width = width
        self.height = height
        self.zones = list(zones or [])
        self.draft: list[Point] = []

    def add_vertex(self, x: int, y: int) -> None:
        # Limita ao frame: cliques na borda da janela podem vir fora da imagem.
        x = min(max(x, 0), self.width - 1)
        y = min(max(y, 0), self.height - 1)
        self.draft.append((x / (self.width - 1), y / (self.height - 1)))

    def undo_vertex(self) -> None:
        if self.draft:
            self.draft.pop()

    def clear_draft(self) -> None:
        self.draft.clear()

    def close_polygon(
        self,
        required_ppe: frozenset[PPE],
        name: str | None = None,
    ) -> RiskZone | None:
        """Cria a zona com o rascunho atual. Devolve None se houver menos de 3 vértices."""
        if len(self.draft) < 3:
            return None
        number = self._next_number()
        zone = RiskZone(
            id=f"zona-{number}",
            name=(name or "").strip() or f"Zona {number}",
            points=tuple(self.draft),
            required_ppe=required_ppe,
        )
        self.zones.append(zone)
        self.draft.clear()
        return zone

    def delete_last_zone(self) -> None:
        if self.zones:
            self.zones.pop()

    def _next_number(self) -> int:
        used = {zone.id for zone in self.zones}
        number = 1
        while f"zona-{number}" in used:
            number += 1
        return number


class RoiEditor:
    """Janela OpenCV que desenha as zonas e as grava no repositório."""

    WINDOW = "SafeWork - Editor de Zonas de Risco"

    def __init__(
        self,
        repository: ZoneRepository,
        required_ppe: frozenset[PPE] = frozenset({PPE.HELMET, PPE.VEST}),
    ) -> None:
        self._repository = repository
        self._required_ppe = required_ppe

    def run(self, background: np.ndarray) -> bool:
        """Abre o editor. Devolve True se as zonas foram salvas."""
        height, width = background.shape[:2]
        state = ZoneEditorState(width, height, self._repository.load())

        def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                state.add_vertex(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                state.undo_vertex()

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW, on_mouse)
        try:
            while True:
                cv2.imshow(self.WINDOW, self._draw(background, state))
                key = cv2.waitKey(20) & 0xFF

                if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return False  # janela fechada no X
                if key in (13, 10):
                    zone = state.close_polygon(self._required_ppe)
                    if zone is None:
                        logger.warning("São necessários ao menos 3 vértices.")
                    else:
                        logger.info("Zona criada: %s", zone.name)
                elif key in (ord("d"), ord("D")):
                    state.delete_last_zone()
                elif key in (ord("c"), ord("C")):
                    state.clear_draft()
                elif key in (ord("s"), ord("S")):
                    self._repository.save(state.zones)
                    logger.info("%d zona(s) salva(s).", len(state.zones))
                    return True
                elif key == 27:
                    return False
        finally:
            cv2.destroyWindow(self.WINDOW)

    @staticmethod
    def _draw(background: np.ndarray, state: ZoneEditorState) -> np.ndarray:
        canvas = background.copy()
        height, width = canvas.shape[:2]

        for zone in state.zones:
            polygon = zone.polygon_px(width, height)
            cv2.polylines(canvas, [polygon], True, _ZONE_COLOR, 2, cv2.LINE_AA)
            origin = tuple(int(value) for value in polygon[0])
            cv2.putText(canvas, ascii_text(zone.name), origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, _ZONE_COLOR, 2, cv2.LINE_AA)

        vertices = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in state.draft]
        if len(vertices) > 1:
            cv2.polylines(canvas, [np.array(vertices, dtype=np.int32)], False, _DRAFT_COLOR, 2,
                          cv2.LINE_AA)
        for vertex in vertices:
            cv2.circle(canvas, vertex, 5, _DRAFT_COLOR, -1, cv2.LINE_AA)

        cv2.rectangle(canvas, (0, height - 28), (width, height), (0, 0, 0), -1)
        cv2.putText(canvas, _HELP, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return canvas
