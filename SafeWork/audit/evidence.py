"""Frame Capture: grava o print de evidência em evidences/."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from safework.core.config import PROJECT_ROOT, EvidenceConfig, resolve_path


class EvidenceStore:
    """Salva o frame com overlay como JPEG e devolve o caminho e o SHA-256 do arquivo.

    Nome: evidencia_<CAMERA>_OP<ID>_<AAAAMMDD>_<HHMMSS>.jpg
    Um arquivo existente nunca é sobrescrito: em colisão, recebe o sufixo _2, _3...
    """

    def __init__(self, config: EvidenceConfig, camera_id: str) -> None:
        self._directory = resolve_path(config.directory)
        self._quality = config.jpeg_quality
        self._camera_tag = re.sub(r"[^A-Za-z0-9]", "", camera_id) or "CAM"

    @property
    def directory(self) -> Path:
        return self._directory

    def file_name(self, track_id: int, occurred_at: datetime, sequence: int = 1) -> str:
        suffix = f"_{sequence}" if sequence > 1 else ""
        return (
            f"evidencia_{self._camera_tag}_OP{track_id}_"
            f"{occurred_at:%Y%m%d_%H%M%S}{suffix}.jpg"
        )

    def save(self, image: np.ndarray, track_id: int, occurred_at: datetime) -> tuple[Path, str]:
        params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        # Sem subamostragem de cor (4:4:4): o padrão 4:2:0 borra bordas vermelhas e
        # texto, justamente o overlay que prova a infração. Custa ~20 KB por imagem.
        if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
            params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
        ok, buffer = cv2.imencode(".jpg", image, params)
        if not ok:
            raise ValueError("Falha ao codificar a evidência em JPEG.")
        data = buffer.tobytes()
        digest = hashlib.sha256(data).hexdigest()

        self._directory.mkdir(parents=True, exist_ok=True)
        sequence = 1
        while True:
            path = self._directory / self.file_name(track_id, occurred_at, sequence)
            try:
                with path.open("xb") as handle:  # "x": falha se o arquivo já existir
                    handle.write(data)
                return path, digest
            except FileExistsError:
                sequence += 1

    @staticmethod
    def relative(path: Path) -> str:
        """Caminho relativo à raiz do projeto (portável entre máquinas), se possível."""
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path)
