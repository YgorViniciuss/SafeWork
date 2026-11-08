"""Etapa 2 - Pré-processamento: redimensionamento e redução de ruído."""

from __future__ import annotations

import cv2
import numpy as np

from safework.core.config import PreprocessingConfig


class FramePreprocessor:
    """Prepara o frame para inferência e exibição.

    A imagem permanece em BGR: o Ultralytics assume BGR ao receber um array
    NumPy e faz a conversão internamente. Converter para RGB aqui inverteria
    os canais e prejudicaria a detecção. A ROI é definida em roi.py.
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        self._config = config

    def process(self, image: np.ndarray) -> np.ndarray:
        result = self._resize(image)
        if self._config.denoise:
            kernel = (self._config.denoise_kernel, self._config.denoise_kernel)
            result = cv2.GaussianBlur(result, kernel, 0)
        return result

    def _resize(self, image: np.ndarray) -> np.ndarray:
        target_width = self._config.target_width
        height, width = image.shape[:2]
        if target_width == 0 or width == target_width:
            return image
        target_height = round(height * target_width / width)
        # INTER_AREA preserva melhor os detalhes ao reduzir a imagem.
        interpolation = cv2.INTER_AREA if target_width < width else cv2.INTER_LINEAR
        return cv2.resize(image, (target_width, target_height), interpolation=interpolation)
