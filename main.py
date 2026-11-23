# -*- coding: utf-8 -*-
"""
SafeWork - Sistema Inteligente de Identificacao e Mitigacao de Riscos de
Acidentes de Trabalho (TCC).

Arquivo unico e autonomo: execute com `python main.py` (ou via SafeWork.bat).
"""

from __future__ import annotations

import csv
import html
import io
import json
import logging
import math
import os
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import unicodedata
import wave
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import customtkinter as ctk

try:
    import yaml
except ImportError:
    yaml = None

try:
    import winsound
except ImportError:
    winsound = None


# ======================================================================
# CAMINHOS E CONSTANTES
# ======================================================================
BASE_DIR = Path(os.environ.get("SAFEWORK_HOME", Path(__file__).resolve().parent))
CONFIG_DIR = BASE_DIR / "config"
EVIDENCE_DIR = BASE_DIR / "evidences"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CUSTOM_EPI_IMG_DIR = DATA_DIR / "custom_epis"

SETTINGS_PATH = CONFIG_DIR / "settings.yaml"
ZONES_PATH = CONFIG_DIR / "zones.json"
CUSTOM_EPIS_PATH = CONFIG_DIR / "custom_epis.json"
CAMERA_RULES_PATH = CONFIG_DIR / "camera_rules.json"
DB_PATH = DATA_DIR / "safework_audit.db"
CSV_PATH = LOGS_DIR / "relatorio_auditoria.csv"
HTML_PATH = LOGS_DIR / "relatorio_auditoria.html"

APP_TITLE = "SafeWork - Identificação e Mitigação de Riscos de Acidentes de Trabalho"

WIN_W, WIN_H = 1366, 768
LEFT_W = 886
RIGHT_W = WIN_W - LEFT_W
VIDEO_W, VIDEO_H = 840, 520

STATE_REGULAR = "REGULAR"
STATE_INFRACAO = "INFRACAO"
STATE_NO_SIGNAL = "SEM_SINAL"

BASE_PPE_LABELS = {"ear": "Fone de concha", "glasses": "Óculos de proteção"}
BASE_PPE_SHORT = {"ear": "FONE", "glasses": "OCULOS"}

C = {
    "bg": "#0B192C",
    "panel": "#0F2038",
    "card": "#12304F",
    "border": "#1E4067",
    "cyan": "#00D8F6",
    "text": "#E6F1FF",
    "muted": "#8FA8C4",
    "green": "#22C5E5",
    "green_bg": "#0E3B2A",
    "red": "#EF4444",
    "red_bg": "#4A1414",
    "gray": "#64748B",
    "gray_bg": "#1C2B3F",
    "amber": "#F59E0B",
}
C["green"] = "#22C55E"

BGR_GREEN = (80, 200, 60)
BGR_RED = (68, 68, 239)
BGR_CYAN = (246, 216, 0)
BGR_AMBER = (11, 158, 245)
BGR_WHITE = (255, 255, 255)
BGR_BLACK = (0, 0, 0)

FONT_FAMILY = "Segoe UI" if os.name == "nt" else "Helvetica"

log = logging.getLogger("safework")


# ======================================================================
# DETECTOR FACIAL NATIVO (ANCORAGEM ANATOMICA REAL)
# ======================================================================
_FACE_CASCADE: Optional[cv2.CascadeClassifier] = None


def _get_face_cascade() -> Optional[cv2.CascadeClassifier]:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        try:
            xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            clf = cv2.CascadeClassifier(xml_path)
            if not clf.empty():
                _FACE_CASCADE = clf
        except Exception:
            pass
    return _FACE_CASCADE


def locate_face_in_person(frame: np.ndarray, person_box: tuple[float, float, float, float]) -> Optional[tuple[int, int, int, int]]:
    """Localiza o rosto real dentro da caixa do operador para ancorar olhos, boca e topo da cabeca."""
    clf = _get_face_cascade()
    if clf is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (max(0, int(person_box[0])), max(0, int(person_box[1])),
                      min(w, int(person_box[2])), min(h, int(person_box[3])))
    ph = y2 - y1
    pw = x2 - x1
    if pw < 30 or ph < 30:
        return None
    search_y2 = min(h, y1 + int(ph * 0.65))
    roi_gray = cv2.cvtColor(frame[y1:search_y2, x1:x2], cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=4, minSize=(36, 36))
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    return (x1 + int(fx), y1 + int(fy), x1 + int(fx + fw), y1 + int(fy + fh))


# ======================================================================
# LOGGING E CONFIGURACOES
# ======================================================================
def setup_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = RotatingFileHandler(
        LOGS_DIR / "safework.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(file_handler)
    root.addHandler(console)


DEFAULT_SETTINGS: dict[str, Any] = {
    "model": {
        "path": "models/safework_ppe.pt",
        "person_model_path": "yolov8n.pt",
        "imgsz": 640,
        "device": None,
    },
    "detection": {
        "conf_person": 0.25,
        "conf_ear": 0.08,
        "conf_glasses": 0.10,
        "iou": 0.45,
        "required_ppe": ["ear", "glasses"],
        "window": 6,
        "needed": 4,
        "class_map": {
            "helmet": "ear",
            "glasses": "glasses",
            "person": "person",
            "no-helmet": "no_ear",
            "no-glasses": "no_glasses",
        },
    },
    "alert": {
        "infraction_seconds": 3.0,
        "cooldown_seconds": 15.0,
        "beep_freq": 1200,
        "beep_ms": 350,
        "beep_count": 3,
    },
    "capture": {"width": 1280, "height": 720},
    "cameras": [
        {"name": "Câmera 1", "source": 0},
        {"name": "Câmera 2", "source": 1},
    ],
    "security": {"manager_password": "admin123"},
}


def deep_merge(base: dict, override: dict) -> dict:
    out: dict[str, Any] = {}
    for key, default in base.items():
        if key not in override:
            out[key] = default
            continue
        value = override[key]
        if isinstance(default, dict):
            if isinstance(value, dict):
                out[key] = deep_merge(default, value)
            else:
                out[key] = default
        elif isinstance(default, list):
            if isinstance(value, list) and value:
                out[key] = value
            else:
                out[key] = default
        else:
            out[key] = default if value is None and default is not None else value
    return out


def load_settings() -> dict[str, Any]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_PATH.exists():
        if yaml is not None:
            try:
                with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
                    yaml.safe_dump(DEFAULT_SETTINGS, fh, allow_unicode=True, sort_keys=False)
            except OSError:
                log.exception("Nao foi possivel criar settings.yaml.")
        return json.loads(json.dumps(DEFAULT_SETTINGS))
    if yaml is None:
        return json.loads(json.dumps(DEFAULT_SETTINGS))
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        if not isinstance(user, dict):
            raise ValueError("raiz do YAML nao e um dicionario")
        merged = deep_merge(DEFAULT_SETTINGS, user)
        default_map = DEFAULT_SETTINGS["detection"]["class_map"]
        user_map = merged["detection"].get("class_map") or {}
        merged["detection"]["class_map"] = {**default_map, **user_map}
        merged["detection"]["conf_ear"] = min(float(merged["detection"].get("conf_ear", 0.08)), 0.10)
        merged["detection"]["conf_glasses"] = min(float(merged["detection"].get("conf_glasses", 0.10)), 0.12)
        return merged
    except Exception:
        log.exception("settings.yaml invalido; usando valores padrao.")
        return json.loads(json.dumps(DEFAULT_SETTINGS))


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else BASE_DIR / p


def open_path(path: Path | str) -> None:
    target = str(path)
    if sys.platform.startswith("win"):
        os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])


# ======================================================================
# SIRENE PROGRESSIVA INTELIGENTE (PARA IMEDIATAMENTE AO REGULARIZAR)
# ======================================================================
def _build_siren_wav(level: int) -> bytes:
    sample_rate = 22050
    duration = 0.55
    n_samples = int(sample_rate * duration)
    amp = min(1.0, 0.35 + level * 0.12) * 32767.0
    f_start = 880.0 + min(level, 6) * 110.0
    f_end = f_start + 520.0

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            t = i / sample_rate
            progress = i / n_samples
            freq = f_start + (f_end - f_start) * progress
            val = 0.75 * math.sin(2.0 * math.pi * freq * t) + 0.25 * math.sin(4.0 * math.pi * freq * t)
            sample = int(max(-32767, min(32767, val * amp)))
            frames.extend(struct.pack("<h", sample))
        wf.writeframes(bytes(frames))
    return buf.getvalue()


class ProgressiveSiren:
    def __init__(self) -> None:
        self._active = threading.Event()
        self._halt = threading.Event()
        self._start_time = 0.0
        self._lock = threading.Lock()
        self._wav_cache: dict[int, bytes] = {lvl: _build_siren_wav(lvl) for lvl in range(7)}
        self._thread = threading.Thread(target=self._loop, daemon=True, name="progressive-siren")
        self._thread.start()

    def start_alarm(self) -> None:
        with self._lock:
            if not self._active.is_set():
                self._start_time = time.monotonic()
                self._active.set()

    def stop_alarm(self) -> None:
        with self._lock:
            if self._active.is_set():
                self._active.clear()
                if winsound is not None:
                    try:
                        winsound.PlaySound(None, winsound.SND_PURGE)
                    except Exception:
                        pass

    def shutdown(self) -> None:
        self.stop_alarm()
        self._halt.set()

    def _loop(self) -> None:
        while not self._halt.is_set():
            if not self._active.is_set():
                time.sleep(0.05)
                continue

            with self._lock:
                if not self._active.is_set():
                    continue
                elapsed = max(0.0, time.monotonic() - self._start_time)

            level = min(6, int(elapsed // 1.2))
            if winsound is not None:
                try:
                    wav_bytes = self._wav_cache[level]
                    winsound.PlaySound(wav_bytes, winsound.SND_MEMORY | winsound.SND_ASYNC)
                    for _ in range(11):
                        if not self._active.is_set() or self._halt.is_set():
                            winsound.PlaySound(None, winsound.SND_PURGE)
                            break
                        time.sleep(0.05)
                except Exception:
                    log.exception("Erro ao reproduzir sirene WAV.")
                    time.sleep(0.2)
            else:
                print("\a", end="", flush=True)
                time.sleep(0.4)


# ======================================================================
# GERENCIADOR DE EPIS CUSTOMIZADOS E ANATOMIA DE DETECCAO
# ======================================================================
def _imread_unicode(path: Path | str) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _skin_weight_mask(bgr: np.ndarray) -> np.ndarray:
    """Atenua pixels de tom de pele humana para que a assinatura foque no material do EPI."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    return np.where(skin > 0, 35, 255).astype(np.uint8)


def extract_visual_signature(crop_bgr: np.ndarray) -> dict[str, Any]:
    """Extrai assinatura focada no material do EPI (atenuando pele) + densidade estrutural e gradiente."""
    if crop_bgr is None or crop_bgr.size == 0:
        return {}
    resized = cv2.resize(crop_bgr, (96, 96), interpolation=cv2.INTER_AREA)
    non_skin_mask = _skin_weight_mask(resized)

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist_hsv = cv2.calcHist([hsv], [0, 1, 2], non_skin_mask, [12, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist_hsv, hist_hsv, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    hist_lab = cv2.calcHist([lab], [0, 1, 2], non_skin_mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_lab, hist_lab, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 130)
    edge_density = float(cv2.countNonZero(edges)) / max(1, edges.size)

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_energy = float(np.mean(cv2.magnitude(sobelx, sobely)))

    struct_small = cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    return {
        "hist_hsv": hist_hsv.flatten().tolist(),
        "hist_lab": hist_lab.flatten().tolist(),
        "edge_density": round(edge_density, 4),
        "grad_energy": round(grad_energy, 2),
        "struct_map": struct_small.flatten().tolist(),
    }


def _compare_roi_with_sig(roi: np.ndarray, sig: dict[str, Any]) -> float:
    non_skin_mask = _skin_weight_mask(roi)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hsv = cv2.calcHist([hsv], [0, 1, 2], non_skin_mask, [12, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(roi_hsv, roi_hsv, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    roi_lab = cv2.calcHist([lab], [0, 1, 2], non_skin_mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(roi_lab, roi_lab, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 130)
    roi_edges = float(cv2.countNonZero(edges)) / max(1, edges.size)

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    roi_grad = float(np.mean(cv2.magnitude(sobelx, sobely)))

    ref_hsv = np.array(sig["hist_hsv"], dtype=np.float32).reshape((12, 8, 8))
    ref_lab = np.array(sig["hist_lab"], dtype=np.float32).reshape((8, 8, 8))
    ref_edge = float(sig.get("edge_density", roi_edges))
    ref_grad = float(sig.get("grad_energy", roi_grad))

    sim_hsv = max(0.0, float(cv2.compareHist(roi_hsv, ref_hsv, cv2.HISTCMP_CORREL)))
    sim_lab = max(0.0, float(cv2.compareHist(roi_lab, ref_lab, cv2.HISTCMP_CORREL)))
    sim_edge = max(0.0, 1.0 - abs(roi_edges - ref_edge) * 3.5)
    sim_grad = max(0.0, 1.0 - abs(roi_grad - ref_grad) / max(25.0, ref_grad))

    sim_struct = 0.5
    if "struct_map" in sig:
        roi_struct = cv2.resize(edges, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).flatten() / 255.0
        ref_struct = np.array(sig["struct_map"], dtype=np.float32)
        denom = float(np.linalg.norm(roi_struct) * np.linalg.norm(ref_struct))
        if denom > 1e-5:
            sim_struct = max(0.0, float(np.dot(roi_struct, ref_struct) / denom))

    return sim_lab * 0.25 + sim_hsv * 0.20 + sim_edge * 0.20 + sim_grad * 0.15 + sim_struct * 0.20


def load_custom_epis() -> list[dict[str, Any]]:
    if not CUSTOM_EPIS_PATH.exists():
        return []
    try:
        with open(CUSTOM_EPIS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("epis", []) if isinstance(data, dict) else []
    except Exception:
        log.exception("Erro ao ler custom_epis.json")
        return []


def save_custom_epis(epis: list[dict[str, Any]]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CUSTOM_EPIS_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"version": 4, "epis": epis}, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, CUSTOM_EPIS_PATH)


def load_camera_rules() -> dict[str, list[str]]:
    if not CAMERA_RULES_PATH.exists():
        return {}
    try:
        with open(CAMERA_RULES_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        rules = data.get("rules", {}) if isinstance(data, dict) else {}
        return {str(k): list(v) for k, v in rules.items() if isinstance(v, list)}
    except Exception:
        log.exception("Erro ao ler camera_rules.json")
        return {}


def save_camera_rules(rules: dict[str, list[str]]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CAMERA_RULES_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "rules": rules}, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, CAMERA_RULES_PATH)


def get_roi_for_region(
    frame: np.ndarray,
    person_box: tuple[float, float, float, float],
    region: str,
    face_box: Optional[tuple[int, int, int, int]] = None,
) -> tuple[int, int, int, int]:
    """Segmentacao anatomica ancorada no rosto real quando detectado, com fallback proporcional."""
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = person_box
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)

    if face_box is not None:
        fx1, fy1, fx2, fy2 = face_box
        fw = max(1.0, float(fx2 - fx1))
        fh = max(1.0, float(fy2 - fy1))
        if region == "eyes":
            rx1, ry1 = fx1 + fw * 0.08, fy1 + fh * 0.22
            rx2, ry2 = fx2 - fw * 0.08, fy1 + fh * 0.50
        elif region == "mouth":
            rx1, ry1 = fx1 + fw * 0.10, fy1 + fh * 0.48
            rx2, ry2 = fx2 - fw * 0.10, fy2 + fh * 0.08
        elif region == "head":
            rx1, ry1 = max(x1, fx1 - fw * 0.28), max(y1, fy1 - fh * 0.65)
            rx2, ry2 = min(x2, fx2 + fw * 0.28), fy1 + fh * 0.35
        else:
            rx1, ry1 = x1 + pw * 0.10, min(y2 - 10, fy2 + fh * 0.15)
            rx2, ry2 = x2 - pw * 0.10, y2
    else:
        aspect = ph / pw
        is_closeup = aspect < 1.75
        if region == "eyes":
            rx1, ry1 = x1 + pw * 0.24, y1 + ph * (0.22 if is_closeup else 0.14)
            rx2, ry2 = x2 - pw * 0.24, y1 + ph * (0.38 if is_closeup else 0.28)
        elif region == "mouth":
            rx1, ry1 = x1 + pw * 0.24, y1 + ph * (0.34 if is_closeup else 0.22)
            rx2, ry2 = x2 - pw * 0.24, y1 + ph * (0.62 if is_closeup else 0.42)
        elif region == "head":
            rx1, ry1 = x1 + pw * 0.12, y1
            rx2, ry2 = x2 - pw * 0.12, y1 + ph * (0.32 if is_closeup else 0.24)
        else:
            rx1, ry1 = x1 + pw * 0.10, y1 + ph * 0.45
            rx2, ry2 = x2 - pw * 0.10, y2

    return (
        max(0, min(frame_w - 1, int(rx1))),
        max(0, min(frame_h - 1, int(ry1))),
        max(1, min(frame_w, int(rx2))),
        max(1, min(frame_h, int(ry2))),
    )


def match_custom_epi(
    frame: np.ndarray,
    person_box: tuple[float, float, float, float],
    epi: dict[str, Any],
    face_box: Optional[tuple[int, int, int, int]] = None,
) -> Optional[tuple[float, tuple[float, float, float, float]]]:
    signatures = epi.get("signatures") or []
    if not signatures:
        return None
    rx1, ry1, rx2, ry2 = get_roi_for_region(frame, person_box, epi.get("region", "mouth"), face_box)
    if rx2 - rx1 < 12 or ry2 - ry1 < 12:
        return None
    roi = cv2.resize(frame[ry1:ry2, rx1:rx2], (96, 96), interpolation=cv2.INTER_AREA)

    pos_scores = [_compare_roi_with_sig(roi, s) for s in signatures if isinstance(s, dict) and "hist_lab" in s]
    if not pos_scores:
        return None
    best_pos = max(pos_scores)

    neg_sigs = epi.get("negative_signatures") or []
    neg_scores = [_compare_roi_with_sig(roi, s) for s in neg_sigs if isinstance(s, dict) and "hist_lab" in s]
    if neg_scores:
        best_neg = max(neg_scores)
        if best_pos <= best_neg + 0.04:
            return None

    threshold = max(0.62, float(epi.get("threshold", 0.64)))
    if best_pos >= threshold:
        return min(0.98, best_pos), (float(rx1), float(ry1), float(rx2), float(ry2))
    return None


def detect_red_earmuffs_visual(frame: np.ndarray, person_box: tuple[float, float, float, float]) -> Optional[tuple[float, tuple[float, float, float, float]]]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = person_box
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)

    hy1 = max(0, int(y1 + ph * 0.04))
    hy2 = min(h, int(y1 + ph * 0.56))
    hx1 = max(0, int(x1))
    hx2 = min(w, int(x2))
    if hy2 - hy1 < 18 or hx2 - hx1 < 18:
        return None

    head_roi = frame[hy1:hy2, hx1:hx2]
    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([0, 75, 40]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 75, 40]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    rw = hx2 - hx1
    mask[:, int(rw * 0.34):int(rw * 0.66)] = 0

    red_pixels = int(cv2.countNonZero(mask))
    side_area = max(1, (hy2 - hy1) * int(rw * 0.68))
    ratio = red_pixels / side_area

    if ratio >= 0.032:
        ys, xs = np.where(mask > 0)
        if xs.size > 0:
            bx1 = float(hx1 + int(xs.min()))
            by1 = float(hy1 + int(ys.min()))
            bx2 = float(hx1 + int(xs.max()))
            by2 = float(hy1 + int(ys.max()))
            conf = min(0.95, 0.58 + ratio * 2.8)
            return conf, (bx1, by1, bx2, by2)
    return None


# ======================================================================
# MODELO E MAPEAMENTO DE CLASSES
# ======================================================================
def _norm(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower().strip()


ROLE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("person", ("person", "pessoa", "operador", "operator", "worker", "trabalhador", "humano", "human")),
    ("glasses", ("glass", "oculos", "goggle", "eyewear", "eye", "visor", "ocular", "lente")),
    ("ear", ("ear", "fone", "muff", "headphone", "hearing", "abafador", "auricular", "concha", "orelha", "helmet")),
]


def build_class_roles(names: dict[int, str], override: dict[str, str]) -> dict[int, str]:
    forced = {_norm(k): _norm(v) for k, v in (override or {}).items()}
    roles: dict[int, str] = {}
    for cid, name in names.items():
        key = _norm(name)
        if key in forced:
            role = forced[key]
        else:
            role = ""
            if key in ("no-helmet", "no-ear", "sem-fone"):
                role = "no_ear"
            elif key in ("no-glasses", "sem-oculos"):
                role = "no_glasses"
            elif not key.startswith("no-") and not key.startswith("sem-"):
                for candidate, words in ROLE_KEYWORDS:
                    if any(w in key for w in words):
                        role = candidate
                        break
        if role in ("person", "ear", "glasses", "no_ear", "no_glasses"):
            roles[int(cid)] = role
    return roles


def load_models(settings: dict[str, Any]):
    from ultralytics import YOLO

    ppe_path = resolve_path(settings["model"]["path"])
    if not ppe_path.exists():
        raise FileNotFoundError(f"Modelo de EPIs nao encontrado: {ppe_path}")
    ppe_model = YOLO(str(ppe_path))
    names = dict(ppe_model.names)
    det_cfg = settings["detection"]
    roles = build_class_roles(names, det_cfg.get("class_map") or {})
    log.info("Classes do modelo EPI: %s -> papeis: %s", names, roles)

    person_model = None
    person_cfg_path = settings["model"].get("person_model_path", "yolov8n.pt")
    if person_cfg_path:
        try:
            p_local = resolve_path(person_cfg_path)
            target = str(p_local) if p_local.exists() else str(person_cfg_path)
            person_model = YOLO(target)
            log.info("Detector auxiliar de operadores carregado: %s", target)
        except Exception as exc:
            log.warning("Detector auxiliar indisponivel (%s).", exc)

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    kwargs = _predict_kwargs(settings)
    ppe_model.predict(dummy, conf=0.5, verbose=False, **kwargs)
    if person_model is not None:
        person_model.predict(dummy, conf=0.5, classes=[0], verbose=False, **kwargs)

    return ppe_model, person_model, roles


def _predict_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"imgsz": int(settings["model"].get("imgsz") or 640)}
    if settings["model"].get("device") not in (None, ""):
        kwargs["device"] = settings["model"]["device"]
    return kwargs


# ======================================================================
# GEOMETRIA / REGRAS DE DECISAO
# ======================================================================
@dataclass
class Detection:
    role: str
    conf: float
    box: tuple[float, float, float, float]
    label_override: str = ""


@dataclass
class FrameEval:
    persons: list[Detection] = field(default_factory=list)
    outside: list[Detection] = field(default_factory=list)
    attached_ppe: list[Detection] = field(default_factory=list)
    missing: set[str] = field(default_factory=set)
    raw_state: str = STATE_REGULAR


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = _area(box_a) + _area(box_b) - inter
    return inter / union if union > 0 else 0.0


def _point_in_box(px: float, py: float, box: tuple[float, float, float, float], margin: float = 0.12) -> bool:
    mx = (box[2] - box[0]) * margin
    my = (box[3] - box[1]) * margin
    return (box[0] - mx) <= px <= (box[2] + mx) and (box[1] - my) <= py <= (box[3] + my)


def _in_polygon(poly: np.ndarray, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def box_in_zone(box: tuple[float, float, float, float], zone_polys: list[np.ndarray]) -> bool:
    if not zone_polys:
        return True
    cx, cy = _center(box)
    foot = ((box[0] + box[2]) / 2.0, box[3])
    head = ((box[0] + box[2]) / 2.0, box[1] + (box[3] - box[1]) * 0.25)
    pts = [(cx, cy), foot, head]
    return any(_in_polygon(p, px, py) for p in zone_polys for (px, py) in pts)


def evaluate_frame(dets: list[Detection], zone_polys: list[np.ndarray], required: list[str]) -> FrameEval:
    ev = FrameEval()
    req_set = set(required)
    persons = [d for d in dets if d.role == "person"]
    ppes = [d for d in dets if d.role not in ("person", "no_ear", "no_glasses") and d.role in req_set]
    neg_ppes = [d for d in dets if (d.role == "no_ear" and "ear" in req_set) or (d.role == "no_glasses" and "glasses" in req_set)]

    for p in persons:
        (ev.persons if box_in_zone(p.box, zone_polys) else ev.outside).append(p)

    if not ev.persons and neg_ppes:
        for neg in neg_ppes:
            if box_in_zone(neg.box, zone_polys):
                ev.persons.append(Detection("person", neg.conf, neg.box))
                break

    present: dict[int, set[str]] = {id(p): set() for p in ev.persons}
    for item in ppes:
        cx, cy = _center(item.box)
        holders = [p for p in ev.persons if _point_in_box(cx, cy, p.box)]
        if holders:
            holder = min(holders, key=lambda p: _area(p.box))
            if item.role == "glasses":
                ph = max(1.0, holder.box[3] - holder.box[1])
                rel_y = (cy - holder.box[1]) / ph
                if rel_y < 0.11:
                    continue
            present[id(holder)].add(item.role)
            ev.attached_ppe.append(item)

    for p in ev.persons:
        ev.missing |= req_set - present[id(p)]

    ev.raw_state = STATE_INFRACAO if ev.missing else STATE_REGULAR
    return ev


class TemporalFilter:
    def __init__(self, window: int = 6, needed: int = 4, initial: str = STATE_REGULAR) -> None:
        self.window = window
        self.needed = needed
        self.state = initial
        self.history: deque[str] = deque(maxlen=window)

    def reset(self, initial: str = STATE_REGULAR) -> None:
        self.history.clear()
        self.state = initial

    def update(self, raw_state: str) -> tuple[str, bool]:
        self.history.append(raw_state)
        previous = self.state
        for candidate in (STATE_INFRACAO, STATE_REGULAR):
            if self.history.count(candidate) >= self.needed:
                self.state = candidate
                break
        return self.state, self.state != previous


# ======================================================================
# DESENHO
# ======================================================================
def letterbox_geometry(w: int, h: int, cw: int, ch: int) -> tuple[float, int, int, int, int]:
    scale = min(cw / w, ch / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return scale, nw, nh, (cw - nw) // 2, (ch - nh) // 2


def letterbox(frame: np.ndarray, cw: int, ch: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale, nw, nh, ox, oy = letterbox_geometry(w, h, cw, ch)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (nw, nh), interpolation=interp)
    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float, thick: int) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    y = max(y, th + base + 2)
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 6, y + 2), color, -1)
    cv2.putText(img, text, (x + 3, y - base // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, BGR_BLACK, thick, cv2.LINE_AA)


def zones_to_pixels(zones_norm: list[list[list[float]]], w: int, h: int) -> list[np.ndarray]:
    polys = []
    for poly in zones_norm:
        if len(poly) >= 3:
            pts = np.array([[int(round(px * w)), int(round(py * h))] for px, py in poly], dtype=np.int32)
            polys.append(pts)
    return polys


def draw_scene(frame: np.ndarray, ev: FrameEval, state: str, zone_polys: list[np.ndarray]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    thick = max(2, w // 420)
    fscale = max(0.5, w / 1500.0)

    if zone_polys:
        overlay = out.copy()
        for poly in zone_polys:
            cv2.fillPoly(overlay, [poly], BGR_AMBER)
        out = cv2.addWeighted(overlay, 0.14, out, 0.86, 0)
        for i, poly in enumerate(zone_polys, start=1):
            cv2.polylines(out, [poly], True, BGR_AMBER, thick, cv2.LINE_AA)
            x, y = int(poly[:, 0].min()), int(poly[:, 1].min())
            _label(out, f"ZONA DE RISCO {i}", (x, y - 4), BGR_AMBER, fscale * 0.8, 1)

    person_color = BGR_RED if state == STATE_INFRACAO else BGR_GREEN
    person_text = "OPERADOR - INFRACAO" if state == STATE_INFRACAO else "OPERADOR - CONFORME"
    for p in ev.persons:
        x1, y1, x2, y2 = (int(v) for v in p.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), person_color, thick + 1, cv2.LINE_AA)
        _label(out, person_text, (x1, y1 - 4), person_color, fscale, 1)

    for item in ev.attached_ppe:
        x1, y1, x2, y2 = (int(v) for v in item.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), BGR_CYAN, max(1, thick - 1), cv2.LINE_AA)
        tag = item.label_override or BASE_PPE_SHORT.get(item.role, item.role.upper())
        _label(out, f"{tag} {item.conf:.2f}", (x1, y2 + int(18 * fscale) + 6), BGR_CYAN, max(0.5, fscale * 0.8), 1)
    return out


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_evidence(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    size = max(13, min(img.height // 34, img.width // 52))
    font = _load_font(size)
    pad = 10
    line_h = size + 6
    banner_h = pad * 2 + line_h * len(lines)
    top = img.height - banner_h
    draw.rectangle([0, top, img.width, img.height], fill=(11, 25, 44, 225))
    for i, text in enumerate(lines):
        color = (239, 68, 68, 255) if i == 0 else (230, 241, 255, 255)
        draw.text((pad, top + pad + i * line_h), text, font=font, fill=color)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


# ======================================================================
# ZONAS (config/zones.json)
# ======================================================================
def load_zones() -> dict[str, list[list[list[float]]]]:
    if not ZONES_PATH.exists():
        return {}
    try:
        with open(ZONES_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, list):
            zones = {"0": raw}
        elif isinstance(raw, dict):
            zones = raw.get("zones", {})
            if isinstance(zones, list):
                zones = {"0": zones}
        else:
            zones = {}
        clean: dict[str, list[list[list[float]]]] = {}
        for cam_key, polys in zones.items():
            good = []
            for poly in polys:
                if isinstance(poly, list):
                    pts = [
                        [min(1.0, max(0.0, float(pt[0]))), min(1.0, max(0.0, float(pt[1])))]
                        for pt in poly
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2
                    ]
                    if len(pts) >= 3:
                        good.append(pts)
            clean[str(cam_key)] = good
        return clean
    except Exception:
        log.exception("zones.json invalido; ignorando.")
        return {}


def save_zones(all_zones: dict[str, list[list[list[float]]]]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "zones": all_zones}
    tmp = ZONES_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, ZONES_PATH)


# ======================================================================
# AUDITORIA: SQLITE + CSV + HTML
# ======================================================================
CSV_HEADER = ["ID", "Data_Hora", "Camera", "Perfil_Ativo", "Tipo_Risco", "Caminho_Evidencia_PNG"]


class AuditStore:
    def __init__(self, db_path: Path = DB_PATH, csv_path: Path = CSV_PATH, html_path: Path = HTML_PATH) -> None:
        self.db_path = db_path
        self.csv_path = csv_path
        self.html_path = html_path
        self._lock = threading.Lock()
        self._csv_dirty = False
        for folder in (db_path.parent, csv_path.parent, EVIDENCE_DIR):
            folder.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=10)

    def _init_db(self) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                """CREATE TABLE IF NOT EXISTS auditoria (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       data_hora TEXT NOT NULL,
                       camera TEXT NOT NULL,
                       perfil_ativo TEXT NOT NULL,
                       tipo_risco TEXT NOT NULL,
                       caminho_evidencia TEXT NOT NULL
                   )"""
            )

    def register(self, when: datetime, camera: str, profile: str, risk: str, evidence_rel: str) -> int:
        stamp = when.strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            with self._connect() as con:
                cur = con.execute(
                    "INSERT INTO auditoria (data_hora, camera, perfil_ativo, tipo_risco, caminho_evidencia) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (stamp, camera, profile, risk, evidence_rel),
                )
                row_id = int(cur.lastrowid)
            self._append_csv([row_id, stamp, camera, profile, risk, evidence_rel])
        return row_id

    def _append_csv(self, row: list[Any]) -> None:
        try:
            new_file = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as fh:
                writer = csv.writer(fh, delimiter=";")
                if new_file:
                    writer.writerow(CSV_HEADER)
                writer.writerow(row)
        except PermissionError:
            self._csv_dirty = True
        except OSError:
            self._csv_dirty = True

    def fetch_all(self) -> list[tuple]:
        with self._lock, self._connect() as con:
            return con.execute(
                "SELECT id, data_hora, camera, perfil_ativo, tipo_risco, caminho_evidencia "
                "FROM auditoria ORDER BY id DESC"
            ).fetchall()

    def export_csv(self) -> Path:
        rows = list(reversed(self.fetch_all()))
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh, delimiter=";")
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)
        self._csv_dirty = False
        return self.csv_path

    def export_html(self) -> Path:
        rows = self.fetch_all()
        by_cam = Counter(r[2] for r in rows)
        body = []
        for rid, when, cam, profile, risk, rel in rows:
            img_path = (BASE_DIR / rel)
            uri = img_path.as_uri() if img_path.exists() else ""
            thumb = (
                f'<a href="{html.escape(uri)}" target="_blank"><img src="{html.escape(uri)}" alt="evidência {rid}"></a>'
                if uri else "<em>arquivo não encontrado</em>"
            )
            body.append(
                "<tr>"
                f"<td>{rid}</td><td>{html.escape(when)}</td><td>{html.escape(cam)}</td>"
                f"<td>{html.escape(profile)}</td><td>{html.escape(risk)}</td>"
                f"<td class='path'>{html.escape(rel)}</td><td>{thumb}</td>"
                "</tr>"
            )
        resumo = " &nbsp;|&nbsp; ".join(f"{html.escape(c)}: <b>{n}</b>" for c, n in sorted(by_cam.items())) or "—"
        doc = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>SafeWork - Relatório de Auditoria</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;background:#0B192C;color:#E6F1FF;margin:0;padding:32px}}
 h1{{color:#00D8F6;margin:0 0 4px}} .sub{{color:#8FA8C4;margin-bottom:20px}}
 .kpi{{display:flex;gap:16px;margin-bottom:22px}}
 .kpi div{{background:#12304F;border:1px solid #1E4067;border-radius:10px;padding:14px 20px}}
 .kpi b{{font-size:26px;color:#EF4444;display:block}}
 table{{border-collapse:collapse;width:100%;background:#0F2038}}
 th,td{{border:1px solid #1E4067;padding:8px 10px;text-align:left;vertical-align:middle}}
 th{{background:#12304F;color:#00D8F6}} td.path{{font-size:12px;color:#8FA8C4}}
 img{{width:180px;border-radius:6px;border:1px solid #1E4067}}
</style></head><body>
<h1>SAFEWORK — Relatório de Auditoria</h1>
<div class="sub">Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} • Fonte: data/safework_audit.db</div>
<div class="kpi"><div><b>{len(rows)}</b>infrações registradas</div><div><span>Por câmera</span><br>{resumo}</div></div>
<table><thead><tr><th>ID</th><th>Data/Hora</th><th>Câmera</th><th>Perfil ativo</th><th>Tipo de risco</th><th>Evidência (arquivo)</th><th>Miniatura</th></tr></thead>
<tbody>{''.join(body) if body else '<tr><td colspan="7">Nenhuma infração registrada até o momento.</td></tr>'}</tbody></table>
</body></html>"""
        self.html_path.parent.mkdir(parents=True, exist_ok=True)
        self.html_path.write_text(doc, encoding="utf-8")
        return self.html_path

    def export_all(self) -> Path:
        self.export_csv()
        return self.export_html()


# ======================================================================
# CAPTURA DE CAMERA
# ======================================================================
class CameraSource(threading.Thread):
    def __init__(self, cameras: list[dict[str, Any]], capture_cfg: dict[str, Any], start_index: int = 0) -> None:
        super().__init__(daemon=True, name="camera")
        self.cameras = cameras
        self.cfg = capture_cfg
        self.active_index = start_index
        self.generation = 0
        self.signal_ok = False
        self.message = ""
        self._requested: Optional[int] = None
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self._frame: Optional[np.ndarray] = None
        self._fid = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_file = False
        self._file_delay = 0.0
        self._last_ok = 0.0
        self._last_try = 0.0

    def request_switch(self, index: int) -> None:
        with self._lock:
            self._requested = index

    def get_latest(self) -> tuple[Optional[np.ndarray], int]:
        with self._lock:
            return self._frame, self._fid

    def stop(self) -> None:
        self._halt.set()

    def _open_source(self, index: int) -> tuple[Optional[cv2.VideoCapture], bool, float]:
        src = self.cameras[index]["source"]
        cap: Optional[cv2.VideoCapture] = None
        is_file, delay = False, 0.0
        as_int: Optional[int] = None
        if isinstance(src, int):
            as_int = src
        elif isinstance(src, str) and src.strip().isdigit():
            as_int = int(src.strip())

        if as_int is not None:
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
            for backend in backends:
                cap = cv2.VideoCapture(as_int, backend)
                if cap.isOpened():
                    break
                cap.release()
                cap = None
            if cap is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg.get("width", 1280)))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.get("height", 720)))
        else:
            raw_str = str(src).strip()
            path = resolve_path(raw_str)
            is_file = path.exists()
            target = str(path) if is_file else raw_str

            if target.lower().startswith("rtsp://"):
                for transport in ("tcp", "udp"):
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}|stimeout;5000000"
                    cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
                    if cap.isOpened():
                        ok, _ = cap.read()
                        if ok:
                            break
                    if cap is not None:
                        cap.release()
                        cap = None
            else:
                cap = cv2.VideoCapture(target)
                if is_file and cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    delay = 1.0 / max(1.0, fps)

        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return None, False, 0.0

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return None, False, 0.0
        if is_file:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap, is_file, delay

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _activate(self, index: int) -> bool:
        cap, is_file, delay = self._open_source(index)
        if cap is None:
            return False
        self._release()
        self._cap, self._is_file, self._file_delay = cap, is_file, delay
        with self._lock:
            self._frame = None
        changed = index != self.active_index
        self.active_index = index
        if changed:
            self.generation += 1
        self._last_ok = time.monotonic()
        self.signal_ok = True
        return True

    def run(self) -> None:
        if not self._activate(self.active_index):
            self.message = f"Sem sinal em {self.cameras[self.active_index]['name']}."
            self._last_try = time.monotonic()
        while not self._halt.is_set():
            try:
                with self._lock:
                    req, self._requested = self._requested, None
                if req is not None and 0 <= req < len(self.cameras):
                    name = self.cameras[req]["name"]
                    if self._activate(req):
                        self.message = ""
                    else:
                        self.message = f"Falha ao abrir {name}. Verifique a URL/rede no settings.yaml."
                    continue

                if self._cap is None:
                    if time.monotonic() - self._last_try >= 2.0:
                        self._last_try = time.monotonic()
                        if self._activate(self.active_index):
                            self.message = ""
                    time.sleep(0.05)
                    continue

                ok, frame = self._cap.read()
                if ok and frame is not None:
                    with self._lock:
                        self._frame = frame
                        self._fid += 1
                    self._last_ok = time.monotonic()
                    self.signal_ok = True
                    if self._file_delay:
                        time.sleep(self._file_delay)
                elif self._is_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                else:
                    if time.monotonic() - self._last_ok > 3.0:
                        self.signal_ok = False
                        self._release()
                        self._last_try = time.monotonic()
                        self.message = f"Sem sinal em {self.cameras[self.active_index]['name']}."
                    time.sleep(0.01)
            except Exception:
                log.exception("Erro na captura.")
                time.sleep(0.2)
        self._release()


# ======================================================================
# MOTOR DE DETECCAO E REGRAS (thread)
# ======================================================================
class RuntimeContext:
    def __init__(self, camera_name: str, camera_key: str, profile: str = "Operador") -> None:
        self.camera_name = camera_name
        self.camera_key = camera_key
        self.profile = profile


@dataclass
class Snapshot:
    state: str = STATE_NO_SIGNAL
    missing: tuple[str, ...] = ()
    operators: int = 0
    fps: float = 0.0
    elapsed: float = 0.0
    registered: bool = False
    zones: int = 0
    active_epis_text: str = ""


class SafetyEngine(threading.Thread):
    def __init__(self, ppe_model, person_model, roles: dict[int, str], settings: dict[str, Any],
                 cam: CameraSource, store: AuditStore, ctx: RuntimeContext, siren: ProgressiveSiren) -> None:
        super().__init__(daemon=True, name="engine")
        det = settings["detection"]
        alert = settings["alert"]
        self.ppe_model = ppe_model
        self.person_model = person_model
        self.roles = roles
        self.cam = cam
        self.store = store
        self.ctx = ctx
        self.siren = siren
        self.base_required: list[str] = list(det["required_ppe"])
        self.thresholds = {
            "person": float(det["conf_person"]),
            "ear": float(det["conf_ear"]),
            "glasses": float(det["conf_glasses"]),
            "no_ear": 0.35,
            "no_glasses": 0.35,
        }
        self.min_conf = min(self.thresholds.values())
        self.iou = float(det["iou"])
        self.window = int(det["window"])
        self.needed = int(det["needed"])
        self.t_infraction = float(alert["infraction_seconds"])
        self.t_cooldown = float(alert["cooldown_seconds"])
        self.alert_cfg = alert
        self.predict_kwargs = _predict_kwargs(settings)

        self._halt = threading.Event()
        self._zones_norm: list[list[list[float]]] = []
        self._custom_epis: list[dict[str, Any]] = load_custom_epis()
        self._camera_rules: dict[str, list[str]] = load_camera_rules()
        self._cfg_lock = threading.Lock()
        self._out_lock = threading.Lock()
        self._out_frame: Optional[np.ndarray] = None
        self._out_stamp = 0
        self._raw_frame: Optional[np.ndarray] = None
        self._snapshot = Snapshot()
        self._last_fid = -1
        self._gen = cam.generation
        self._fps_times: deque[float] = deque(maxlen=20)

        self._reset_logic()

    def set_zones(self, zones_norm: list[list[list[float]]]) -> None:
        with self._cfg_lock:
            self._zones_norm = [list(map(list, poly)) for poly in zones_norm]

    def reload_ppe_config(self, epis: list[dict[str, Any]], rules: dict[str, list[str]]) -> None:
        with self._cfg_lock:
            self._custom_epis = list(epis)
            self._camera_rules = {str(k): list(v) for k, v in rules.items()}
        self._reset_logic()

    def _active_requirements(self, cam_key: Optional[str] = None) -> tuple[list[str], dict[str, str]]:
        key = cam_key if cam_key is not None else self.ctx.camera_key
        with self._cfg_lock:
            customs = list(self._custom_epis)
            rules = dict(self._camera_rules)

        labels = dict(BASE_PPE_LABELS)
        all_valid_ids = list(self.base_required)
        for epi in customs:
            cid = str(epi["id"])
            labels[cid] = str(epi["name"])
            if epi.get("enabled", True):
                all_valid_ids.append(cid)

        if key in rules:
            allowed = set(all_valid_ids)
            req = [eid for eid in rules[key] if eid in allowed]
        else:
            req = all_valid_ids

        return req, labels

    def get_output(self) -> tuple[Optional[np.ndarray], int, Snapshot]:
        with self._out_lock:
            return self._out_frame, self._out_stamp, self._snapshot

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self._out_lock:
            return None if self._raw_frame is None else self._raw_frame.copy()

    def stop(self) -> None:
        self.siren.stop_alarm()
        self._halt.set()

    def _reset_logic(self) -> None:
        self.siren.stop_alarm()
        self.filter = TemporalFilter(self.window, self.needed)
        self.missing_hist: deque[frozenset[str]] = deque(maxlen=self.window)
        self.count_hist: deque[int] = deque(maxlen=self.window)
        self.inf_start: Optional[float] = None
        self.next_trigger: Optional[float] = None
        self.registered = False
        self.last_missing_display: tuple[str, ...] = ()

    def _zones_px(self, w: int, h: int) -> list[np.ndarray]:
        with self._cfg_lock:
            zones = list(self._zones_norm)
        return zones_to_pixels(zones, w, h)

    def _detect(self, frame: np.ndarray, required: list[str]) -> list[Detection]:
        req_set = set(required)
        dets: list[Detection] = []
        person_boxes: list[Detection] = []

        # 1. Modelo customizado (YOLOv8 EPIs)
        res_ppe = self.ppe_model.predict(frame, conf=self.min_conf, iou=self.iou, verbose=False, **self.predict_kwargs)
        boxes_ppe = res_ppe[0].boxes
        if boxes_ppe is not None and len(boxes_ppe) > 0:
            xyxy = boxes_ppe.xyxy.cpu().numpy()
            confs = boxes_ppe.conf.cpu().numpy()
            classes = boxes_ppe.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
                role = self.roles.get(int(cls))
                if role is None or float(conf) < self.thresholds[role]:
                    continue
                d = Detection(role, float(conf), (float(x1), float(y1), float(x2), float(y2)))
                if role == "person":
                    person_boxes.append(d)
                elif role in req_set or role in ("no_ear", "no_glasses"):
                    dets.append(d)

        # 2. Detector auxiliar de operadores (yolov8n.pt)
        if self.person_model is not None:
            res_p = self.person_model.predict(
                frame, conf=self.thresholds["person"], iou=self.iou, classes=[0], verbose=False, **self.predict_kwargs
            )
            boxes_p = res_p[0].boxes
            if boxes_p is not None and len(boxes_p) > 0:
                xyxy = boxes_p.xyxy.cpu().numpy()
                confs = boxes_p.conf.cpu().numpy()
                for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                    cand = Detection("person", float(conf), (float(x1), float(y1), float(x2), float(y2)))
                    if not any(_iou(cand.box, existing.box) > 0.45 for existing in person_boxes):
                        person_boxes.append(cand)

        # 3. Ancoragem facial + EPIs Customizados ativos nesta camera
        with self._cfg_lock:
            customs = [e for e in self._custom_epis if e.get("enabled", True) and str(e["id"]) in req_set]

        has_yolo_ear = any(d.role == "ear" for d in dets)
        for p in person_boxes:
            if "ear" in req_set and not has_yolo_ear:
                ear_match = detect_red_earmuffs_visual(frame, p.box)
                if ear_match is not None:
                    conf_e, box_e = ear_match
                    dets.append(Detection("ear", conf_e, box_e, label_override="FONE"))

            face_box = locate_face_in_person(frame, p.box) if customs else None
            for epi in customs:
                match = match_custom_epi(frame, p.box, epi, face_box=face_box)
                if match is not None:
                    conf_c, box_c = match
                    dets.append(Detection(str(epi["id"]), conf_c, box_c, label_override=_norm(epi["name"]).upper()[:14]))

        dets.extend(person_boxes)
        return dets

    def _stable_missing(self, required: list[str]) -> tuple[str, ...]:
        counts: Counter[str] = Counter()
        for item in self.missing_hist:
            counts.update(item)
        stable = [k for k in required if counts[k] >= self.needed]
        if stable:
            self.last_missing_display = tuple(stable)
        elif not self.last_missing_display:
            recent = next((s for s in reversed(self.missing_hist) if s), frozenset())
            self.last_missing_display = tuple(k for k in required if k in recent)
        return self.last_missing_display

    def _publish(self, frame: Optional[np.ndarray], snap: Snapshot) -> None:
        with self._out_lock:
            if frame is not None:
                self._out_frame = frame
                self._out_stamp += 1
            self._snapshot = snap

    def _fps(self) -> float:
        now = time.monotonic()
        self._fps_times.append(now)
        if len(self._fps_times) < 2:
            return 0.0
        span = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / span if span > 0 else 0.0

    def _process(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        h, w = frame.shape[:2]
        with self._out_lock:
            self._raw_frame = frame
        zone_polys = self._zones_px(w, h)
        required, labels_map = self._active_requirements()

        dets = self._detect(frame, required)
        ev = evaluate_frame(dets, zone_polys, required)
        state, _ = self.filter.update(ev.raw_state)
        self.missing_hist.append(frozenset(ev.missing))
        self.count_hist.append(len(ev.persons))

        operators = Counter(self.count_hist).most_common(1)[0][0]
        missing: tuple[str, ...] = ()
        elapsed = 0.0
        trigger = False

        if state == STATE_INFRACAO:
            self.siren.start_alarm()
            missing = self._stable_missing(required)
            if self.inf_start is None:
                self.inf_start = now
                self.next_trigger = now + self.t_infraction
                self.registered = False
            elapsed = now - self.inf_start
            if self.next_trigger is not None and now >= self.next_trigger:
                trigger = True
                self.registered = True
                self.next_trigger = now + self.t_cooldown
        else:
            self.siren.stop_alarm()
            self.inf_start = None
            self.next_trigger = None
            self.registered = False
            self.last_missing_display = ()

        annotated = draw_scene(frame, ev, state, zone_polys)
        if trigger:
            self._fire_infraction(annotated, missing, labels_map)

        active_names = ", ".join(labels_map.get(r, r) for r in required) if required else "Nenhum EPI exigido"
        snap = Snapshot(
            state=state,
            missing=missing,
            operators=operators,
            fps=self._fps(),
            elapsed=elapsed,
            registered=self.registered,
            zones=len(zone_polys),
            active_epis_text=active_names,
        )
        self._publish(annotated, snap)

    def _fire_infraction(self, annotated: np.ndarray, missing: tuple[str, ...], labels_map: dict[str, str]) -> None:
        when = datetime.now()
        camera = self.ctx.camera_name
        profile = self.ctx.profile
        labels = ", ".join(labels_map.get(m, m) for m in missing) or "EPI obrigatório"
        risk = f"Ausência de EPI ({labels})"
        image = annotated.copy()

        def _persist() -> None:
            try:
                EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
                base = f"infracao_{when.strftime('%Y%m%d_%H%M%S')}"
                path = EVIDENCE_DIR / f"{base}.png"
                n = 2
                while path.exists():
                    path = EVIDENCE_DIR / f"{base}_{n}.png"
                    n += 1
                banner = [
                    "INFRAÇÃO DE SEGURANÇA - " + risk,
                    f"Câmera: {camera}   |   Perfil ativo: {profile}",
                    f"Data/Hora: {when.strftime('%d/%m/%Y %H:%M:%S')}",
                ]
                if not imwrite_unicode(path, annotate_evidence(image, banner)):
                    raise OSError("cv2.imencode falhou")
                rel = path.relative_to(BASE_DIR).as_posix()
                row_id = self.store.register(when, camera, profile, risk, rel)
                log.info("Infracao #%s registrada: %s", row_id, rel)
            except Exception:
                log.exception("Falha ao registrar a infracao.")

        threading.Thread(target=_persist, daemon=True, name="audit-writer").start()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                if not self.cam.signal_ok:
                    self._reset_logic()
                    self._publish(None, Snapshot(state=STATE_NO_SIGNAL, zones=len(self._zones_px(1, 1))))
                    time.sleep(0.1)
                    continue
                frame, fid = self.cam.get_latest()
                if frame is None or fid == self._last_fid:
                    time.sleep(0.004)
                    continue
                self._last_fid = fid
                if self.cam.generation != self._gen:
                    self._gen = self.cam.generation
                    self._reset_logic()
                self._process(frame)
            except Exception:
                log.exception("Erro no processamento de um quadro.")
                time.sleep(0.1)


# ======================================================================
# RECORTE ESTILO ROBOFLOW PARA CADASTRO DE EPI
# ======================================================================
class BoundingBoxCropperDialog(ctk.CTkToplevel):
    """Permite ao cliente/Administrador demarcar o retangulo exato do EPI (estilo Roboflow)."""

    CW, CH = 560, 380

    def __init__(self, master, fonts: dict, image_bgr: np.ndarray, title_hint: str,
                 on_cropped: Callable[[np.ndarray], None]) -> None:
        super().__init__(master, fg_color=C["panel"])
        self._img = image_bgr
        self._on_cropped = on_cropped
        self.title(f"Demarcar EPI (Estilo Roboflow) - {title_hint}")
        self.resizable(False, False)
        center_on(master, self, 600, 510)
        self.transient(master)

        fh, fw = image_bgr.shape[:2]
        self._scale, self._nw, self._nh, self._ox, self._oy = letterbox_geometry(fw, fh, self.CW, self.CH)
        canvas_img = letterbox(image_bgr, self.CW, self.CH)
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)))

        self._start: Optional[tuple[int, int]] = None
        self._rect_id: Optional[int] = None
        self._box_canvas: Optional[tuple[int, int, int, int]] = None

        ctk.CTkLabel(
            self,
            text="Arraste o mouse desenhando um retângulo APENAS ao redor da região desejada:",
            font=fonts["small"], text_color=C["cyan"], width=560, height=28, anchor="w", wraplength=560
        ).place(x=20, y=12)

        self.canvas = tk.Canvas(self, width=self.CW, height=self.CH, bg="black",
                                highlightthickness=1, highlightbackground=C["border"], cursor="crosshair")
        self.canvas.place(x=20, y=46)
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")
        self.canvas.bind("<ButtonPress-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)

        ctk.CTkButton(self, text="✔ Confirmar Recorte", width=270, height=40, font=fonts["btn"],
                      command=self._confirm, fg_color=C["cyan"], hover_color="#00B5D0", text_color="#04202B").place(x=20, y=444)
        ctk.CTkButton(self, text="Usar Imagem Inteira", width=270, height=40, font=fonts["btn"],
                      command=self._use_full, fg_color=C["border"], hover_color=C["card"]).place(x=310, y=444)
        self.after(120, self._grab)

    def _grab(self) -> None:
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def _on_down(self, event) -> None:
        self._start = (event.x, event.y)
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
            self._rect_id = None

    def _on_drag(self, event) -> None:
        if self._start is None:
            return
        x1, y1 = self._start
        x2, y2 = event.x, event.y
        if self._rect_id is not None:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline=C["cyan"], width=3)

    def _on_up(self, event) -> None:
        if self._start is None:
            return
        x1, y1 = self._start
        x2, y2 = event.x, event.y
        self._box_canvas = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def _confirm(self) -> None:
        if self._box_canvas is None:
            messagebox.showwarning("Recorte", "Desenhe um retângulo ao redor da região ou clique em 'Usar Imagem Inteira'.", parent=self)
            return
        cx1, cy1, cx2, cy2 = self._box_canvas
        if (cx2 - cx1) < 10 or (cy2 - cy1) < 10:
            messagebox.showwarning("Recorte", "Retângulo muito pequeno. Arraste sobre a região.", parent=self)
            return
        fh, fw = self._img.shape[:2]
        ix1 = int(max(0, min(fw - 1, (cx1 - self._ox) / self._scale)))
        iy1 = int(max(0, min(fh - 1, (cy1 - self._oy) / self._scale)))
        ix2 = int(max(1, min(fw, (cx2 - self._ox) / self._scale)))
        iy2 = int(max(1, min(fh, (cy2 - self._oy) / self._scale)))
        if (ix2 - ix1) < 8 or (iy2 - iy1) < 8:
            self._use_full()
            return
        crop = self._img[iy1:iy2, ix1:ix2].copy()
        cb = self._on_cropped
        self.destroy()
        cb(crop)

    def _use_full(self) -> None:
        cb = self._on_cropped
        img = self._img.copy()
        self.destroy()
        cb(img)


# ======================================================================
# JANELAS AUXILIARES (login, zonas, cadastro/regras de EPI, auditoria)
# ======================================================================
def make_fonts() -> dict[str, ctk.CTkFont]:
    return {
        "title": ctk.CTkFont(family=FONT_FAMILY, size=24, weight="bold"),
        "h2": ctk.CTkFont(family=FONT_FAMILY, size=16, weight="bold"),
        "body": ctk.CTkFont(family=FONT_FAMILY, size=14),
        "small": ctk.CTkFont(family=FONT_FAMILY, size=12),
        "tiny": ctk.CTkFont(family=FONT_FAMILY, size=11),
        "big": ctk.CTkFont(family=FONT_FAMILY, size=28, weight="bold"),
        "btn": ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
        "logo": ctk.CTkFont(family=FONT_FAMILY, size=26, weight="bold"),
        "splash": ctk.CTkFont(family=FONT_FAMILY, size=56, weight="bold"),
    }


def center_on(parent: tk.Misc, win: tk.Misc, w: int, h: int) -> None:
    parent.update_idletasks()
    x = parent.winfo_rootx() + max(0, (parent.winfo_width() - w) // 2)
    y = parent.winfo_rooty() + max(0, (parent.winfo_height() - h) // 2)
    win.geometry(f"{w}x{h}+{x}+{y}")


class LoginDialog(ctk.CTkToplevel):
    def __init__(self, master, fonts: dict, password: str, on_success: Callable[[], None]) -> None:
        super().__init__(master, fg_color=C["panel"])
        self._password = password
        self._on_success = on_success
        self.title("Acesso restrito - Administrador")
        self.resizable(False, False)
        center_on(master, self, 400, 250)
        self.transient(master)

        ctk.CTkLabel(self, text="🔐 Acesso do Administrador", font=fonts["h2"], text_color=C["cyan"], width=360, height=30, anchor="w").place(x=20, y=18)
        ctk.CTkLabel(self, text="Informe a senha de administrador para liberar esta função.", font=fonts["small"], text_color=C["muted"], width=360, height=22, anchor="w").place(x=20, y=52)
        self.entry = ctk.CTkEntry(self, show="•", width=360, height=38, placeholder_text="Senha do Administrador", font=fonts["body"])
        self.entry.place(x=20, y=88)
        self.error = ctk.CTkLabel(self, text="", font=fonts["small"], text_color=C["red"], width=360, height=22, anchor="w")
        self.error.place(x=20, y=134)
        ctk.CTkButton(self, text="Entrar", width=170, height=38, font=fonts["btn"], command=self._submit,
                      fg_color=C["cyan"], hover_color="#00B5D0", text_color="#04202B").place(x=20, y=176)
        ctk.CTkButton(self, text="Cancelar", width=170, height=38, font=fonts["btn"], command=self.destroy,
                      fg_color=C["gray_bg"], hover_color=C["border"]).place(x=210, y=176)
        self.bind("<Return>", lambda _e: self._submit())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.after(120, self._focus)

    def _focus(self) -> None:
        try:
            self.grab_set()
            self.entry.focus_set()
        except tk.TclError:
            pass

    def _submit(self) -> None:
        if self.entry.get() == self._password:
            callback = self._on_success
            self.destroy()
            callback()
        else:
            self.error.configure(text="Senha incorreta.")
            self.entry.delete(0, "end")


class CustomEPIManagerDialog(ctk.CTkToplevel):
    """Janela do Administrador para cadastrar novos EPIs (estilo Roboflow + webcam + referencia negativa)."""

    REGIONS = {
        "Nariz / Boca (ex: Máscara Hospitalar, Respirador)": "mouth",
        "Faixa dos Olhos (ex: Óculos de Proteção, Viseira)": "eyes",
        "Cabeça / Orelhas (ex: Capacete, Boné, Abafador)": "head",
        "Tronco / Corpo (ex: Avental, Colete Refletivo)": "torso",
    }

    def __init__(self, master, fonts: dict, cameras: list[dict[str, Any]], active_cam_idx: int,
                 base_required: list[str], get_live_frame: Callable[[], Optional[np.ndarray]],
                 on_updated: Callable[[list[dict[str, Any]], dict[str, list[str]]], None]) -> None:
        super().__init__(master, fg_color=C["panel"])
        self._on_updated = on_updated
        self._get_live_frame = get_live_frame
        self.cameras = cameras
        self.base_required = list(base_required)
        self.epis: list[dict[str, Any]] = load_custom_epis()
        self.rules: dict[str, list[str]] = load_camera_rules()
        self.cropped_samples: list[np.ndarray] = []
        self.negative_samples: list[np.ndarray] = []
        self.check_vars: dict[str, tk.BooleanVar] = {}

        self.title("Gerenciar EPIs por Câmera e Cadastrar Novos - SafeWork")
        self.resizable(False, False)
        center_on(master, self, 960, 620)
        self.transient(master)

        ctk.CTkLabel(self, text="🦺 Configuração de EPIs por Área / Câmera e Cadastro Dinâmico",
                     font=fonts["h2"], text_color=C["cyan"], width=920, height=30, anchor="w").place(x=20, y=14)
        ctk.CTkLabel(self, text="Demarque o EPI na foto ou câmera. Para óculos/transparentes, capture 1 amostra SEM o EPI para precisão absoluta.",
                     font=fonts["small"], text_color=C["muted"], width=920, height=22, anchor="w").place(x=20, y=46)

        form = ctk.CTkFrame(self, width=420, height=520, corner_radius=12, fg_color=C["card"], border_width=1, border_color=C["border"])
        form.place(x=20, y=80)
        form.pack_propagate(False)

        ctk.CTkLabel(form, text="1. Cadastrar Novo EPI (Recorte Guiado)", font=fonts["h2"], text_color=C["text"], width=380, height=28, anchor="w").place(x=20, y=12)

        ctk.CTkLabel(form, text="Nome do EPI:", font=fonts["small"], text_color=C["muted"], width=380, height=20, anchor="w").place(x=20, y=44)
        self.name_entry = ctk.CTkEntry(form, width=380, height=36, placeholder_text="Ex: Óculos com Grau / Máscara / Boné", font=fonts["body"])
        self.name_entry.place(x=20, y=66)

        ctk.CTkLabel(form, text="Região anatômica de verificação:", font=fonts["small"], text_color=C["muted"], width=380, height=20, anchor="w").place(x=20, y=108)
        self.region_menu = ctk.CTkOptionMenu(form, values=list(self.REGIONS.keys()), width=380, height=36,
                                             dynamic_resizing=False, font=fonts["small"], fg_color=C["border"],
                                             button_color=C["border"], button_hover_color=C["panel"])
        self.region_menu.place(x=20, y=130)

        ctk.CTkButton(form, text="📁 Abrir Foto e Demarcar EPI...", width=380, height=36, font=fonts["btn"],
                      command=self._pick_and_crop_image, fg_color=C["border"], hover_color=C["panel"]).place(x=20, y=178)

        ctk.CTkButton(form, text="📸 Capturar COM EPI (Câmera Agora)...", width=380, height=36, font=fonts["btn"],
                      command=lambda: self._capture_and_crop_live(negative=False), fg_color=C["border"], hover_color=C["panel"]).place(x=20, y=222)

        ctk.CTkButton(form, text="🚫 Capturar SEM EPI (Referência Negativa)...", width=380, height=36, font=fonts["small"],
                      command=lambda: self._capture_and_crop_live(negative=True), fg_color=C["gray_bg"], hover_color=C["border"]).place(x=20, y=266)

        self.lbl_files = ctk.CTkLabel(form, text="0 amostra(s) COM EPI  |  0 amostra(s) SEM EPI", font=fonts["small"], text_color=C["amber"],
                                      width=380, height=36, anchor="nw", wraplength=380, justify="left")
        self.lbl_files.place(x=20, y=310)

        ctk.CTkButton(form, text="✔ Salvar e Ativar EPI na Câmera", width=380, height=42, font=fonts["btn"],
                      command=self._register_epi, fg_color=C["cyan"], hover_color="#00B5D0", text_color="#04202B").place(x=20, y=354)

        ctk.CTkButton(form, text="Limpar Amostras", width=185, height=36, font=fonts["small"],
                      command=self._clear_pending, fg_color=C["gray_bg"], hover_color=C["border"]).place(x=20, y=466)
        ctk.CTkButton(form, text="🗑 Excluir EPI Customizado", width=185, height=36, font=fonts["small"],
                      command=self._delete_selected_custom, fg_color=C["red_bg"], hover_color=C["red"]).place(x=215, y=466)

        right_box = ctk.CTkFrame(self, width=480, height=520, corner_radius=12, fg_color=C["card"], border_width=1, border_color=C["border"])
        right_box.place(x=460, y=80)
        right_box.pack_propagate(False)

        ctk.CTkLabel(right_box, text="2. EPIs Obrigatórios por Câmera / Setor", font=fonts["h2"], text_color=C["text"], width=440, height=28, anchor="w").place(x=20, y=14)

        ctk.CTkLabel(right_box, text="Selecione a câmera / zona:", font=fonts["small"], text_color=C["muted"], width=440, height=20, anchor="w").place(x=20, y=48)
        cam_names = [c["name"] for c in self.cameras]
        self.cam_selector = ctk.CTkOptionMenu(right_box, values=cam_names, width=440, height=36,
                                              dynamic_resizing=False, font=fonts["body"],
                                              command=lambda _v: self._render_checklist(),
                                              fg_color=C["border"], button_color=C["border"], button_hover_color=C["panel"])
        self.cam_selector.place(x=20, y=72)
        self.cam_selector.set(cam_names[active_cam_idx])

        self.scroll = ctk.CTkScrollableFrame(right_box, width=418, height=300, fg_color=C["panel"], corner_radius=8)
        self.scroll.place(x=20, y=120)

        ctk.CTkButton(right_box, text="Aplicar nesta Câmera", width=215, height=42, font=fonts["btn"],
                      command=lambda: self._save_rules(apply_all=False), fg_color=C["cyan"], hover_color="#00B5D0", text_color="#04202B").place(x=20, y=460)
        ctk.CTkButton(right_box, text="Aplicar em Todas", width=215, height=42, font=fonts["btn"],
                      command=lambda: self._save_rules(apply_all=True), fg_color=C["border"], hover_color=C["panel"]).place(x=245, y=460)

        self._selected_custom_id: Optional[str] = None
        self._render_checklist()
        self.after(150, self._grab)

    def _grab(self) -> None:
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def _current_cam_key(self) -> str:
        names = [c["name"] for c in self.cameras]
        idx = names.index(self.cam_selector.get())
        return str(self.cameras[idx]["source"])

    def _all_epi_items(self) -> list[tuple[str, str, str]]:
        items: list[tuple[str, str, str]] = []
        for bid in self.base_required:
            items.append((bid, BASE_PPE_LABELS.get(bid, bid), "Nativo (YOLOv8)"))
        for epi in self.epis:
            items.append((str(epi["id"]), str(epi["name"]), f"Custom ({epi.get('samples', 1)} recortes)"))
        return items

    def _render_checklist(self) -> None:
        for child in self.scroll.winfo_children():
            child.destroy()
        self.check_vars.clear()
        self._selected_custom_id = None

        cam_key = self._current_cam_key()
        all_items = self._all_epi_items()
        active_set = set(self.rules[cam_key]) if cam_key in self.rules else {eid for eid, _, _ in all_items}

        for eid, label, origin in all_items:
            row = ctk.CTkFrame(self.scroll, fg_color=C["card"], corner_radius=6, height=40)
            row.pack(fill="x", pady=4, padx=4)
            var = tk.BooleanVar(value=(eid in active_set))
            self.check_vars[eid] = var
            cb = ctk.CTkCheckBox(row, text=f"{label}", variable=var, font=self.master.fonts["body"],
                                 fg_color=C["cyan"], hover_color="#00B5D0", text_color=C["text"])
            cb.pack(side="left", padx=10, pady=6)
            tag = ctk.CTkLabel(row, text=origin, font=self.master.fonts["tiny"],
                               text_color=C["cyan"] if "Nativo" in origin else C["amber"])
            tag.pack(side="right", padx=10)
            if eid.startswith("custom_"):
                row.bind("<Button-1>", lambda _e, cid=eid: setattr(self, "_selected_custom_id", cid))

    def _update_counts(self) -> None:
        self.lbl_files.configure(
            text=f"{len(self.cropped_samples)} amostra(s) COM EPI  |  {len(self.negative_samples)} amostra(s) SEM EPI",
            text_color=C["green"] if self.cropped_samples else C["amber"],
        )

    def _add_cropped_sample(self, crop: np.ndarray, negative: bool = False) -> None:
        if negative:
            self.negative_samples.append(crop)
        else:
            self.cropped_samples.append(crop)
        self._update_counts()

    def _pick_and_crop_image(self) -> None:
        file_path = filedialog.askopenfilename(
            parent=self,
            title="Selecione uma foto do EPI para demarcar",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.webp")],
        )
        if not file_path:
            return
        img = _imread_unicode(file_path)
        if img is None:
            messagebox.showerror("Imagem", "Não foi possível abrir esta imagem.", parent=self)
            return
        BoundingBoxCropperDialog(
            self, self.master.fonts, img, Path(file_path).name,
            lambda c: self._add_cropped_sample(c, negative=False)
        )

    def _capture_and_crop_live(self, negative: bool = False) -> None:
        frame = self._get_live_frame()
        if frame is None:
            messagebox.showwarning("Câmera", "Sem imagem ao vivo da câmera no momento.", parent=self)
            return
        hint = "SEM EPI (Referência Negativa)" if negative else "COM EPI"
        BoundingBoxCropperDialog(
            self, self.master.fonts, frame, hint,
            lambda c: self._add_cropped_sample(c, negative=negative)
        )

    def _clear_pending(self) -> None:
        self.cropped_samples.clear()
        self.negative_samples.clear()
        self._update_counts()

    def _register_epi(self) -> None:
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Cadastro de EPI", "Informe o nome do EPI.", parent=self)
            return
        if not self.cropped_samples:
            messagebox.showwarning("Cadastro de EPI", "Demarque pelo menos 1 recorte COM EPI.", parent=self)
            return

        region = self.REGIONS[self.region_menu.get()]
        epi_id = f"custom_{int(time.time())}"
        target_dir = CUSTOM_EPI_IMG_DIR / epi_id
        target_dir.mkdir(parents=True, exist_ok=True)

        signatures = []
        for idx, crop in enumerate(self.cropped_samples, start=1):
            sig = extract_visual_signature(crop)
            if sig:
                signatures.append(sig)
                imwrite_unicode(target_dir / f"com_epi_{idx}.png", crop)

        neg_signatures = []
        for idx, crop in enumerate(self.negative_samples, start=1):
            sig = extract_visual_signature(crop)
            if sig:
                neg_signatures.append(sig)
                imwrite_unicode(target_dir / f"sem_epi_{idx}.png", crop)

        if not signatures:
            messagebox.showerror("Cadastro de EPI", "Não foi possível extrair características dos recortes.", parent=self)
            return

        new_epi = {
            "id": epi_id,
            "name": name,
            "region": region,
            "samples": len(signatures),
            "threshold": 0.64,
            "enabled": True,
            "signatures": signatures,
            "negative_signatures": neg_signatures,
        }
        self.epis.append(new_epi)
        save_custom_epis(self.epis)

        cam_key = self._current_cam_key()
        if cam_key not in self.rules:
            self.rules[cam_key] = [eid for eid, _, _ in self._all_epi_items()]
        elif epi_id not in self.rules[cam_key]:
            self.rules[cam_key].append(epi_id)
        save_camera_rules(self.rules)
        self._on_updated(self.epis, self.rules)

        self.name_entry.delete(0, "end")
        self.cropped_samples.clear()
        self.negative_samples.clear()
        self.lbl_files.configure(text=f"'{name}' cadastrado e ativado com calibração diferencial!", text_color=C["cyan"])
        self._render_checklist()

    def _delete_selected_custom(self) -> None:
        customs = [e for e in self.epis if str(e["id"]).startswith("custom_")]
        if not customs:
            messagebox.showinfo("EPIs Customizados", "Não há EPIs customizados cadastrados.", parent=self)
            return
        target_id = self._selected_custom_id or str(customs[-1]["id"])
        self.epis = [e for e in self.epis if str(e["id"]) != target_id]
        for k in list(self.rules.keys()):
            self.rules[k] = [eid for eid in self.rules[k] if eid != target_id]
        save_custom_epis(self.epis)
        save_camera_rules(self.rules)
        self._on_updated(self.epis, self.rules)
        self._render_checklist()

    def _save_rules(self, apply_all: bool) -> None:
        selected = [eid for eid, var in self.check_vars.items() if var.get()]
        if apply_all:
            for cam in self.cameras:
                self.rules[str(cam["source"])] = list(selected)
        else:
            self.rules[self._current_cam_key()] = list(selected)
        save_camera_rules(self.rules)
        self._on_updated(self.epis, self.rules)
        scope = "todas as câmeras" if apply_all else self.cam_selector.get()
        messagebox.showinfo("Regras Salvas", f"EPIs obrigatórios atualizados para {scope} ({len(selected)} item(ns)).", parent=self)


class ZoneEditor(ctk.CTkToplevel):
    CW, CH = VIDEO_W, VIDEO_H

    def __init__(self, master, fonts: dict, frame_bgr: np.ndarray, zones_norm: list[list[list[float]]],
                 camera_name: str, on_save: Callable[[list[list[list[float]]]], None]) -> None:
        super().__init__(master, fg_color=C["panel"])
        self._on_save = on_save
        self.title(f"Delimitar Zonas de Risco - {camera_name}")
        self.resizable(False, False)
        center_on(master, self, 880, 700)
        self.transient(master)

        fh, fw = frame_bgr.shape[:2]
        self._scale, self._nw, self._nh, self._ox, self._oy = letterbox_geometry(fw, fh, self.CW, self.CH)
        canvas_img = letterbox(frame_bgr, self.CW, self.CH)
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)))

        self.polys: list[list[tuple[float, float]]] = [[(x, y) for x, y in p] for p in zones_norm]
        self.current: list[tuple[float, float]] = []

        ctk.CTkLabel(self, text="Clique para marcar os vértices da zona. Botão direito (ou \"Fechar zona\") conclui o polígono. "
                                "Sem zonas = toda a imagem é monitorada.",
                     font=fonts["small"], text_color=C["muted"], width=840, height=40, anchor="w", justify="left", wraplength=840
                     ).place(x=20, y=8)
        self.canvas = tk.Canvas(self, width=self.CW, height=self.CH, bg="black", highlightthickness=1, highlightbackground=C["border"], cursor="crosshair")
        self.canvas.place(x=20, y=52)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", lambda _e: self._close_polygon())

        y = 586
        specs = [
            ("Fechar zona", self._close_polygon, C["border"]),
            ("Desfazer ponto", self._undo, C["border"]),
            ("Limpar tudo", self._clear, C["border"]),
            ("Salvar e aplicar", self._save, C["cyan"]),
            ("Cancelar", self.destroy, C["gray_bg"]),
        ]
        x = 20
        for text, cmd, color in specs:
            is_save = text.startswith("Salvar")
            ctk.CTkButton(self, text=text, width=160, height=40, font=fonts["btn"], command=cmd, fg_color=color,
                          hover_color="#00B5D0" if is_save else C["card"],
                          text_color="#04202B" if is_save else C["text"]).place(x=x, y=y)
            x += 170
        self.info = ctk.CTkLabel(self, text="", font=fonts["small"], text_color=C["amber"], width=840, height=24, anchor="w")
        self.info.place(x=20, y=640)
        self._redraw()
        self.bind("<Escape>", lambda _e: self.destroy())
        self.after(150, self._grab)

    def _grab(self) -> None:
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def _to_canvas(self, pt: tuple[float, float]) -> tuple[float, float]:
        return self._ox + pt[0] * self._nw, self._oy + pt[1] * self._nh

    def _on_click(self, event) -> None:
        nx = min(1.0, max(0.0, (event.x - self._ox) / self._nw))
        ny = min(1.0, max(0.0, (event.y - self._oy) / self._nh))
        self.current.append((nx, ny))
        self._redraw()

    def _close_polygon(self) -> None:
        if len(self.current) < 3:
            self.info.configure(text="Marque pelo menos 3 pontos para fechar uma zona.")
            return
        self.polys.append(self.current)
        self.current = []
        self.info.configure(text=f"Zona {len(self.polys)} criada. Você pode desenhar outra ou salvar.")
        self._redraw()

    def _undo(self) -> None:
        if self.current:
            self.current.pop()
        elif self.polys:
            self.polys.pop()
        self._redraw()

    def _clear(self) -> None:
        self.polys, self.current = [], []
        self.info.configure(text="Todas as zonas removidas (salve para monitorar a imagem inteira).")
        self._redraw()

    def _redraw(self) -> None:
        cv = self.canvas
        cv.delete("all")
        cv.create_image(0, 0, image=self._photo, anchor="nw")
        for i, poly in enumerate(self.polys, start=1):
            pts = [c for p in poly for c in self._to_canvas(p)]
            cv.create_polygon(*pts, outline=C["amber"], fill="", width=3)
            cv.create_text(pts[0] + 6, pts[1] + 12, text=f"ZONA {i}", fill=C["amber"], anchor="w", font=(FONT_FAMILY, 11, "bold"))
        if self.current:
            pts = [self._to_canvas(p) for p in self.current]
            for a, b in zip(pts, pts[1:]):
                cv.create_line(*a, *b, fill=C["cyan"], width=2)
            for px, py in pts:
                cv.create_oval(px - 4, py - 4, px + 4, py + 4, fill=C["cyan"], outline="")

    def _save(self) -> None:
        if len(self.current) >= 3:
            self.polys.append(self.current)
            self.current = []
        elif self.current:
            self.info.configure(text="Há pontos soltos (< 3). Feche a zona, desfaça ou limpe antes de salvar.")
            return
        zones = [[[round(x, 5), round(y, 5)] for x, y in poly] for poly in self.polys]
        self._on_save(zones)
        self.destroy()


class AuditWindow(ctk.CTkToplevel):
    def __init__(self, master, fonts: dict, store: AuditStore) -> None:
        super().__init__(master, fg_color=C["panel"])
        self.store = store
        self.title("Relatórios de Auditoria - SafeWork")
        self.resizable(False, False)
        center_on(master, self, 1060, 600)
        self.transient(master)

        ctk.CTkLabel(self, text="Registros de Auditoria", font=fonts["h2"], text_color=C["cyan"], width=500, height=30, anchor="w").place(x=20, y=14)
        self.count = ctk.CTkLabel(self, text="", font=fonts["small"], text_color=C["muted"], width=400, height=24, anchor="e")
        self.count.place(x=640, y=16)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Safe.Treeview", background=C["card"], fieldbackground=C["card"], foreground=C["text"],
                        rowheight=26, borderwidth=0, font=(FONT_FAMILY, 10))
        style.configure("Safe.Treeview.Heading", background=C["border"], foreground=C["cyan"], font=(FONT_FAMILY, 10, "bold"))
        style.map("Safe.Treeview", background=[("selected", "#1D5F8A")])

        cols = ("id", "data", "camera", "perfil", "risco", "arquivo")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", style="Safe.Treeview", height=16)
        heads = {"id": ("ID", 50), "data": ("Data/Hora", 150), "camera": ("Câmera", 110), "perfil": ("Perfil", 115),
                 "risco": ("Tipo de risco", 315), "arquivo": ("Evidência (PNG)", 260)}
        for key, (title, width) in heads.items():
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width, anchor="w", stretch=False)
        self.tree.place(x=20, y=50, width=1020, height=450)
        self.tree.bind("<Double-1>", lambda _e: self._open_selected())

        buttons = [
            ("Abrir evidência", self._open_selected),
            ("Pasta de evidências", lambda: open_path(EVIDENCE_DIR)),
            ("Relatório HTML", self._open_html),
            ("Abrir CSV", self._open_csv),
            ("Atualizar", self.refresh),
        ]
        x = 20
        for text, cmd in buttons:
            ctk.CTkButton(self, text=text, width=190, height=38, font=fonts["btn"], command=cmd,
                          fg_color=C["border"], hover_color=C["card"]).place(x=x, y=520)
            x += 200
        self.refresh()
        self.after(150, lambda: self._try_grab())

    def _try_grab(self) -> None:
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def refresh(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        rows = self.store.fetch_all()
        for rid, when, cam, profile, risk, rel in rows:
            self.tree.insert("", "end", values=(rid, when, cam, profile, risk, rel))
        self.count.configure(text=f"{len(rows)} infração(ões) registrada(s)")

    def _open_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Auditoria", "Selecione um registro.", parent=self)
            return
        rel = self.tree.item(sel[0], "values")[5]
        path = BASE_DIR / rel
        if path.exists():
            open_path(path)
        else:
            messagebox.showwarning("Auditoria", f"Arquivo não encontrado:\n{path}", parent=self)

    def _open_html(self) -> None:
        try:
            open_path(self.store.export_all())
        except Exception as exc:
            messagebox.showerror("Auditoria", f"Não foi possível abrir o relatório:\n{exc}", parent=self)

    def _open_csv(self) -> None:
        try:
            open_path(self.store.export_csv())
        except PermissionError:
            messagebox.showwarning("Auditoria", "O CSV está aberto em outro programa. Feche-o e tente novamente.", parent=self)
        except Exception as exc:
            messagebox.showerror("Auditoria", f"Não foi possível abrir o CSV:\n{exc}", parent=self)


# ======================================================================
# SPLASH
# ======================================================================
class Splash(ctk.CTkToplevel):
    DURATION = 2.0

    def __init__(self, master, fonts: dict, is_ready: Callable[[], bool], on_done: Callable[[], None]) -> None:
        super().__init__(master, fg_color=C["bg"])
        self._is_ready = is_ready
        self._on_done = on_done
        self._t0 = time.monotonic()
        self._finished = False
        self.overrideredirect(True)
        w, h = 640, 340
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        try:
            self.attributes("-topmost", True)
        except tk.TclError:
            pass

        ctk.CTkLabel(self, text="SAFEWORK", font=fonts["splash"], text_color=C["cyan"], width=600, height=80).place(x=20, y=60)
        ctk.CTkLabel(self, text="Sistema Inteligente de Identificação e Mitigação de Riscos de Acidentes de Trabalho",
                     font=fonts["body"], text_color=C["muted"], width=560, height=44, wraplength=540, justify="center").place(x=40, y=146)
        self.bar = ctk.CTkProgressBar(self, width=520, height=10, progress_color=C["cyan"], fg_color=C["border"], corner_radius=5)
        self.bar.place(x=60, y=238)
        self.bar.set(0)
        self.status = ctk.CTkLabel(self, text="Inicializando módulos...", font=fonts["small"], text_color=C["muted"], width=520, height=24)
        self.status.place(x=60, y=260)
        self.after(20, self._step)

    def _step(self) -> None:
        if self._finished:
            return
        progress = min(1.0, (time.monotonic() - self._t0) / self.DURATION)
        self.bar.set(progress)
        if progress < 1.0:
            self.status.configure(text="Inicializando módulos...")
            self.after(20, self._step)
        elif not self._is_ready():
            self.status.configure(text="Carregando modelos de visão computacional...")
            self.after(100, self._step)
        else:
            self._finished = True
            self.destroy()
            self._on_done()


# ======================================================================
# APLICATIVO PRINCIPAL
# ======================================================================
class SafeWorkApp(ctk.CTk):
    def __init__(self, settings: dict[str, Any]) -> None:
        super().__init__(fg_color=C["bg"])
        self.withdraw()
        self.settings = settings
        self.title(APP_TITLE)
        self.resizable(False, False)
        x = max(0, (self.winfo_screenwidth() - WIN_W) // 2)
        y = max(0, (self.winfo_screenheight() - WIN_H) // 2 - 20)
        self.geometry(f"{WIN_W}x{WIN_H}+{x}+{y}")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.fonts = make_fonts()
        self.cameras = self._normalize_cameras(settings["cameras"])
        self.profile = "Operador"
        self.password = str(settings["security"]["manager_password"])
        self.ctx = RuntimeContext(self.cameras[0]["name"], str(self.cameras[0]["source"]), self.profile)
        self.store = AuditStore()
        self.zones_by_cam = load_zones()
        self.siren = ProgressiveSiren()

        self.cam: Optional[CameraSource] = None
        self.engine: Optional[SafetyEngine] = None
        self._boot: dict[str, Any] = {"done": False, "ppe_model": None, "person_model": None, "roles": None, "error": None}
        self._cfg_cache: dict[int, dict] = {}
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._last_stamp = -1
        self._tick_job: Optional[str] = None
        self._msg_until = 0.0
        self._closing = False
        self._active_cam_index = 0

        threading.Thread(target=self._load_model_bg, daemon=True, name="model-loader").start()
        self.splash = Splash(self, self.fonts, lambda: self._boot["done"], self._finish_boot)

    @staticmethod
    def _normalize_cameras(raw: list) -> list[dict[str, Any]]:
        cams = []
        for i, item in enumerate(raw):
            if isinstance(item, dict) and "source" in item:
                cams.append({"name": str(item.get("name", f"Câmera {i + 1}")), "source": item["source"]})
            else:
                cams.append({"name": f"Câmera {i + 1}", "source": item})
        return cams or [{"name": "Câmera 1", "source": 0}]

    def _load_model_bg(self) -> None:
        try:
            ppe_model, person_model, roles = load_models(self.settings)
            self._boot.update(ppe_model=ppe_model, person_model=person_model, roles=roles)
        except Exception as exc:
            log.exception("Falha ao carregar os modelos.")
            self._boot["error"] = exc
        finally:
            self._boot["done"] = True

    def _finish_boot(self) -> None:
        if self._boot["error"] is not None:
            messagebox.showerror("SafeWork - Erro ao iniciar", f"{self._boot['error']}\n\nDetalhes em logs/safework.log")
            self.destroy()
            return
        self._build_ui()
        self.cam = CameraSource(self.cameras, self.settings["capture"], 0)
        self.engine = SafetyEngine(
            self._boot["ppe_model"], self._boot["person_model"], self._boot["roles"],
            self.settings, self.cam, self.store, self.ctx, self.siren
        )
        self.engine.set_zones(self.zones_by_cam.get(self._cam_key(0), []))
        self.cam.start()
        self.engine.start()
        self.deiconify()
        self._tick()

    def _cam_key(self, index: int) -> str:
        return str(self.cameras[index]["source"])

    def _card(self, parent, x: int, y: int, w: int, h: int, fg: str = C["card"], border: str = C["border"]) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, width=w, height=h, corner_radius=14, fg_color=fg, border_width=2, border_color=border)
        card.place(x=x, y=y)
        card.grid_propagate(False)
        card.pack_propagate(False)
        return card

    def _build_ui(self) -> None:
        f = self.fonts
        self.grid_columnconfigure(0, weight=0, minsize=LEFT_W)
        self.grid_columnconfigure(1, weight=0, minsize=RIGHT_W)
        self.grid_rowconfigure(0, weight=0, minsize=WIN_H)

        self.left = ctk.CTkFrame(self, width=LEFT_W, height=WIN_H, corner_radius=0, fg_color=C["bg"])
        self.left.grid(row=0, column=0, sticky="nsew")
        self.left.grid_propagate(False)
        self.left.pack_propagate(False)

        self.right = ctk.CTkFrame(self, width=RIGHT_W, height=WIN_H, corner_radius=0, fg_color=C["panel"])
        self.right.grid(row=0, column=1, sticky="nsew")
        self.right.grid_propagate(False)
        self.right.pack_propagate(False)

        self._build_left(f)
        self._build_right(f)

    def _build_left(self, f: dict) -> None:
        ox = (LEFT_W - VIDEO_W) // 2
        ctk.CTkLabel(self.left, text="MONITORAMENTO EM TEMPO REAL", font=f["h2"], text_color=C["cyan"],
                     width=VIDEO_W, height=36, anchor="w").place(x=ox, y=18)

        self.video_box = ctk.CTkFrame(self.left, width=VIDEO_W, height=VIDEO_H, corner_radius=0, fg_color="#000000")
        self.video_box.place(x=ox, y=64)
        self.video_box.grid_propagate(False)
        self.video_box.pack_propagate(False)
        self.video_label = tk.Label(self.video_box, bg="#000000", bd=0, highlightthickness=0)
        self.video_label.place(x=0, y=0, width=VIDEO_W, height=VIDEO_H)

        legend = [("■ Conforme", C["green"]), ("■ Infração", C["red"]), ("■ EPI detectado", C["cyan"]), ("■ Zona de risco", C["amber"])]
        x = ox
        for text, color in legend:
            ctk.CTkLabel(self.left, text=text, font=f["small"], text_color=color, width=200, height=26, anchor="w").place(x=x, y=596)
            x += 210

        self.lbl_active_epis = ctk.CTkLabel(
            self.left, text="EPIs exigidos nesta câmera: ...", font=f["small"], text_color=C["cyan"],
            width=VIDEO_W, height=24, anchor="w"
        )
        self.lbl_active_epis.place(x=ox, y=626)

        info = ("Regra de decisão: Operador (X) na área e EPI obrigatório (Y) ausente → INFRAÇÃO (alarme progressivo). "
                "Operador (X) com EPIs completos ou fora da área → CONFORME (silencioso).")
        ctk.CTkLabel(self.left, text=info, font=f["small"], text_color=C["muted"], width=VIDEO_W, height=44,
                     wraplength=VIDEO_W - 8, justify="left", anchor="nw").place(x=ox, y=654)

    def _build_right(self, f: dict) -> None:
        r = self.right
        pad = 16
        cw = RIGHT_W - 2 * pad

        ctk.CTkLabel(r, text="SAFEWORK", font=f["logo"], text_color=C["cyan"], width=cw, height=32, anchor="w").place(x=pad, y=12)
        ctk.CTkLabel(r, text="Painel de telemetria", font=f["small"], text_color=C["muted"], width=cw, height=18, anchor="w").place(x=pad, y=44)

        self.status_card = self._card(r, pad, 70, cw, 176, fg=C["gray_bg"], border=C["gray"])
        self.lbl_status = ctk.CTkLabel(self.status_card, text="INICIANDO", font=f["title"], text_color=C["text"],
                                       width=cw - 32, height=40, anchor="w", wraplength=cw - 40, justify="left")
        self.lbl_status.place(x=16, y=12)
        self.lbl_detail = ctk.CTkLabel(self.status_card, text="", font=f["body"], text_color=C["text"],
                                       width=cw - 32, height=48, anchor="nw", wraplength=cw - 40, justify="left")
        self.lbl_detail.place(x=16, y=54)
        self.lbl_timer = ctk.CTkLabel(self.status_card, text="", font=f["small"], text_color=C["text"],
                                      width=cw - 32, height=22, anchor="w", wraplength=cw - 40, justify="left")
        self.lbl_timer.place(x=16, y=110)
        self.bar_timer = ctk.CTkProgressBar(self.status_card, width=cw - 36, height=14, corner_radius=7,
                                            progress_color=C["red"], fg_color=C["border"])
        self.bar_timer.place(x=16, y=142)
        self.bar_timer.set(0)

        tele = self._card(r, pad, 254, cw, 88)
        self.tele_values: dict[str, ctk.CTkLabel] = {}
        for i, (key, title) in enumerate([("ops", "OPERADORES NA ÁREA"), ("fps", "FPS"), ("zones", "ZONAS DE RISCO")]):
            cx = 12 + i * 146
            ctk.CTkLabel(tele, text=title, font=f["tiny"], text_color=C["muted"], width=136, height=18, anchor="w").place(x=cx, y=10)
            lbl = ctk.CTkLabel(tele, text="-", font=f["big"], text_color=C["text"], width=136, height=38, anchor="w")
            lbl.place(x=cx, y=34)
            self.tele_values[key] = lbl

        cam_card = self._card(r, pad, 350, cw, 72)
        ctk.CTkLabel(cam_card, text="CÂMERA ATIVA", font=f["tiny"], text_color=C["muted"], width=cw - 32, height=16, anchor="w").place(x=16, y=6)
        self.cam_menu = ctk.CTkOptionMenu(cam_card, values=[c["name"] for c in self.cameras], width=cw - 36, height=32,
                                          dynamic_resizing=False, font=f["body"], dropdown_font=f["body"],
                                          command=self._on_camera_selected, fg_color=C["border"],
                                          button_color=C["border"], button_hover_color=C["panel"])
        self.cam_menu.place(x=16, y=26)
        self.cam_menu.set(self.cameras[0]["name"])

        prof_card = self._card(r, pad, 430, cw, 72)
        ctk.CTkLabel(prof_card, text="PERFIL ATIVO", font=f["tiny"], text_color=C["muted"], width=200, height=16, anchor="w").place(x=16, y=6)
        self.lbl_profile = ctk.CTkLabel(prof_card, text="Operador", font=f["title"], text_color=C["cyan"], width=215, height=34, anchor="w")
        self.lbl_profile.place(x=16, y=26)
        self.btn_login = ctk.CTkButton(prof_card, text="Login Administrador", width=185, height=34, font=f["btn"],
                                       command=self._toggle_login, fg_color=C["border"], hover_color=C["panel"])
        self.btn_login.place(x=cw - 200, y=19)

        self.btn_zones = ctk.CTkButton(r, text="", width=cw, height=40, font=f["btn"], command=lambda: self._guard(self._open_zone_editor),
                                       fg_color=C["card"], hover_color=C["border"], border_width=1, border_color=C["border"], anchor="w")
        self.btn_zones.place(x=pad, y=512)

        self.btn_epis = ctk.CTkButton(r, text="", width=cw, height=40, font=f["btn"], command=lambda: self._guard(self._open_custom_epis),
                                      fg_color=C["card"], hover_color=C["border"], border_width=1, border_color=C["border"], anchor="w")
        self.btn_epis.place(x=pad, y=560)

        self.btn_audit = ctk.CTkButton(r, text="", width=cw, height=40, font=f["btn"], command=lambda: self._guard(self._open_audit),
                                       fg_color=C["card"], hover_color=C["border"], border_width=1, border_color=C["border"], anchor="w")
        self.btn_audit.place(x=pad, y=608)

        self.btn_export = ctk.CTkButton(r, text="", width=cw, height=40, font=f["btn"], command=lambda: self._guard(self._export_report),
                                        fg_color=C["card"], hover_color=C["border"], border_width=1, border_color=C["border"], anchor="w")
        self.btn_export.place(x=pad, y=656)

        self.lbl_footer = ctk.CTkLabel(r, text="", font=f["tiny"], text_color=C["amber"], width=cw, height=24, anchor="w", wraplength=cw)
        self.lbl_footer.place(x=pad, y=712)
        self._refresh_profile_ui()

    def _cfg(self, widget, **kw) -> None:
        key = id(widget)
        if self._cfg_cache.get(key) == kw:
            return
        widget.configure(**kw)
        self._cfg_cache[key] = dict(kw)

    def _refresh_profile_ui(self) -> None:
        admin = self.profile == "Administrador"
        lock = "" if admin else "🔒 "
        self.lbl_profile.configure(text=self.profile, text_color=C["amber"] if admin else C["cyan"])
        self.btn_login.configure(text="Sair do Administrador" if admin else "Login Administrador")
        self.btn_zones.configure(text=f"{lock}Delimitar Zonas de Risco")
        self.btn_epis.configure(text=f"{lock}🦺 Gerenciar / Cadastrar EPIs")
        self.btn_audit.configure(text=f"{lock}Acessar Relatórios de Auditoria")
        self.btn_export.configure(text=f"{lock}📄 Exportar / Abrir Relatório")
        self.ctx.profile = self.profile

    def _guard(self, action: Callable[[], None]) -> None:
        if self.profile == "Administrador":
            action()
            return
        LoginDialog(self, self.fonts, self.password, lambda: (self._set_profile("Administrador"), action()))

    def _set_profile(self, profile: str) -> None:
        self.profile = profile
        self._refresh_profile_ui()

    def _toggle_login(self) -> None:
        if self.profile == "Administrador":
            self._set_profile("Operador")
        else:
            LoginDialog(self, self.fonts, self.password, lambda: self._set_profile("Administrador"))

    def _open_zone_editor(self) -> None:
        assert self.engine is not None
        frame = self.engine.get_raw_frame()
        if frame is None:
            messagebox.showwarning("Zonas de risco", "Sem imagem da câmera no momento.", parent=self)
            return
        idx = self._active_cam_index
        ZoneEditor(self, self.fonts, frame, self.zones_by_cam.get(self._cam_key(idx), []),
                   self.cameras[idx]["name"], lambda zones, i=idx: self._apply_zones(i, zones))

    def _open_custom_epis(self) -> None:
        assert self.engine is not None

        def _on_epis_changed(updated_epis: list[dict[str, Any]], updated_rules: dict[str, list[str]]) -> None:
            if self.engine is not None:
                self.engine.reload_ppe_config(updated_epis, updated_rules)
            self._flash("Regras de EPIs por câmera atualizadas e aplicadas.")

        CustomEPIManagerDialog(
            self, self.fonts, self.cameras, self._active_cam_index,
            self.engine.base_required, self.engine.get_raw_frame, _on_epis_changed
        )

    def _apply_zones(self, cam_index: int, zones: list[list[list[float]]]) -> None:
        key = self._cam_key(cam_index)
        self.zones_by_cam[key] = zones
        try:
            save_zones(self.zones_by_cam)
        except OSError as exc:
            messagebox.showerror("Zonas de risco", f"Não foi possível salvar config/zones.json:\n{exc}", parent=self)
            return
        if cam_index == self._active_cam_index and self.engine is not None:
            self.engine.set_zones(zones)
        self._flash(f"Zonas salvas em config/zones.json ({len(zones)} zona(s)).")

    def _open_audit(self) -> None:
        AuditWindow(self, self.fonts, self.store)

    def _export_report(self) -> None:
        try:
            html_path = self.store.export_all()
            open_path(html_path)
            self._flash(f"Relatório gerado: {html_path.name} (CSV e HTML atualizados).")
        except Exception as exc:
            messagebox.showerror("Relatório", f"Não foi possível gerar/abrir o relatório:\n{exc}", parent=self)

    def _flash(self, text: str, seconds: float = 6.0) -> None:
        self._cfg(self.lbl_footer, text=text)
        self._msg_until = time.monotonic() + seconds

    def _on_camera_selected(self, name: str) -> None:
        names = [c["name"] for c in self.cameras]
        if name not in names or self.cam is None:
            return
        idx = names.index(name)
        if idx != self._active_cam_index:
            self.cam.request_switch(idx)
            self._flash(f"Conectando {name}...", 4.0)

    def _sync_camera(self) -> None:
        assert self.cam is not None and self.engine is not None
        idx = self.cam.active_index
        if idx != self._active_cam_index:
            self._active_cam_index = idx
            self.ctx.camera_name = self.cameras[idx]["name"]
            self.ctx.camera_key = self._cam_key(idx)
            self.engine.set_zones(self.zones_by_cam.get(self._cam_key(idx), []))
            self.cam_menu.set(self.cameras[idx]["name"])
        if self.cam.message:
            self._flash(self.cam.message, 6.0)
            if self.cam.signal_ok or "Falha" in self.cam.message:
                self.cam.message = ""
                self.cam_menu.set(self.cameras[self._active_cam_index]["name"])
        if self._msg_until and time.monotonic() > self._msg_until:
            self._msg_until = 0.0
            self._cfg(self.lbl_footer, text="")

    def _tick(self) -> None:
        if self._closing:
            return
        try:
            assert self.engine is not None
            frame, stamp, snap = self.engine.get_output()
            self._sync_camera()
            if stamp != self._last_stamp or snap.state == STATE_NO_SIGNAL:
                self._render_video(frame if snap.state != STATE_NO_SIGNAL else None)
                self._last_stamp = stamp
            self._render_panel(snap)
        except Exception:
            log.exception("Erro na atualizacao da interface.")
        self._tick_job = self.after(30, self._tick)

    def _render_video(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            canvas = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
            cv2.putText(canvas, "SEM SINAL DE VIDEO", (VIDEO_W // 2 - 190, VIDEO_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 116, 139), 2, cv2.LINE_AA)
        else:
            canvas = letterbox(frame, VIDEO_W, VIDEO_H)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=self._photo)

    def _render_panel(self, s: Snapshot) -> None:
        limit = self.engine.t_infraction if self.engine else 3.0
        _, labels_map = self.engine._active_requirements() if self.engine else ([], BASE_PPE_LABELS)
        if s.active_epis_text:
            self._cfg(self.lbl_active_epis, text=f"EPIs obrigatórios nesta câmera: {s.active_epis_text}")

        if s.state == STATE_INFRACAO:
            self._cfg(self.status_card, fg_color=C["red_bg"], border_color=C["red"])
            self._cfg(self.lbl_status, text="ALERTA DE INFRAÇÃO", text_color=C["red"])
            labels = ", ".join(labels_map.get(m, m) for m in s.missing) or "EPI obrigatório"
            self._cfg(self.lbl_detail, text=f"EPI ausente: {labels}")
            if s.registered:
                self._cfg(self.lbl_timer, text="🔊 Sirene ativa • Infração registrada na auditoria")
                self._cfg(self.bar_timer, progress_color=C["red"])
                self.bar_timer.set(1.0)
            else:
                self._cfg(self.lbl_timer, text=f"⏱ Infração contínua: {min(s.elapsed, limit):.1f} s / {limit:.0f} s")
                self.bar_timer.set(min(1.0, s.elapsed / limit))
        elif s.state == STATE_REGULAR:
            self._cfg(self.status_card, fg_color=C["green_bg"], border_color=C["green"])
            self._cfg(self.lbl_status, text="ÁREA REGULAR", text_color=C["green"])
            if s.operators > 0:
                self._cfg(self.lbl_detail, text="Operador(es) com todos os EPIs obrigatórios.")
            else:
                self._cfg(self.lbl_detail, text="Nenhum operador na área monitorada.")
            self._cfg(self.lbl_timer, text="Monitorando em tempo real")
            self.bar_timer.set(0)
        else:
            self._cfg(self.status_card, fg_color=C["gray_bg"], border_color=C["gray"])
            self._cfg(self.lbl_status, text="SEM SINAL DE VÍDEO", text_color=C["muted"])
            self._cfg(self.lbl_detail, text="Verifique a câmera selecionada ou troque de câmera.")
            self._cfg(self.lbl_timer, text="Aguardando sinal...")
            self.bar_timer.set(0)

        self._cfg(self.tele_values["ops"], text=str(s.operators) if s.state != STATE_NO_SIGNAL else "-")
        self._cfg(self.tele_values["fps"], text=f"{s.fps:.0f}" if s.state != STATE_NO_SIGNAL else "-")
        self._cfg(self.tele_values["zones"], text=str(s.zones) if s.zones else "Total")

    def _on_close(self) -> None:
        self._closing = True
        self.siren.shutdown()
        if self._tick_job is not None:
            try:
                self.after_cancel(self._tick_job)
            except Exception:
                pass
        for worker in (self.engine, self.cam):
            if worker is not None:
                worker.stop()
        self.destroy()


# ======================================================================
# PONTO DE ENTRADA
# ======================================================================
def main() -> None:
    setup_logging()
    for folder in (CONFIG_DIR, EVIDENCE_DIR, DATA_DIR, LOGS_DIR, CUSTOM_EPI_IMG_DIR, BASE_DIR / "models"):
        folder.mkdir(parents=True, exist_ok=True)
    settings = load_settings()

    ctk.set_appearance_mode("dark")
    ctk.set_widget_scaling(1.0)
    ctk.set_window_scaling(1.0)

    threading.excepthook = lambda args: log.error(
        "Excecao em thread %s", args.thread.name if args.thread else "?",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )

    app = SafeWorkApp(settings)
    app.mainloop()


if __name__ == "__main__":
    main()