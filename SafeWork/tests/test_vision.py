"""Testes do Módulo 1. Não exigem câmera, janela nem pesos do YOLO."""

from __future__ import annotations

import json
import time
from datetime import datetime

import cv2
import numpy as np
import pytest

from safework.core.config import (
    CameraConfig,
    ConfigError,
    DisplayConfig,
    PreprocessingConfig,
    Settings,
    SpatialConfig,
    resolve_path,
)
from safework.core.models import (
    PPE,
    BoundingBox,
    Detection,
    Frame,
    ObjectClass,
    PersonStatus,
)
from safework.vision.capture import VideoSource, mask_credentials
from safework.vision.detector import build_class_map
from safework.vision.pipeline import VisionPipeline
from safework.vision.preprocessing import FramePreprocessor
from safework.vision.renderer import FrameRenderer, ascii_text
from safework.vision.roi import RiskZone, ZoneRepository
from safework.vision.roi_editor import ZoneEditorState
from safework.vision.spatial import SpatialAnalyzer

ALL_CLASSES = frozenset(ObjectClass)
PERSON_ONLY = frozenset({ObjectClass.PERSON})
ALIASES = {
    "person": ["person"],
    "helmet": ["helmet", "hardhat"],
    "vest": ["vest", "safety vest"],
}
SQUARE = ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75))


def zone(required=frozenset({PPE.HELMET, PPE.VEST})) -> RiskZone:
    return RiskZone("z1", "Zona Crítica", SQUARE, required)


def person(x1=450, y1=300, x2=550, y2=600, track_id=1) -> Detection:
    return Detection(ObjectClass.PERSON, 0.9, BoundingBox(x1, y1, x2, y2), track_id)


def helmet(cx=500, cy=330) -> Detection:
    return Detection(ObjectClass.HELMET, 0.8, BoundingBox(cx - 20, cy - 20, cx + 20, cy + 20))


def vest(cx=500, cy=440) -> Detection:
    return Detection(ObjectClass.VEST, 0.8, BoundingBox(cx - 40, cy - 60, cx + 40, cy + 60))


def analyze(detections, supported=ALL_CLASSES, zones=None, config=None):
    analyzer = SpatialAnalyzer(config or SpatialConfig(), zones or [zone()])
    return analyzer.analyze(detections, 1000, 800, supported)


# ------------------------------------------------------------------------- ROI
@pytest.mark.parametrize(
    ("point", "expected"),
    [((500, 400), True), ((100, 100), False), ((250, 400), True), ((900, 700), False)],
)
def test_point_in_polygon(point, expected):
    assert zone().contains(point, 1001, 801) is expected


def test_concave_polygon_excludes_notch():
    u_shape = RiskZone(
        "u", "U",
        ((0.1, 0.1), (0.3, 0.1), (0.3, 0.7), (0.7, 0.7), (0.7, 0.1), (0.9, 0.1),
         (0.9, 0.9), (0.1, 0.9)),
    )
    assert not u_shape.contains((500, 200), 1001, 1001)
    assert u_shape.contains((200, 200), 1001, 1001)


def test_mask_matches_polygon():
    mask = zone().mask(1000, 800)
    assert mask.shape == (800, 1000)
    assert mask[400, 500] == 255 and mask[50, 50] == 0


def test_zone_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        RiskZone("a", "A", ((0, 0), (1, 1)))
    with pytest.raises(ValueError):
        RiskZone("a", "A", ((0, 0), (1.5, 0), (1, 1)))


def test_repository_round_trip_keeps_accents(tmp_path):
    path = tmp_path / "zones.json"
    repository = ZoneRepository(path)
    repository.save([zone()])
    assert repository.load() == [zone()]
    assert "Zona Crítica" in path.read_text(encoding="utf-8")


def test_repository_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text(json.dumps({"zones": [zone().to_dict(), zone().to_dict()]}), encoding="utf-8")
    with pytest.raises(ValueError):
        ZoneRepository(path).load()


def test_editor_state_flow():
    state = ZoneEditorState(1001, 801, [zone()])
    state.add_vertex(0, 0)
    state.add_vertex(1000, 0)
    assert state.close_polygon(frozenset({PPE.HELMET})) is None
    state.add_vertex(1000, 800)
    state.add_vertex(5, 5)
    state.undo_vertex()
    created = state.close_polygon(frozenset({PPE.HELMET}))
    assert created is not None and created.id == "zona-1"
    assert created.points == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    assert state.draft == []


# ---------------------------------------------------------------------- spatial
def test_person_with_all_ppe_is_compliant():
    result = analyze([person(), helmet(), vest()])[0]
    assert result.status is PersonStatus.COMPLIANT
    assert result.zone_name == "Zona Crítica"


def test_person_without_helmet_is_non_compliant():
    result = analyze([person(), vest()])[0]
    assert result.status is PersonStatus.NON_COMPLIANT
    assert result.missing_ppe == {PPE.HELMET}


def test_person_outside_zone_is_not_an_alert():
    result = analyze([person(10, 10, 100, 200)])[0]
    assert result.status is PersonStatus.OUTSIDE_ZONE
    assert result.zone_id is None


def test_anchor_mode_changes_zone_decision():
    tall = person(450, 300, 550, 790)  # centro dentro, pés fora da zona
    assert analyze([tall])[0].zone_id == "z1"
    feet = SpatialConfig(anchor="bottom_center")
    assert analyze([tall], config=feet)[0].zone_id is None


def test_helmet_on_floor_is_not_worn():
    result = analyze([person(), helmet(cy=590), vest()])[0]
    assert PPE.HELMET in result.missing_ppe


def test_ppe_goes_to_only_one_overlapping_person():
    left = person(400, 300, 520, 600, track_id=1)
    right = person(480, 300, 600, 600, track_id=2)
    results = analyze([left, right, helmet(cx=450)], zones=[zone(frozenset({PPE.HELMET}))])
    by_id = {item.track_id: item.status for item in results}
    assert by_id == {1: PersonStatus.COMPLIANT, 2: PersonStatus.NON_COMPLIANT}


def test_model_without_ppe_classes_never_reports_violation():
    result = analyze([person()], supported=PERSON_ONLY)[0]
    assert result.status is PersonStatus.UNVERIFIED
    assert result.missing_ppe == frozenset()


def test_partial_model_flags_only_what_it_can_see():
    supported = frozenset({ObjectClass.PERSON, ObjectClass.HELMET})
    result = analyze([person()], supported=supported)[0]
    assert result.status is PersonStatus.NON_COMPLIANT
    assert result.missing_ppe == {PPE.HELMET}
    assert result.unverified_ppe == {PPE.VEST}


# --------------------------------------------------------------------- detector
def test_coco_indices_are_not_treated_as_ppe():
    coco = {0: "person", 1: "bicycle", 2: "car"}
    assert build_class_map(coco, ALIASES) == {0: ObjectClass.PERSON}


def test_class_map_uses_names_and_rejects_negative_classes():
    roboflow = {0: "Hardhat", 1: "NO-Hardhat", 2: "Person", 3: "Safety_Vest"}
    assert build_class_map(roboflow, ALIASES) == {
        0: ObjectClass.HELMET,
        2: ObjectClass.PERSON,
        3: ObjectClass.VEST,
    }


# ------------------------------------------------------------------------ frame
def test_preprocessor_keeps_aspect_ratio_and_channels():
    image = np.zeros((720, 1280, 3), np.uint8)
    image[:, :, 0] = 255  # azul em BGR
    result = FramePreprocessor(PreprocessingConfig(target_width=640)).process(image)
    assert result.shape == (360, 640, 3)
    assert result[0, 0].tolist() == [255, 0, 0]


def test_renderer_does_not_modify_original_and_accepts_accents():
    image = np.zeros((800, 1000, 3), np.uint8)
    renderer = FrameRenderer(CameraConfig(area="Área de Prensas"), DisplayConfig(), [zone()])
    persons = analyze([person()])
    output = renderer.render(image, persons, [person()], datetime(2026, 11, 5, 12, 0, 0), 15.0)
    assert image.sum() == 0
    assert output.sum() > 0
    assert ascii_text("Área Crítica") == "Area Critica"


class FakeDetector:
    supported_classes = ALL_CLASSES

    def detect(self, image):
        return [person(), vest()]


def test_pipeline_process_end_to_end():
    pipeline = VisionPipeline(
        FramePreprocessor(PreprocessingConfig(target_width=1000)),
        FakeDetector(),
        SpatialAnalyzer(SpatialConfig(), [zone()]),
        FrameRenderer(CameraConfig(), DisplayConfig(), [zone()]),
    )
    frame = Frame(np.zeros((400, 500, 3), np.uint8), 0, 0.0, datetime.now())
    result = pipeline.process(frame)
    assert result.annotated_image.shape == (800, 1000, 3)
    assert result.persons[0].missing_ppe == {PPE.HELMET}


# ---------------------------------------------------------------------- capture
def test_credentials_never_appear_in_logs():
    url = "rtsp://admin:s3nh4@10.0.0.5:554/live"
    assert mask_credentials(url) == "rtsp://***@10.0.0.5:554/live"
    assert mask_credentials(0) == "0"


class FakeCapture:
    """Simula uma câmera que entrega alguns frames e então cai."""

    def __init__(self, frames, opened=True):
        self._frames = frames
        self._opened = opened

    def isOpened(self):
        return self._opened

    def read(self):
        if self._frames <= 0:
            return False, None
        self._frames -= 1
        time.sleep(0.005)
        return True, np.zeros((10, 10, 3), np.uint8)

    def set(self, *_):
        return True

    def get(self, *_):
        return 0.0

    def release(self):
        pass


def collect(source, seconds=2.0):
    frames = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not source.finished:
        frame = source.read(timeout=0.05)
        if frame:
            frames.append(frame)
    return frames


def test_live_source_reconnects_after_stream_failure():
    captures = iter([FakeCapture(3), FakeCapture(0, opened=False), FakeCapture(3)])
    config = CameraConfig(source="rtsp://c/x", reconnect_delay_s=0.01,
                          max_reconnect_delay_s=0.02)
    with VideoSource(config, lambda _: next(captures, FakeCapture(0, opened=False))) as source:
        frames = collect(source, 1.0)
    assert source.error is None
    assert len(frames) >= 4  # recebeu frames de depois da reconexão


def test_live_source_stops_at_attempt_limit():
    config = CameraConfig(source="rtsp://c/x", reconnect_delay_s=0.01,
                          max_reconnect_attempts=3)
    with VideoSource(config, lambda _: FakeCapture(0, opened=False)) as source:
        collect(source)
    assert source.finished and "3 tentativas" in source.error


def test_file_source_delivers_every_frame_in_order(tmp_path):
    path = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    for _ in range(15):
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()

    with VideoSource(CameraConfig(source=str(path))) as source:
        frames = []
        while True:
            frame = source.read(timeout=0.5)
            if frame is None and source.finished:
                break
            if frame:
                time.sleep(0.005)  # consumidor mais lento que o leitor
                frames.append(frame)
    assert [item.index for item in frames] == list(range(15))
    assert frames[10].timestamp == pytest.approx(1.0)


def test_missing_file_reports_error(tmp_path):
    with VideoSource(CameraConfig(source=str(tmp_path / "nao_existe.mp4"))) as source:
        collect(source)
    assert source.finished and source.error


# ----------------------------------------------------------------------- config
def test_project_settings_are_valid():
    settings = Settings.load()
    assert settings.detector.class_aliases["helmet"]
    assert isinstance(settings.spatial.helmet_region, tuple)
    assert ZoneRepository(resolve_path(settings.zones_file)).load()


def test_invalid_settings_are_rejected(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("spatial:\n  anchor: joelho\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        Settings.load(path)
    path.write_text("camera:\n  chave_errada: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        Settings.load(path)


def test_editor_clamps_clicks_outside_the_image():
    state = ZoneEditorState(101, 101)
    state.add_vertex(-5, 500)
    assert state.draft == [(0.0, 1.0)]
