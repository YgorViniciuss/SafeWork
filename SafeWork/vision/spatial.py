"""Etapa 4 - Interseção espacial: operador na ROI e vínculo de EPIs à pessoa."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from safework.core.config import SpatialConfig
from safework.core.models import (
    PPE,
    Detection,
    ObjectClass,
    PersonAnalysis,
    PersonStatus,
    Point,
)
from safework.vision.roi import RiskZone


class SpatialAnalyzer:
    """Avalia cada operador no frame atual, sem lógica temporal.

    O cronômetro de 5 segundos pertence à máquina de estados (Módulo 2).
    """

    def __init__(self, config: SpatialConfig, zones: Sequence[RiskZone]) -> None:
        self._config = config
        self._zones = list(zones)
        self._regions = {PPE.HELMET: config.helmet_region, PPE.VEST: config.vest_region}

    def update_zones(self, zones: Sequence[RiskZone]) -> None:
        """Troca as zonas em uso (editor de ROI da GUI), sem reiniciar o pipeline."""
        self._zones = list(zones)  # troca atômica da referência: seguro entre threads

    def analyze(
        self,
        detections: Iterable[Detection],
        width: int,
        height: int,
        supported_classes: frozenset[ObjectClass],
    ) -> list[PersonAnalysis]:
        items = list(detections)
        persons = [item for item in items if item.object_class is ObjectClass.PERSON]
        worn_by_person = self._assign_ppe(persons, items)
        return [
            self._evaluate(person, worn_by_person[index], width, height, supported_classes)
            for index, person in enumerate(persons)
        ]

    def _evaluate(
        self,
        person: Detection,
        worn: set[PPE],
        width: int,
        height: int,
        supported_classes: frozenset[ObjectClass],
    ) -> PersonAnalysis:
        anchor = self._anchor(person)
        zone = next((item for item in self._zones if item.contains(anchor, width, height)), None)
        if zone is None:
            return PersonAnalysis(
                detection=person,
                status=PersonStatus.OUTSIDE_ZONE,
                anchor=anchor,
                worn_ppe=frozenset(worn),
            )

        unverified = {
            item for item in zone.required_ppe if item.object_class not in supported_classes
        }
        missing = zone.required_ppe - worn - unverified
        if missing:
            status = PersonStatus.NON_COMPLIANT
        elif unverified:
            status = PersonStatus.UNVERIFIED
        else:
            status = PersonStatus.COMPLIANT

        return PersonAnalysis(
            detection=person,
            status=status,
            anchor=anchor,
            zone_id=zone.id,
            zone_name=zone.name,
            worn_ppe=frozenset(worn),
            missing_ppe=frozenset(missing),
            unverified_ppe=frozenset(unverified),
        )

    def _anchor(self, person: Detection) -> Point:
        if self._config.anchor == "bottom_center":
            return person.box.bottom_center
        return person.box.center

    def _assign_ppe(
        self,
        persons: Sequence[Detection],
        detections: Iterable[Detection],
    ) -> dict[int, set[PPE]]:
        """Vincula cada EPI a uma única pessoa: a de melhor encaixe anatômico.

        Sem essa exclusividade, em operadores sobrepostos o capacete de um
        também contaria para o outro, escondendo uma infração real.
        """
        worn: dict[int, set[PPE]] = {index: set() for index in range(len(persons))}
        for detection in detections:
            if detection.object_class is ObjectClass.PERSON:
                continue
            item = PPE.HELMET if detection.object_class is ObjectClass.HELMET else PPE.VEST
            costs = {
                index: cost
                for index, person in enumerate(persons)
                if (cost := self._fit_cost(person, detection, item)) is not None
            }
            if costs:
                worn[min(costs, key=costs.__getitem__)].add(item)
        return worn

    def _fit_cost(self, person: Detection, ppe: Detection, item: PPE) -> float | None:
        """Distância do EPI à posição esperada, ou None se ele não couber na pessoa."""
        box = person.box
        if box.width <= 0 or box.height <= 0:
            return None
        center_x, center_y = ppe.box.center
        top, bottom = self._regions[item]
        relative_y = (center_y - box.y1) / box.height
        if not (box.x1 <= center_x <= box.x2 and top <= relative_y <= bottom):
            return None
        horizontal = abs(center_x - box.center[0]) / box.width
        vertical = abs(relative_y - (top + bottom) / 2)
        return horizontal + vertical
