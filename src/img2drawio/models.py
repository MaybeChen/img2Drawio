from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class BBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class RecognitionItem:
    id: str
    label: str
    bbox: BBox
    kind: Literal["element", "text", "connector"]
    confidence: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecognitionResult:
    image_path: Path
    stage: Literal["vlm", "ocr", "yolo"]
    items: list[RecognitionItem] = field(default_factory=list)
    raw_response: Any | None = None

    def model_dump_json(self, indent: int | None = None) -> str:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        return json.dumps(data, ensure_ascii=False, indent=indent)


@dataclass
class Relationship:
    source_id: str
    target_id: str
    connector_id: str | None = None
    relation_type: str = "connects_to"


@dataclass
class DiagramModel:
    image_path: Path
    items: list[RecognitionItem]
    relationships: list[Relationship] = field(default_factory=list)
