from __future__ import annotations

from pathlib import Path

from img2drawio.models import DiagramModel, RecognitionResult


def build_diagram_model(
    image_path: str | Path,
    vlm: RecognitionResult,
    ocr: RecognitionResult,
    yolo: RecognitionResult,
) -> DiagramModel:
    """Combine recognition outputs. Relationship inference will be expanded here."""

    return DiagramModel(
        image_path=Path(image_path),
        items=[*vlm.items, *ocr.items, *yolo.items],
        relationships=[],
    )
