from __future__ import annotations

from pathlib import Path

from img2drawio.models import RecognitionResult


class YOLOConnectorRecognizer:
    """YOLO11 arrow/connector recognizer placeholder for local model files."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)

    def recognize(self, image_path: str | Path) -> RecognitionResult:
        # The project intentionally avoids downloading YOLO weights. Put weights
        # under self.model_dir and load them here in production.
        return RecognitionResult(image_path=Path(image_path), stage="yolo", items=[], raw_response={"model_dir": str(self.model_dir)})
