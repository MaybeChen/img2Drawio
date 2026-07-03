from __future__ import annotations

from pathlib import Path

from img2drawio.models import RecognitionResult


class PaddleOCRRecognizer:
    """PaddleOCR v6 recognizer placeholder wired for local model directories."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)

    def recognize(self, image_path: str | Path) -> RecognitionResult:
        # The project intentionally avoids downloading OCR models. Add PaddleOCR
        # initialization here after placing required models under self.model_dir.
        return RecognitionResult(image_path=Path(image_path), stage="ocr", items=[], raw_response={"model_dir": str(self.model_dir)})
