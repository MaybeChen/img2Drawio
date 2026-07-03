from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VLMConfig:
    """OpenAI-compatible VLM service configuration."""

    base_url: str
    model: str
    x_hw_id: str
    x_hw_appkey: str
    timeout: float = 120


@dataclass(frozen=True)
class PathsConfig:
    """Local model and output paths."""

    ocr_model_dir: Path = Path("models/ocr")
    yolo_model_dir: Path = Path("models/yolo")
    output_dir: Path = Path("outputs")


@dataclass(frozen=True)
class AppConfig:
    vlm: VLMConfig
    paths: PathsConfig = PathsConfig()


def load_config(path: str | Path) -> AppConfig:
    """Load and validate YAML configuration."""

    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file_obj:
        raw = yaml.safe_load(file_obj) or {}
    return _parse_config(raw)


def _parse_config(raw: dict[str, Any]) -> AppConfig:
    vlm_raw = raw.get("vlm") or {}
    required = ("base_url", "model", "x_hw_id", "x_hw_appkey")
    missing = [key for key in required if not vlm_raw.get(key)]
    if missing:
        raise ValueError(f"Missing VLM config keys: {', '.join(missing)}")

    paths_raw = raw.get("paths") or {}
    return AppConfig(
        vlm=VLMConfig(
            base_url=str(vlm_raw["base_url"]),
            model=str(vlm_raw["model"]),
            x_hw_id=str(vlm_raw["x_hw_id"]),
            x_hw_appkey=str(vlm_raw["x_hw_appkey"]),
            timeout=float(vlm_raw.get("timeout", 120)),
        ),
        paths=PathsConfig(
            ocr_model_dir=Path(paths_raw.get("ocr_model_dir", "models/ocr")),
            yolo_model_dir=Path(paths_raw.get("yolo_model_dir", "models/yolo")),
            output_dir=Path(paths_raw.get("output_dir", "outputs")),
        ),
    )
