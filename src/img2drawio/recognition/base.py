from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


from img2drawio.models import RecognitionItem, RecognitionResult


def write_result_json(result: RecognitionResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def draw_annotations(
    image_path: str | Path,
    items: Iterable[RecognitionItem],
    output_path: str | Path,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for item in items:
        x1, y1, x2, y2 = item.bbox.xyxy
        color = _color_for_kind(item.kind)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{item.id}:{item.label}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), label, fill="white", font=font)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def save_stage_outputs(result: RecognitionResult, output_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    stem = Path(result.image_path).stem
    json_path = write_result_json(result, out_dir / f"{stem}_{result.stage}.json")
    annotated_path = draw_annotations(
        result.image_path,
        result.items,
        out_dir / f"{stem}_{result.stage}_annotated.png",
    )
    return json_path, annotated_path


def parse_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return json.loads(stripped)


def _color_for_kind(kind: str) -> tuple[int, int, int]:
    return {
        "element": (0, 128, 255),
        "text": (46, 160, 67),
        "connector": (209, 84, 0),
    }.get(kind, (220, 20, 60))
