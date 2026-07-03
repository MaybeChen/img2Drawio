from __future__ import annotations

from pathlib import Path

import click

from img2drawio.config import load_config
from img2drawio.export.drawio import export_drawio
from img2drawio.postprocess.relationships import build_diagram_model
from img2drawio.recognition.base import save_stage_outputs
from img2drawio.recognition.ocr import PaddleOCRRecognizer
from img2drawio.recognition.vlm import VLMRecognizer
from img2drawio.recognition.yolo import YOLOConnectorRecognizer


@click.group()
def main() -> None:
    """Convert images to draw.io diagrams."""


@main.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--config", "config_path", default="config.yaml", show_default=True, type=click.Path(path_type=Path))
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
def ocr(image_path: Path, config_path: Path, output_dir: Path | None) -> None:
    """Run PaddleOCR text recognition only."""

    config = load_config(config_path)
    out_dir = output_dir or config.paths.output_dir
    result = PaddleOCRRecognizer(config.paths.ocr_model_dir).recognize(image_path)
    json_path, annotated_path = save_stage_outputs(result, out_dir)
    click.echo(f"OCR JSON: {json_path}")
    click.echo(f"OCR annotated image: {annotated_path}")


@main.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--config", "config_path", default="config.yaml", show_default=True, type=click.Path(path_type=Path))
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
def yolo(image_path: Path, config_path: Path, output_dir: Path | None) -> None:
    """Run YOLO11 arrow and connector recognition only."""

    config = load_config(config_path)
    out_dir = output_dir or config.paths.output_dir
    result = YOLOConnectorRecognizer(config.paths.yolo_model_dir).recognize(image_path)
    json_path, annotated_path = save_stage_outputs(result, out_dir)
    click.echo(f"YOLO JSON: {json_path}")
    click.echo(f"YOLO annotated image: {annotated_path}")


@main.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--config", "config_path", default="config.yaml", show_default=True, type=click.Path(path_type=Path))
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
def vlm(image_path: Path, config_path: Path, output_dir: Path | None) -> None:
    """Run VLM element recognition only."""

    config = load_config(config_path)
    out_dir = output_dir or config.paths.output_dir
    result = VLMRecognizer(config.vlm).recognize(image_path)
    json_path, annotated_path = save_stage_outputs(result, out_dir)
    click.echo(f"VLM JSON: {json_path}")
    click.echo(f"VLM annotated image: {annotated_path}")


@main.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--config", "config_path", default="config.yaml", show_default=True, type=click.Path(path_type=Path))
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--drawio", "drawio_path", type=click.Path(path_type=Path), default="outputs/result.drawio", show_default=True)
def convert(image_path: Path, config_path: Path, output_dir: Path | None, drawio_path: Path) -> None:
    """Run the complete conversion pipeline."""

    config = load_config(config_path)
    out_dir = output_dir or config.paths.output_dir

    ocr_result = PaddleOCRRecognizer(config.paths.ocr_model_dir).recognize(image_path)
    yolo_result = YOLOConnectorRecognizer(config.paths.yolo_model_dir).recognize(image_path)
    vlm_result = VLMRecognizer(config.vlm).recognize(
        image_path,
        ocr_result=ocr_result,
        yolo_result=yolo_result,
    )

    for result in (ocr_result, yolo_result, vlm_result):
        save_stage_outputs(result, out_dir)

    diagram = build_diagram_model(image_path, vlm_result, ocr_result, yolo_result)
    output = export_drawio(diagram, drawio_path)
    click.echo(f"draw.io file: {output}")
