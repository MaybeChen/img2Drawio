# img2drawio

`img2drawio` is a Poetry-based Python project for converting an input image into a draw.io diagram.

## Pipeline

1. **PaddleOCR v6 text recognition**: uses `paddleocr>=3.7.0,<4.0.0` and `paddlepaddle>=3.0.0,<3.3.0`. Put local OCR models under `models/ocr`; the project does not download models.
2. **YOLO11 arrow/connector recognition**: put local YOLO model files under `models/yolo`; the project does not download models.
3. **VLM element recognition**: identifies non-text and non-connector visual elements through an OpenAI-compatible custom service. Runtime settings are loaded from `config.yaml`, including `base_url`, `model`, `x_hw_id`, `x_hw_appkey`, and optional `proxy`. Requests include `X-HW-ID` and `X-HW-APPKEY` headers, and OCR/YOLO boxes are passed as optional exclusion context.
4. **Relationship building**: combines coordinates from text, elements, and connectors.
5. **draw.io export**: emits a `.drawio` file.

Each recognition stage writes a JSON result and an annotated image.

## Quick start

```bash
poetry install
cp config.yaml.example config.yaml
img2drawio ocr path/to/image.png --config config.yaml --output-dir outputs
img2drawio yolo path/to/image.png --config config.yaml --output-dir outputs
img2drawio vlm path/to/image.png --config config.yaml --output-dir outputs
```

OCR、YOLO、VLM 三个阶段都可以单独执行，并会分别生成 JSON 中间结果和标注图片。

Run the full pipeline skeleton:

```bash
img2drawio convert path/to/image.png --config config.yaml --output-dir outputs --drawio outputs/result.drawio
```

> OCR and YOLO stages are wired to local model directories but intentionally avoid downloading models. Add your model files before enabling those recognizers in production.
