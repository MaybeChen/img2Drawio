from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI
    from img2drawio.config import VLMConfig
from img2drawio.models import BBox, RecognitionItem, RecognitionResult
from img2drawio.recognition.base import parse_json_object

VLM_SYSTEM_PROMPT = """你是专业的图像结构识别助手。请只识别图片中非文字的视觉元素，忽略纯文本内容。
输出必须是 JSON，格式为：
{"elements":[{"label":"元素名称","bbox":{"x":0,"y":0,"width":0,"height":0},"confidence":0.0,"attributes":{}}]}
坐标使用输入图片像素坐标，bbox 为左上角 x/y 与宽高。不要输出 Markdown 或额外解释。
"""


class VLMRecognizer:
    """Recognize non-text elements through an OpenAI-compatible VLM service."""

    def __init__(self, config: "VLMConfig") -> None:
        self.config = config
        from openai import OpenAI

        self.client = OpenAI(
            api_key="unused",
            base_url=str(config.base_url),
            default_headers={
                "X-HW-ID": config.x_hw_id,
                "X-HW-APPKEY": config.x_hw_appkey,
            },
            timeout=config.timeout,
        )

    def recognize(self, image_path: str | Path) -> RecognitionResult:
        path = Path(image_path)
        image_url = _image_to_data_url(path)
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "识别图片中除文本以外的元素，并按要求返回 JSON。"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = parse_json_object(content)
        return RecognitionResult(
            image_path=path,
            stage="vlm",
            items=_items_from_response(parsed),
            raw_response=parsed,
        )


def _image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _items_from_response(parsed: dict[str, Any]) -> list[RecognitionItem]:
    elements = parsed.get("elements", [])
    items: list[RecognitionItem] = []
    for index, element in enumerate(elements, start=1):
        bbox = element.get("bbox") or {}
        items.append(
            RecognitionItem(
                id=f"element_{index}",
                label=str(element.get("label") or "element"),
                bbox=BBox(
                    x=float(bbox.get("x", 0)),
                    y=float(bbox.get("y", 0)),
                    width=float(bbox.get("width", 0)),
                    height=float(bbox.get("height", 0)),
                ),
                confidence=element.get("confidence"),
                kind="element",
                attributes=element.get("attributes") or {},
            )
        )
    return items
