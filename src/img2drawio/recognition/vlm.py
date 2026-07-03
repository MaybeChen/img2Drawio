from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from img2drawio.models import BBox, RecognitionItem, RecognitionResult
from img2drawio.recognition.base import parse_json_object

if TYPE_CHECKING:
    from img2drawio.config import VLMConfig

VLM_SYSTEM_PROMPT = """你是“图片转 draw.io 非文本结构识别器”。

目标：
从输入图片中识别可转换为 draw.io 的非文本视觉结构，包括页面布局、容器、图形节点、图片资产、图标和装饰图形。

文本由 PaddleOCR 识别。
连线、箭头和连接关系由训练后的 YOLO11 识别。
因此，你不得识别文本、不得识别线条、不得识别箭头、不得推断节点之间的关系。

输入包括：

1. 原始图片；
2. 可选的 OCR 文本框坐标，仅用于避免将文本误识别为图形；
3. 可选的 YOLO 连线检测框，仅用于避免将连线误识别为图形。

坐标规则：

* 坐标使用 normalized_0_1000；
* 图片左上角为 (0, 0)，右下角为 (1000, 1000)；
* bbox 格式为 [x, y, width, height]；
* bbox 必须贴合元素实际边界；
* 不得将大面积空白包含到元素 bbox 中。

只允许识别以下类别：

页面区域：

* background
* header_area
* footer_area
* legend_area
* note_area
* divider_area

容器：

* container
* group
* swimlane
* boundary
* grid_container
* table_container

图形节点：

* rectangle
* rounded_rectangle
* ellipse
* diamond
* parallelogram
* hexagon
* document
* cylinder
* cloud
* actor
* generic_shape

图片与视觉资产：

* image_node
* icon_node
* logo_node
* complex_visual_node

装饰元素：

* decorative_shape
* decorative_badge
* color_block

严格禁止识别：

* 任何文字、字符、数字、标题、标签；
* 任何连接线、箭头、折线、虚线、直线；
* 任何节点间关系；
* 任何边标签；
* 任何业务语义，例如“服务”“数据库”“审批”“订单”等。

识别规则：

1. 不要输出 label、text、title、content、edge、connector、arrow、line、relation、source_id、target_id 等字段。
2. 不得依据文字内容判断节点类型，只能依据视觉形状判断。
3. OCR 文本框所在区域不是图形元素，不得输出为 node、icon 或 image。
4. YOLO 连线框、箭头框所在区域不是图形元素，不得输出为 node 或 decorative_shape。
5. 节点内部即使包含文字，也只识别节点外框，不识别其中的文字。
6. 容器应优先识别外层分组、泳道、边界框、表格整体框，不要将容器内部每条分隔线识别为元素。
7. 表格、矩阵、看板等复杂区域优先识别为 table_container 或 grid_container；不拆分单元格，除非每个单元格明显是独立业务节点。
8. 截图、照片、图表、地图、复杂插画、复杂 UI 优先识别为 image_node 或 complex_visual_node，不要强拆成基础形状。
9. 图标只有在具备独立业务含义时才识别为 icon_node；纯装饰小图标忽略。
10. 不确定的图形使用 generic_shape，不要猜测。
11. 低于 0.65 confidence 的对象不得作为确定元素，应写入 uncertainties。
12. 每个元素必须有唯一 id、type、bbox、style、confidence。
13. parent_id 只表示视觉包含关系，不表示连线关系。

输出严格遵循以下 JSON Schema：

{
"canvas": {
"background": {
"type": "solid | gradient | image | unknown",
"value": null
},
"layout_type": "flowchart | swimlane | hierarchy | network | mindmap | dashboard | grid | freeform | unknown",
"layout_direction": "left_to_right | top_to_bottom | radial | mixed | unknown",
"confidence": 0.0
},
"page_regions": [
{
"id": "region_001",
"type": "header_area | footer_area | legend_area | note_area | divider_area",
"bbox": [0, 0, 0, 0],
"confidence": 0.0
}
],
"containers": [
{
"id": "container_001",
"type": "container | group | swimlane | boundary | grid_container | table_container",
"bbox": [0, 0, 0, 0],
"parent_id": null,
"style": {
"fill": null,
"stroke": null,
"stroke_width_level": "thin | normal | thick | unknown",
"rounded": null,
"dashed": null,
"shadow": null
},
"confidence": 0.0
}
],
"nodes": [
{
"id": "node_001",
"type": "rectangle | rounded_rectangle | ellipse | diamond | parallelogram | hexagon | document | cylinder | cloud | actor | generic_shape | image_node | icon_node | logo_node | complex_visual_node | decorative_shape | decorative_badge | color_block",
"bbox": [0, 0, 0, 0],
"parent_id": null,
"style": {
"fill": null,
"stroke": null,
"stroke_width_level": "thin | normal | thick | unknown",
"rounded": null,
"dashed": null,
"shadow": null,
"opacity": null,
"shape_hint": null
},
"confidence": 0.0
}
],
"uncertainties": [
{
"region_bbox": [0, 0, 0, 0],
"issue": "",
"candidates": [],
"confidence": 0.0
}
]
}

输出要求：

* 只输出 JSON；
* 不输出 Markdown；
* 不输出解释；
* 不输出任何图片中的文字；
* 不输出任何连线、箭头或关系信息；
* 不得虚构图片中不存在的元素。
"""


class VLMRecognizer:
    """Recognize non-text, non-connector visual structures with a streaming VLM API."""

    def __init__(self, config: "VLMConfig") -> None:
        self.config = config

    def recognize(
        self,
        image_path: str | Path,
        ocr_result: RecognitionResult | None = None,
        yolo_result: RecognitionResult | None = None,
    ) -> RecognitionResult:
        from PIL import Image

        path = Path(image_path)
        prompt = _build_prompt(ocr_result, yolo_result)
        with Image.open(path) as image:
            content = "".join(
                api_image_request(
                    image=image,
                    prompt=prompt,
                    base_url=str(self.config.base_url),
                    model=self.config.model,
                    x_hw_id=self.config.x_hw_id,
                    x_hw_appkey=self.config.x_hw_appkey,
                    timeout=self.config.timeout,
                    proxy=self.config.proxy,
                    response_format={"type": "json_object"},
                )
            )
        parsed = parse_json_object(content or "{}")
        return RecognitionResult(
            image_path=path,
            stage="vlm",
            items=_items_from_response(parsed),
            raw_response=parsed,
        )


def api_image_request(
    image: "Image.Image",
    prompt: str,
    base_url: str,
    model: str,
    x_hw_id: str,
    x_hw_appkey: str,
    timeout: float,
    proxy: str | None = None,
    **params: Any,
) -> Iterable[str]:
    from io import BytesIO

    import requests

    img_io = BytesIO()
    image.save(img_io, "PNG")
    image_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_base64},
            ],
        }
    ]
    headers = {
        "X-HW-ID": x_hw_id,
        "X-HW-APPKEY": x_hw_appkey,
    }
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        **params,
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
        verify=False,
        stream=True,
        proxies=_build_proxies(proxy),
    )
    if not response.ok:
        raise RuntimeError(f"Error calling the API. Response was {response.text}")
    response.raise_for_status()
    return _parse_stream_response(response)


def _parse_stream_response(response: Any) -> Iterable[str]:
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith(b"data:"):
            chunk = line[5:].decode("utf-8").strip()
            if chunk == "[DONE]":
                return
            chunk_json = json.loads(chunk)
            if not chunk_json.get("choices"):
                yield ""
                continue
            generations = [
                generation.get("delta", {}).get("content", "")
                for generation in chunk_json.get("choices", [])
            ]
            yield "".join(generations)


def _build_proxies(proxy: str | None) -> dict[str, str] | None:
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def _build_prompt(
    ocr_result: RecognitionResult | None,
    yolo_result: RecognitionResult | None,
) -> str:
    return f"{VLM_SYSTEM_PROMPT}\n\n输入辅助信息：\n{_build_user_context(ocr_result, yolo_result)}"

def _build_user_context(
    ocr_result: RecognitionResult | None,
    yolo_result: RecognitionResult | None,
) -> str:
    context = {
        "instruction": "请基于原始图片识别非文本、非连线的 draw.io 视觉结构，并严格返回系统消息指定的 JSON Schema。",
        "ocr_text_boxes_normalized_0_1000": _reference_boxes(ocr_result),
        "yolo_connector_boxes_normalized_0_1000": _reference_boxes(yolo_result),
    }
    return json.dumps(context, ensure_ascii=False)


def _reference_boxes(result: RecognitionResult | None) -> list[dict[str, Any]]:
    if result is None:
        return []
    boxes = []
    for item in result.items:
        boxes.append(
            {
                "id": item.id,
                "bbox": [item.bbox.x, item.bbox.y, item.bbox.width, item.bbox.height],
                "confidence": item.confidence,
            }
        )
    return boxes


def _items_from_response(parsed: dict[str, Any]) -> list[RecognitionItem]:
    items: list[RecognitionItem] = []
    for section, kind in (("page_regions", "element"), ("containers", "element"), ("nodes", "element")):
        for fallback_index, element in enumerate(parsed.get(section, []) or [], start=1):
            confidence = element.get("confidence")
            if confidence is not None and float(confidence) < 0.65:
                continue
            bbox = _bbox_from_list(element.get("bbox") or [0, 0, 0, 0])
            element_id = str(element.get("id") or f"{section}_{fallback_index:03d}")
            element_type = str(element.get("type") or "generic_shape")
            items.append(
                RecognitionItem(
                    id=element_id,
                    label=element_type,
                    bbox=bbox,
                    confidence=confidence,
                    kind=kind,  # type: ignore[arg-type]
                    attributes={
                        "vlm_section": section,
                        "type": element_type,
                        "parent_id": element.get("parent_id"),
                        "style": element.get("style") or {},
                        "coordinate_system": "normalized_0_1000",
                    },
                )
            )
    return items


def _bbox_from_list(raw_bbox: list[Any]) -> BBox:
    values = [float(value) for value in [*raw_bbox, 0, 0, 0, 0][:4]]
    return BBox(x=values[0], y=values[1], width=values[2], height=values[3])
