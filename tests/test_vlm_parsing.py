from img2drawio.models import BBox, RecognitionItem, RecognitionResult
from img2drawio.recognition.vlm import (
    _build_proxies,
    _chat_completions_url,
    _build_user_context,
    _items_from_response,
    _parse_stream_response,
)


def test_items_from_vlm_schema_response_skips_low_confidence():
    items = _items_from_response(
        {
            "page_regions": [
                {"id": "region_001", "type": "header_area", "bbox": [0, 0, 1000, 80], "confidence": 0.9}
            ],
            "containers": [
                {
                    "id": "container_001",
                    "type": "container",
                    "bbox": [10, 20, 300, 400],
                    "parent_id": None,
                    "style": {"stroke_width_level": "normal"},
                    "confidence": 0.8,
                }
            ],
            "nodes": [
                {"id": "node_001", "type": "rectangle", "bbox": [20, 30, 80, 40], "confidence": 0.64}
            ],
        }
    )

    assert [item.id for item in items] == ["region_001", "container_001"]
    assert items[1].label == "container"
    assert items[1].bbox.x == 10
    assert items[1].attributes["coordinate_system"] == "normalized_0_1000"


def test_user_context_includes_ocr_and_yolo_reference_boxes():
    ocr = RecognitionResult(
        image_path="demo.png",
        stage="ocr",
        items=[RecognitionItem(id="text_1", label="text", kind="text", bbox=BBox(1, 2, 3, 4))],
    )
    yolo = RecognitionResult(
        image_path="demo.png",
        stage="yolo",
        items=[RecognitionItem(id="connector_1", label="arrow", kind="connector", bbox=BBox(5, 6, 7, 8))],
    )

    context = _build_user_context(ocr, yolo)

    assert '"ocr_text_boxes_normalized_0_1000": [{"id": "text_1", "bbox": [1, 2, 3, 4]' in context
    assert '"yolo_connector_boxes_normalized_0_1000": [{"id": "connector_1", "bbox": [5, 6, 7, 8]' in context


def test_parse_stream_response_concatenates_delta_content():
    class FakeResponse:
        def iter_lines(self):
            return iter(
                [
                    b'',
                    b'data: {"choices":[{"delta":{"content":"{\\\"a\\\":"}}]}',
                    b'data: {"choices":[{"delta":{"content":"1}"}}]}',
                    b'data: [DONE]',
                ]
            )

    assert ''.join(_parse_stream_response(FakeResponse())) == '{"a":1}'


def test_build_proxies_returns_requests_proxy_mapping():
    assert _build_proxies(None) is None
    assert _build_proxies("http://127.0.0.1:8080") == {
        "http": "http://127.0.0.1:8080",
        "https": "http://127.0.0.1:8080",
    }


def test_chat_completions_url_does_not_duplicate_suffix():
    assert _chat_completions_url("https://example.com/v1") == "https://example.com/v1/chat/completions"
    assert (
        _chat_completions_url("https://example.com/v1/chat/completions")
        == "https://example.com/v1/chat/completions"
    )
    assert (
        _chat_completions_url("https://example.com/v1/chat/completions/")
        == "https://example.com/v1/chat/completions"
    )
