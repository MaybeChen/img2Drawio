from img2drawio.recognition.vlm import _items_from_response


def test_items_from_vlm_response():
    items = _items_from_response(
        {
            "elements": [
                {
                    "label": "server",
                    "bbox": {"x": 10, "y": 20, "width": 30, "height": 40},
                    "confidence": 0.9,
                }
            ]
        }
    )

    assert len(items) == 1
    assert items[0].id == "element_1"
    assert items[0].label == "server"
    assert items[0].bbox.x == 10
