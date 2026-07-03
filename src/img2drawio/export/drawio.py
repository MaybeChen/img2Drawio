from __future__ import annotations

import html
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from img2drawio.models import DiagramModel


def export_drawio(diagram: DiagramModel, output_path: str | Path) -> Path:
    """Export recognized items as a minimal draw.io mxfile."""

    mxfile = Element("mxfile", host="img2drawio")
    diagram_el = SubElement(mxfile, "diagram", name="Page-1")
    graph = SubElement(diagram_el, "mxGraphModel")
    root = SubElement(graph, "root")
    SubElement(root, "mxCell", id="0")
    SubElement(root, "mxCell", id="1", parent="0")

    for item in diagram.items:
        cell = SubElement(
            root,
            "mxCell",
            id=item.id,
            value=html.escape(item.label),
            style="rounded=1;whiteSpace=wrap;html=1;",
            vertex="1",
            parent="1",
        )
        geometry = SubElement(
            cell,
            "mxGeometry",
            x=str(item.bbox.x),
            y=str(item.bbox.y),
            width=str(item.bbox.width),
            height=str(item.bbox.height),
        )
        geometry.set("as", "geometry")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tostring(mxfile, encoding="utf-8", xml_declaration=True))
    return path
