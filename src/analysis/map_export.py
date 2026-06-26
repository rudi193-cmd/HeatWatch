from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Color used for pixels with no valid NDVI value (NaN / no-data).
NODATA_COLOR = "#999999"

# NDVI color legend breakpoints and their corresponding colors
# Based on standard remote sensing classification
NDVI_LEGEND = [
    {"label": "Water / No Data",   "min": -1.0, "max": 0.0,  "color": "#4575b4"},
    {"label": "Bare Soil / Urban", "min":  0.0, "max": 0.15, "color": "#d73027"},
    {"label": "Sparse Vegetation", "min": 0.15, "max": 0.30, "color": "#fc8d59"},
    {"label": "Moderate Cover",    "min": 0.30, "max": 0.50, "color": "#fee090"},
    {"label": "Dense Vegetation",  "min": 0.50, "max": 0.70, "color": "#91cf60"},
    {"label": "Very Dense Canopy", "min": 0.70, "max": 1.0,  "color": "#1a9850"},
]


def ndvi_to_color(value: float) -> str:
    """Map a single NDVI value to its legend hex color.

    Args:
        value: NDVI value in range [-1, 1].

    Returns:
        Hex color string (e.g. '#91cf60'). NaN / no-data values return
        NODATA_COLOR rather than being misclassified as dense vegetation.
    """
    # NaN compares False against every breakpoint, so guard it explicitly;
    # otherwise no-data pixels fall through to the dense-canopy color.
    if value != value:
        return NODATA_COLOR
    for entry in NDVI_LEGEND:
        if entry["min"] <= value < entry["max"]:
            return entry["color"]
    # Catch upper bound of 1.0
    return NDVI_LEGEND[-1]["color"]


def build_ndvi_legend_html() -> str:
    """Generate an HTML snippet for the NDVI color legend.

    Returns:
        Self-contained HTML string suitable for embedding in a folium map
        or any HTML document.
    """
    rows = ""
    for entry in NDVI_LEGEND:
        rows += (
            f'<tr>'
            f'<td style="background:{entry["color"]};width:24px;height:16px;'
            f'border:1px solid #ccc;"></td>'
            f'<td style="padding:2px 6px;font-size:12px;">{entry["label"]}</td>'
            f'<td style="padding:2px 6px;font-size:11px;color:#666;">'
            f'{entry["min"]:.2f} – {entry["max"]:.2f}</td>'
            f'</tr>'
        )
    return f"""
<div style="
    position:absolute;
    bottom:30px;
    right:10px;
    z-index:1000;
    background:white;
    padding:10px;
    border-radius:6px;
    border:1px solid #ccc;
    font-family:sans-serif;
    box-shadow:2px 2px 6px rgba(0,0,0,0.2);
">
  <b style="font-size:13px;">NDVI</b>
  <table style="border-collapse:collapse;margin-top:6px;">
    {rows}
  </table>
</div>
"""


def export_ndvi_map(
    ndvi: np.ndarray,
    output_path: Path,
    city: str = "Unknown",
) -> Path:
    """Export an NDVI array as a standalone HTML map with color legend.

    Generates a simple HTML grid visualization with an embedded NDVI
    color legend. For production use, replace the grid with a folium
    choropleth layer.

    Args:
        ndvi: 2-D array of NDVI values in range [-1, 1].
        output_path: Path to write the HTML file.
        city: City name shown in the map title.

    Returns:
        Path to the written HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows, cols = ndvi.shape
    cell_size = max(2, min(8, 400 // max(rows, cols)))

    grid_html = ""
    for r in range(rows):
        grid_html += "<tr>"
        for c in range(cols):
            val = float(ndvi[r, c])
            color = NODATA_COLOR if np.isnan(val) else ndvi_to_color(val)
            grid_html += (
                f'<td style="background:{color};width:{cell_size}px;'
                f'height:{cell_size}px;" title="NDVI={val:.3f}"></td>'
            )
        grid_html += "</tr>"

    legend = build_ndvi_legend_html()

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>HeatWatch NDVI Map - {city}</title>
  <style>
    body {{ margin: 0; font-family: sans-serif; background: #1a1a1a; color: white; }}
    h2 {{ padding: 12px 16px; margin: 0; font-size: 16px; }}
    #map-container {{ position: relative; display: inline-block; margin: 8px 16px; }}
    table {{ border-collapse: collapse; }}
  </style>
</head>
<body>
  <h2>HeatWatch - NDVI Map: {city}</h2>
  <div id="map-container">
    <table>{grid_html}</table>
    {legend}
  </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
