import numpy as np
import pytest
from pathlib import Path
from src.analysis.map_export import (
    ndvi_to_color,
    build_ndvi_legend_html,
    export_ndvi_map,
    NDVI_LEGEND,
    NODATA_COLOR,
)


def test_ndvi_to_color_nan_is_nodata():
    # No-data pixels must not be colored as dense vegetation.
    assert ndvi_to_color(float("nan")) == NODATA_COLOR
    assert ndvi_to_color(np.float32("nan")) == NODATA_COLOR


def test_ndvi_to_color_water():
    assert ndvi_to_color(-0.5) == "#4575b4"


def test_ndvi_to_color_urban():
    assert ndvi_to_color(0.05) == "#d73027"


def test_ndvi_to_color_dense():
    assert ndvi_to_color(0.8) == "#1a9850"


def test_ndvi_to_color_upper_bound():
    assert ndvi_to_color(1.0) == "#1a9850"


def test_ndvi_to_color_all_breakpoints():
    for entry in NDVI_LEGEND:
        color = ndvi_to_color(entry["min"])
        assert color == entry["color"]


def test_legend_html_contains_all_labels():
    html = build_ndvi_legend_html()
    for entry in NDVI_LEGEND:
        assert entry["label"] in html
        assert entry["color"] in html


def test_legend_html_is_string():
    assert isinstance(build_ndvi_legend_html(), str)


def test_export_ndvi_map_creates_file(tmp_path):
    ndvi = np.random.default_rng(0).uniform(-0.1, 0.9, size=(20, 20)).astype(np.float32)
    out = export_ndvi_map(ndvi, tmp_path / "test_map.html", city="Detroit")
    assert out.exists()
    assert out.suffix == ".html"


def test_export_ndvi_map_contains_legend(tmp_path):
    ndvi = np.full((5, 5), 0.5, dtype=np.float32)
    out = export_ndvi_map(ndvi, tmp_path / "map.html", city="Phoenix")
    content = out.read_text()
    assert "NDVI" in content
    assert "Dense Vegetation" in content
    assert "#91cf60" in content


def test_export_ndvi_map_handles_nan(tmp_path):
    ndvi = np.full((5, 5), np.nan, dtype=np.float32)
    out = export_ndvi_map(ndvi, tmp_path / "nan_map.html", city="Test")
    content = out.read_text()
    assert "#999999" in content
