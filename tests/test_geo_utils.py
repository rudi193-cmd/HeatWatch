import numpy as np
import pytest

from src.utils.geo_utils import (
    bounds_to_bbox,
    clip_array_to_bounds,
    haversine_distance,
    pixel_to_coords,
)


# --- bounds_to_bbox ---------------------------------------------------------

def test_bounds_to_bbox_returns_named_dict():
    bbox = bounds_to_bbox(-100.0, 40.0, -99.0, 41.0)
    assert bbox == {
        "min_lon": -100.0,
        "min_lat": 40.0,
        "max_lon": -99.0,
        "max_lat": 41.0,
    }


def test_bounds_to_bbox_rejects_inverted_longitude():
    with pytest.raises(ValueError):
        bounds_to_bbox(-99.0, 40.0, -100.0, 41.0)


def test_bounds_to_bbox_rejects_inverted_latitude():
    with pytest.raises(ValueError):
        bounds_to_bbox(-100.0, 41.0, -99.0, 40.0)


def test_bounds_to_bbox_rejects_degenerate_extent():
    # Equal bounds have zero extent and must be rejected.
    with pytest.raises(ValueError):
        bounds_to_bbox(-100.0, 40.0, -100.0, 41.0)


# --- pixel_to_coords --------------------------------------------------------

def test_pixel_to_coords_origin():
    # GDAL affine: (x_origin, pixel_width, row_rot, y_origin, col_rot, pixel_height)
    transform = (-100.0, 0.01, 0.0, 41.0, 0.0, -0.01)
    lon, lat = pixel_to_coords(0, 0, transform)
    assert lon == pytest.approx(-100.0)
    assert lat == pytest.approx(41.0)


def test_pixel_to_coords_offset():
    transform = (-100.0, 0.01, 0.0, 41.0, 0.0, -0.01)
    lon, lat = pixel_to_coords(10, 20, transform)
    # col moves east by 20 * 0.01, row moves south by 10 * 0.01.
    assert lon == pytest.approx(-99.8)
    assert lat == pytest.approx(40.9)


# --- haversine_distance -----------------------------------------------------

def test_haversine_zero_distance():
    assert haversine_distance(40.0, -100.0, 40.0, -100.0) == pytest.approx(0.0)


def test_haversine_one_degree_latitude():
    # One degree of latitude is ~111.19 km on a sphere of radius 6371 km.
    d = haversine_distance(40.0, -100.0, 41.0, -100.0)
    assert d == pytest.approx(111.19, abs=0.1)


def test_haversine_known_city_pair():
    # New York City to Los Angeles great-circle distance is ~3936 km.
    d = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3900.0 < d < 3970.0


def test_haversine_is_symmetric():
    a = haversine_distance(19.4326, -99.1332, 18.4655, -66.1057)
    b = haversine_distance(18.4655, -66.1057, 19.4326, -99.1332)
    assert a == pytest.approx(b)


# --- clip_array_to_bounds ---------------------------------------------------

def _grid(rows=10, cols=10):
    return np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)


def test_clip_interior_window():
    array = _grid(10, 10)
    array_bounds = bounds_to_bbox(-100.0, 40.0, -99.0, 41.0)
    clip_bounds = bounds_to_bbox(-99.5, 40.3, -99.2, 40.7)
    clipped = clip_array_to_bounds(array, array_bounds, clip_bounds)
    # lon 0.5..0.8 -> cols 5..8 ; lat (top-down) 0.3..0.7 -> rows 3..7
    assert clipped.shape == (4, 3)
    np.testing.assert_array_equal(clipped, array[3:7, 5:8])


def test_clip_clamps_to_array_extent():
    array = _grid(10, 10)
    array_bounds = bounds_to_bbox(-100.0, 40.0, -99.0, 41.0)
    # clip request larger than the array on every side
    clip_bounds = bounds_to_bbox(-200.0, 0.0, 0.0, 90.0)
    clipped = clip_array_to_bounds(array, array_bounds, clip_bounds)
    assert clipped.shape == array.shape


def test_clip_non_overlapping_returns_empty():
    array = _grid(10, 10)
    array_bounds = bounds_to_bbox(-100.0, 40.0, -99.0, 41.0)
    clip_bounds = bounds_to_bbox(10.0, 10.0, 11.0, 11.0)
    clipped = clip_array_to_bounds(array, array_bounds, clip_bounds)
    assert clipped.size == 0
