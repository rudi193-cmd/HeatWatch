from __future__ import annotations

import numpy as np

# Tolerance for floating-point error when converting a geographic offset into an
# integer pixel index (see clip_array_to_bounds). Far smaller than one pixel,
# large enough to absorb IEEE-754 rounding of products like 0.8 * 10.
_PIXEL_EPS = 1e-9


def bounds_to_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> dict[str, float]:
    """Convert bounding box coordinates to a named dictionary.

    Args:
        min_lon: Western longitude boundary in decimal degrees.
        min_lat: Southern latitude boundary in decimal degrees.
        max_lon: Eastern longitude boundary in decimal degrees.
        max_lat: Northern latitude boundary in decimal degrees.

    Returns:
        Dictionary with keys: min_lon, min_lat, max_lon, max_lat.

    Raises:
        ValueError: If min_lon >= max_lon or min_lat >= max_lat.
    """
    if min_lon >= max_lon:
        raise ValueError(f"min_lon ({min_lon}) must be less than max_lon ({max_lon})")
    if min_lat >= max_lat:
        raise ValueError(f"min_lat ({min_lat}) must be less than max_lat ({max_lat})")
    return {"min_lon": min_lon, "min_lat": min_lat, "max_lon": max_lon, "max_lat": max_lat}


def pixel_to_coords(
    row: int,
    col: int,
    transform: tuple[float, float, float, float, float, float],
) -> tuple[float, float]:
    """Convert raster pixel indices to geographic coordinates.

    Uses an affine transform tuple in GDAL order:
        (x_origin, pixel_width, row_rotation, y_origin, col_rotation, pixel_height)

    Args:
        row: Row index (0-based) in the raster.
        col: Column index (0-based) in the raster.
        transform: Six-element affine transform tuple in GDAL convention.

    Returns:
        Tuple of (longitude, latitude) in decimal degrees.
    """
    x_origin, pixel_width, _, y_origin, _, pixel_height = transform
    lon = x_origin + col * pixel_width
    lat = y_origin + row * pixel_height
    return lon, lat


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Compute the great-circle distance between two points on Earth.

    Uses the haversine formula. Accurate for distances up to ~20,000 km.

    Args:
        lat1: Latitude of point 1 in decimal degrees.
        lon1: Longitude of point 1 in decimal degrees.
        lat2: Latitude of point 2 in decimal degrees.
        lon2: Longitude of point 2 in decimal degrees.

    Returns:
        Distance in kilometers.
    """
    R = 6371.0  # Earth's mean radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def clip_array_to_bounds(
    array: np.ndarray,
    array_bounds: dict[str, float],
    clip_bounds: dict[str, float],
) -> np.ndarray:
    """Return the subset of a raster array that falls within clip_bounds.

    Both bounds dicts must have keys: min_lon, min_lat, max_lon, max_lat.
    Assumes the array covers array_bounds uniformly (one value per pixel).

    Args:
        array: 2-D numpy array representing the raster.
        array_bounds: Geographic extent of the full array.
        clip_bounds: Geographic extent to clip to.

    Returns:
        Clipped 2-D numpy array. Returns empty array if bounds do not overlap.
    """
    rows, cols = array.shape
    lon_range = array_bounds["max_lon"] - array_bounds["min_lon"]
    lat_range = array_bounds["max_lat"] - array_bounds["min_lat"]

    # Snap to the nearest pixel within a small tolerance before truncating.
    # Plain int() truncation turns a mathematically-integer index that floating
    # point stores as e.g. 7.999999999 into 7, shifting the clip window by a
    # whole pixel. _PIXEL_EPS absorbs that representation error.
    def _index(value: float) -> int:
        return int(value + _PIXEL_EPS)

    col_start = _index((clip_bounds["min_lon"] - array_bounds["min_lon"]) / lon_range * cols)
    col_end = _index((clip_bounds["max_lon"] - array_bounds["min_lon"]) / lon_range * cols)
    row_start = _index((array_bounds["max_lat"] - clip_bounds["max_lat"]) / lat_range * rows)
    row_end = _index((array_bounds["max_lat"] - clip_bounds["min_lat"]) / lat_range * rows)

    col_start = max(0, col_start)
    col_end = min(cols, col_end)
    row_start = max(0, row_start)
    row_end = min(rows, row_end)

    if col_start >= col_end or row_start >= row_end:
        return np.empty((0, 0), dtype=array.dtype)

    return array[row_start:row_end, col_start:col_end]
