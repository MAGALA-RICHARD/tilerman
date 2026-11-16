"""Lossless, window-based GeoTIFF tiling engine and optional Tkinter interface.

Requires rasterio and numpy. Run `python geotiff_tiler.py --help` or
`python geotiff_tiler.py --gui`. A tile width/height describes the non-overlap
core; overlap is an additional halo on each available side.
"""
from __future__ import annotations

import math
import shutil
import string
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import rasterio
from rasterio.enums import MaskFlags
from rasterio.windows import Window, transform as window_transform


@dataclass(frozen=True)
class Config:
    source: Path
    destination: Path
    width: float = 512
    height: float = 512
    mode: str = 'pixels'  # pixels | distance
    units: str = 'm'  # m | ft (only for distance mode)
    overlap: float = 0
    overlap_units: str = 'pixels'  # pixels | map | m | ft
    edges: str = 'keep'  # keep | pad | discard
    pad_value: float | None = None  # None = dataset nodata, or 0 with mask
    compress: str = 'source'  # source | none | lzw | deflate | zstd
    naming: str = 'Tile_Row{row:03d}_Col{col:03d}.tif'
    folders: str = 'single'  # single | row | grid
    dtype: str = 'source'  # strict lossless default; alternative disallowed

    def __post_init__(self):
        object.__setattr__(self, 'source', Path(self.source))
        object.__setattr__(self, 'destination', Path(self.destination))


@dataclass(frozen=True)
class Tile:
    index: int
    row: int
    col: int
    core: Window
    window: Window  # potentially outside image for pad mode
    valid: Window  # intersection with source image


def _posint(value: float, label: str) -> int:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be a finite positive number')
    n = int(round(value))
    if n < 1 or not math.isclose(n, value, abs_tol=1e-9):
        raise ValueError(f'{label} must be a positive whole number of pixels')
    return n


def _map_meters_per_unit(ds) -> float:
    if ds.crs is None or not ds.crs.is_projected:
        raise ValueError('Distance-based tiles require a projected CRS with linear units; reproject separately if desired')
    factor = ds.crs.linear_units_factor[1]
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('Cannot determine projected CRS linear unit conversion')
    return factor


def dimensions(ds, cfg: Config) -> tuple[int, int, int, int]:
    if cfg.mode not in ('pixels', 'distance'):
        raise ValueError('Mode must be pixels or distance')
    if cfg.mode == 'pixels':
        w, h = _posint(cfg.width, 'Tile width'), _posint(cfg.height, 'Tile height')
    else:
        if cfg.units not in ('m', 'ft'):
            raise ValueError('Distance units must be m or ft')
        meters_per_map_unit = _map_meters_per_unit(ds)
        factor = 1 if cfg.units == 'm' else 0.3048
        # Norms support rotated and skewed affine transforms without resampling.
        px_x = math.hypot(ds.transform.a, ds.transform.d) * meters_per_map_unit
        px_y = math.hypot(ds.transform.b, ds.transform.e) * meters_per_map_unit
        if min(px_x, px_y) <= 0:
            raise ValueError('Invalid affine pixel resolution')
        w = max(1, math.ceil(cfg.width * factor / px_x)) if cfg.width > 0 else 0
        h = max(1, math.ceil(cfg.height * factor / px_y)) if cfg.height > 0 else 0
        if not w or not h or not all(math.isfinite(x) for x in (cfg.width, cfg.height)):
            raise ValueError('Tile dimensions must be finite positive numbers')
    if not math.isfinite(cfg.overlap) or cfg.overlap < 0:
        raise ValueError('Overlap must be finite and nonnegative')
    if cfg.overlap_units == 'pixels':
        halo = _posint(cfg.overlap, 'Overlap') if cfg.overlap else 0
        ox = oy = halo
    elif cfg.overlap_units in ('map', 'm', 'ft'):
        # Distance overlap is measured along raster pixel axes, even when rotated.
        unit_multiplier = (1 if cfg.overlap_units == 'map' else
                           (1 if cfg.overlap_units == 'm' else 0.3048) / _map_meters_per_unit(ds))
        ox = math.ceil(cfg.overlap * unit_multiplier / math.hypot(ds.transform.a, ds.transform.d))
        oy = math.ceil(cfg.overlap * unit_multiplier / math.hypot(ds.transform.b, ds.transform.e))
    else:
        raise ValueError('Overlap units must be pixels, map, m or ft')
    if ox >= w or oy >= h:
        raise ValueError('Overlap on each side must be smaller than tile width/height')
    if cfg.edges not in ('keep', 'pad', 'discard'):
        raise ValueError('Edge handling must be keep, pad or discard')
    if cfg.compress.lower() not in ('source', 'none', 'lzw', 'deflate', 'zstd'):
        raise ValueError('Use a lossless compression: source, none, lzw, deflate, zstd')
    if cfg.dtype != 'source':
        raise ValueError('Changing dtype conflicts with exact pixel preservation; choose source')
    if cfg.folders not in ('single', 'row', 'grid'):
        raise ValueError('Folder organization must be single, row or grid')
    return w, h, ox, oy


def tiles(ds, cfg: Config) -> Iterator[Tile]:
    w, h, ox, oy = dimensions(ds, cfg)
    index = 0
    for row, y in enumerate(range(0, ds.height, h), 1):
        for col, x in enumerate(range(0, ds.width, w), 1):
            cw, ch = min(w, ds.width - x), min(h, ds.height - y)
            if cfg.edges == 'discard' and (cw < w or ch < h):
                continue
            index += 1
            core = Window(x, y, cw, ch)
            left, top = x - ox, y - oy
            right, bottom = x + w + ox, y + h + oy
            if cfg.edges != 'pad':
                left, top = max(0, left), max(0, top)
                right, bottom = min(ds.width, right), min(ds.height, bottom)
            win = Window(left, top, right-left, bottom-top)
            vx, vy = max(0, left), max(0, top)
            valid = Window(vx, vy, max(0, min(ds.width, right)-vx), max(0, min(ds.height, bottom)-vy))
            yield Tile(index, row, col, core, win, valid)


def tile_count(ds, cfg: Config) -> int:
    w, h, _, _ = dimensions(ds, cfg)
    if cfg.edges == 'discard':
        return (ds.width // w) * (ds.height // h)
    return math.ceil(ds.width/w) * math.ceil(ds.height/h)


def _output_path(cfg: Config, tile: Tile) -> Path:
    fields = {'index':tile.index, 'row':tile.row, 'col':tile.col,
              'x':int(tile.core.col_off), 'y':int(tile.core.row_off)}
    try:
        parts = {name for _, name, _, _ in string.Formatter().parse(cfg.naming) if name}
        if not parts.issubset(fields):
            raise ValueError(f'Unknown naming fields: {parts - fields.keys()}')
        filename = cfg.naming.format(**fields)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f'Invalid naming template: {exc}') from exc
    if Path(filename).name != filename or not filename.lower().endswith(('.tif', '.tiff')):
        raise ValueError('Naming template must produce a .tif/.tiff filename without directories')
    folder = cfg.destination
    if cfg.folders in ('row','grid'):
        folder = folder / f'Row{tile.row:03d}'
    if cfg.folders == 'grid':
        folder = folder / f'Col{tile.col:03d}'
    return folder / filename


def _copy_metadata(src, dst, tile: Tile) -> None:
    # Copy metadata whose meaning remains correct at subdataset scale.
    excluded = {'IMAGE_STRUCTURE', 'DERIVED_SUBDATASETS'}
    for ns in ('', *src.tag_namespaces()):
        if ns in excluded:
            continue
        tags = src.tags(ns=ns)
        tags = {k:v for k,v in tags.items() if not k.upper().startswith(('STATISTICS_', 'HISTOGRAM_'))}
        if tags:
            dst.update_tags(ns=ns, **tags)
    for b in range(1, src.count + 1):
        for ns in ('', *src.tag_namespaces(b)):
            if ns in excluded:
                continue
            tags = {k:v for k,v in src.tags(b, ns=ns).items()
                    if not k.upper().startswith(('STATISTICS_', 'HISTOGRAM_'))}
            if tags:
                dst.update_tags(b, ns=ns, **tags)
        if src.descriptions[b-1]:
            dst.set_band_description(b, src.descriptions[b-1])
        if src.units[b-1]:
            dst.set_band_unit(b, src.units[b-1])
        try:
            cm = src.colormap(b)
        except ValueError:
            cm = None
        if cm:
            dst.write_colormap(b, cm)
    dst.colorinterp = src.colorinterp
    dst.scales = src.scales
    dst.offsets = src.offsets
    # Dataset statistics are intentionally excluded: source-wide values are
    # generally invalid for individual tiles. Users may recompute them later.


def _profile(src, cfg: Config, tile: Tile) -> dict:
    p = src.profile.copy()
    p.update(driver='GTiff', width=int(tile.window.width), height=int(tile.window.height),
             transform=window_transform(tile.window, src.transform), dtype=src.dtypes[0])
    # GDAL chooses safe block geometry, rather than inheriting oversized blocks.
    p.pop('blockxsize', None)
    p.pop('blockysize', None)
    p.pop('tiled', None)
    for k in ('jpeg_quality', 'webp_level', 'predictor', 'zlevel', 'zstd_level'):
        p.pop(k, None)
    if cfg.compress != 'source':
        p.pop('compress', None)
        if cfg.compress != 'none':
            p['compress'] = cfg.compress.upper()
    return p


def _check_src(src, cfg: Config) -> None:
    if src.driver != 'GTiff':
        raise ValueError('Input must be a GeoTIFF')
    if not src.crs:
        raise ValueError('Input has no CRS; geospatial integrity cannot be verified')
    if src.transform.is_identity:
        raise ValueError('Input has no meaningful affine geotransform')
    if len(set(src.dtypes)) > 1:
        raise ValueError('Mixed per-band dtypes are not supported')
    # Avoid silently dropping advanced models that require special handling.
    if src.gcps[0] or src.rpcs:
        raise ValueError('GCP/RPC georeferencing is not supported by affine-only tiling')
    if src.profile.get('compress', '').lower() in ('jpeg','webp') and cfg.compress == 'source':
        raise ValueError('Lossy source compression cannot be preserved losslessly; select LZW/DEFLATE')
    dimensions(src, cfg)


def run(cfg: Config, progress: Callable[[int,int,float],None] | None = None,
        cancel: threading.Event | None = None) -> list[Path]:
    """Write tiles with original sample values; returns completed paths.

    No automatic deletion on cancellation. Never rewrites existing files.
    """
    if not cfg.source.is_file():
        raise FileNotFoundError(f'Source not found: {cfg.source}')
    cfg.destination.mkdir(parents=True, exist_ok=True)
    if cfg.source.resolve().parent == cfg.destination.resolve() and cfg.naming == cfg.source.name:
        raise ValueError('Output filename would overwrite input')
    produced: list[Path] = []
    start = time.monotonic()
    with rasterio.open(cfg.source) as src:
        _check_src(src, cfg)
        total = tile_count(src, cfg)
        # Disk estimate is conservative, not a guarantee (compression varies).
        capacity = shutil.disk_usage(cfg.destination).free
        minimum = min(total, 1) * min(src.width, int(cfg.width) if cfg.mode == 'pixels' else src.width) * src.count * np.dtype(src.dtypes[0]).itemsize
        if capacity < minimum:
            raise OSError('Insufficient free space even for one uncompressed tile')
        for tile in tiles(src, cfg):
            if cancel is not None and cancel.is_set():
                break
            target = _output_path(cfg, tile)
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite existing tile: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            profile = _profile(src, cfg, tile)
            fill = cfg.pad_value if cfg.pad_value is not None else src.nodata
            if fill is None:
                fill = 0
            if not np.can_cast(np.asarray(fill).dtype, np.dtype(src.dtypes[0]), casting='safe'):
                # numpy may infer int64 for a safely representable small integer.
                try:
                    cast_fill = np.asarray(fill).astype(src.dtypes[0]).item()
                    if cast_fill != fill:
                        raise ValueError()
                except (ValueError, OverflowError):
                    raise ValueError('Padding value cannot be represented exactly in source dtype')
            try:
                with rasterio.open(target, 'w', **profile) as dst:
                    # Boundless window read returns edge padding without resampling.
                    data = src.read(window=tile.window, boundless=True, fill_value=fill)
                    dst.write(data)
                    # Preserve an explicit per-dataset mask, or mark padding invalid.
                    explicit_mask = any(MaskFlags.per_dataset in flags for flags in src.mask_flag_enums)
                    if explicit_mask or cfg.edges == 'pad':
                        mask = src.dataset_mask(window=tile.window, boundless=True)
                        dst.write_mask(mask)
                    _copy_metadata(src, dst, tile)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            produced.append(target)
            if progress:
                elapsed = time.monotonic() - start
                eta = (elapsed / len(produced)) * (total - len(produced))
                progress(len(produced), total, eta)
    return produced



def tile_by_pixels(source: str | Path, destination: str | Path, width: int = 512,
                   height: int = 512, **kwargs) -> list[Path]:
    """Split a GeoTIFF using pixel-size tile cores; see Config for options."""
    return run(Config(Path(source), Path(destination), width, height, mode='pixels', **kwargs))


def tile_by_distance(source: str | Path, destination: str | Path, width: float,
                     height: float, units: str = 'm', **kwargs) -> list[Path]:
    """Split a projected GeoTIFF into ground-distance-based tile cores."""
    return run(Config(Path(source), Path(destination), width, height, mode='distance', units=units, **kwargs))


def preview_grid(cfg: Config) -> dict:
    """Return metadata and tile layout without writing raster files."""
    with rasterio.open(cfg.source) as src:
        _check_src(src, cfg)
        w, h, ox, oy = dimensions(src, cfg)
        return dict(count=tile_count(src, cfg), tile_pixels=(w, h), overlap_pixels=(ox, oy),
                    raster_pixels=(src.width, src.height), crs=str(src.crs),
                    resolution=src.res, bounds=tuple(src.bounds))
