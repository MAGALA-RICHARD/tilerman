# geotiff-tiler

Installable **Python library** for window-based tiling of georeferenced GeoTIFFs. No UI required. Can be imported into PrairiePro, notebooks or custom pipelines.

## Installation

```bash
pip install .
# For editable development:
pip install -e '.[test]'
```

## Python API

```python
from geotiff_tiler import tile_by_pixels, tile_by_distance, Config, preview_grid, run

# 512x512 pixel cores, 32 extra pixels on each available side
tiles = tile_by_pixels('orthophoto.tif', 'tiles', 512, 512, overlap=32, edges='keep', compress='deflate')

# 100x100 meter cores, 5 meter halo on each side
tiles = tile_by_distance('orthophoto.tif', 'tiles_100m', 100, 100, units='m', overlap=5, overlap_units='m')

cfg = Config(source='orthophoto.tif', destination='tiles', width=256, height=256,
             naming='Tile_{index:04d}.tif', folders='row')
print(preview_grid(cfg))  # no files written
run(cfg, progress=lambda completed, total, eta_seconds: print(completed, total, eta_seconds))
```

`Config` uses `source` and `destination` paths (strings or `Path`). Other arguments: `mode='pixels'|'distance'`, `units='m'|'ft'`, `overlap=0`, `overlap_units='pixels'|'map'|'m'|'ft'`, `edges='keep'|'pad'|'discard'`, `pad_value=None`, `compress='source'|'none'|'lzw'|'deflate'|'zstd'`, `naming='Tile_Row{row:03d}_Col{col:03d}.tif'`, `folders='single'|'row'|'grid'`, `dtype='source'`. Naming fields: `{index}`, `{row}`, `{col}`, `{x}` and `{y}` (x/y are core offsets in pixels, not map coordinates).

The overlap is a **halo** around every tile's core; it does not reduce the spacing between tile cores. In `pad` mode, halo regions beyond source bounds are padded and masked. Source data is never resampled or reprojected. `discard` omits incomplete **core** tiles at source edges. The output is a list of tile file paths; for very large rasters this list may be large.

## Optional CLI

```bash
geotiff-tile orthophoto.tif output --mode pixels --width 512 --height 512 --overlap 32
geotiff-tile orthophoto.tif output --mode distance --width 100 --height 100 --units m --preview
```

## Accuracy and limitations

* Reads only the required raster windows; writes exact samples in the source dtype, retaining affine pixel alignment and CRS. The translated affine transform describes each tile's **new origin**; original full-raster transform cannot be copied verbatim.
* Copies dataset/band tags, band descriptions and units, color interpretation, palettes, scales/offsets, and supported masks. The driver stores supported georeferencing tags when writing.
* Source-wide statistics/histograms are deliberately **not copied** because they would be invalid on individual tiles; recompute if needed.
* Projected CRS required for physical-distance tiling. Pixel mode works for geographic CRS too. For rotated/skewed grids, distance is measured along raster pixel axes, so tile world-space bounding boxes are not axis-aligned rectangles.
* Explicit GCP/RPC-only georeferencing is rejected; lossless output is not guaranteed for specialized TIFF private tags or all TIFF profile details.
* Source JPEG/WEBP compression needs explicit lossless output compression, otherwise source mode refuses it. Disk-space validation is approximate. Pre-existing output files are not overwritten.
* `pad_value` must be representable in the source dtype. For a single-file GeoTIFF, all bands have the same dtype.

## Tests

```bash
pytest -q
```
