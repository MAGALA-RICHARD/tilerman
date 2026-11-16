
"""Memory-bounded, pixel-exact assembly of affine-aligned GeoTIFF tiles."""

from __future__ import annotations

import math
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable
import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

from orthophoto import _copy_metadata


def merge_raster_tiles(
    tiles: Iterable[str | Path] | str | Path,
    output: str | Path,*,
    reference: str | Path | None = None,
    compress: str = "source",
    overwrite: bool = True,
    block_size: int = 512, ) -> Path:
    """Merge aligned GeoTIFF tiles without resampling or modifying valid samples.

    The first valid tile wins in overlapping regions. Invalid pixels never
    replace valid samples.

    If reference is supplied, the output retains its exact dimensions,
    transform, CRS, and band structure. Otherwise, the output covers the
    union of the tile extents, including any padding.

    The reference raster supplies grid information and metadata only.
    Its pixel values are not used to fill gaps.
    """
    output = Path(output)

    if isinstance(tiles, (str, Path)):
        folder = Path(tiles)
        if not folder.is_dir():
            raise ValueError("A single tiles argument must be a directory")
        paths = sorted([*folder.rglob("*.tif"), *folder.rglob("*.tiff")])
    else:
        paths = [Path(p) for p in tiles]

    if not paths:
        raise ValueError("No GeoTIFF tiles supplied")
    if block_size < 1:
        raise ValueError("block_size must be positive")

    compress = compress.lower()
    if compress not in ("source", "none", "lzw", "deflate", "zstd"):
        raise ValueError("Compression must be source, none, lzw, deflate, or zstd")

    if output.resolve() in {p.resolve() for p in paths}:
        raise ValueError("Output must not be one of the input tiles")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if reference is not None and output.resolve() == Path(reference).resolve():
        raise ValueError("Output must not overwrite the reference raster")

    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(p)) for p in paths]
        ref = stack.enter_context(rasterio.open(reference)) if reference is not None else sources[0]

        if ref.crs is None or ref.transform.is_identity:
            raise ValueError("Reference must have a CRS and valid affine transform")
        if ref.gcps[0] or ref.rpcs:
            raise ValueError("GCP/RPC reference grids are not supported")
        if any(not math.isfinite(v) for v in ref.transform[:6]):
            raise ValueError("Invalid reference geotransform")

        base = ref.transform
        positions = []

        for path, src in zip(paths, sources):
            if src.driver != "GTiff" or src.crs != ref.crs or src.count != ref.count or src.dtypes != ref.dtypes:
                raise ValueError(f"CRS/band count/dtype mismatch: {path}")
            if src.gcps[0] or src.rpcs:
                raise ValueError(f"GCP/RPC tile unsupported: {path}")

            if not np.allclose(
                [src.transform.a, src.transform.b, src.transform.d, src.transform.e],
                [base.a, base.b, base.d, base.e],
                rtol=0, atol=1e-10
            ):
                raise ValueError(f"Pixel grids differ (resampling prohibited): {path}")

            # Use @ for affine matrix multiplication.
            px, py = (~base) @ (src.transform.c, src.transform.f)
            ix, iy = round(px), round(py)

            if abs(px - ix) > 1e-5 or abs(py - iy) > 1e-5:
                raise ValueError(f"Tile is not pixel-aligned: {path}")

            positions.append((ix, iy))

        if reference is not None:
            min_x = min_y = 0
            width, height = ref.width, ref.height
        else:
            min_x = min(x for x, y in positions)
            min_y = min(y for x, y in positions)
            width = max(x + src.width for (x, y), src in zip(positions, sources)) - min_x
            height = max(y + src.height for (x, y), src in zip(positions, sources)) - min_y

        if width <= 0 or height <= 0:
            raise ValueError("Invalid mosaic extent")

        profile = ref.profile.copy()
        profile.update(
            driver="GTiff",
            width=width,
            height=height,
            transform=base @ Affine.translation(min_x, min_y)
        )

        for key in ("blockxsize", "blockysize", "tiled", "jpeg_quality", "webp_level"):
            profile.pop(key, None)

        if compress != "source":
            profile.pop("compress", None)
            if compress != "none":
                profile["compress"] = compress.upper()

        if str(profile.get("compress", "")).lower() in ("jpeg", "webp"):
            raise ValueError("Lossy compression is not supported for exact-value merging")

        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(output, "w+", **profile) as dst:
                    # Initially, all destination pixels are invalid.
                    dst.write_mask(False)

                    for (x, y), src in zip(positions, sources):
                        dx, dy = x - min_x, y - min_y
                        left, top = max(0, dx), max(0, dy)
                        right = min(width, dx + src.width)
                        bottom = min(height, dy + src.height)

                        if left >= right or top >= bottom:
                            continue

                        for row in range(top, bottom, block_size):
                            for col in range(left, right, block_size):
                                w = min(block_size, right - col)
                                h = min(block_size, bottom - row)

                                dst_win = Window(col, row, w, h)
                                src_win = Window(col - dx, row - dy, w, h)

                                incoming_valid = src.read_masks(window=src_win) > 0
                                current_valid = dst.read_masks(window=dst_win) > 0

                                # Preserve the first valid observation per pixel/band.
                                take = incoming_valid & ~current_valid

                                if not take.any():
                                    continue

                                old = dst.read(window=dst_win)
                                new = src.read(window=src_win)

                                np.copyto(old, new, where=take)
                                dst.write(old, window=dst_win)

                                # GeoTIFF dataset-wide mask: union of band validity.
                                new_mask = (current_valid | incoming_valid).any(axis=0)
                                dst.write_mask(
                                    (new_mask * 255).astype("uint8"),
                                    window=dst_win
                                )

                    _copy_metadata(ref, dst, None)

        except Exception:
            output.unlink(missing_ok=True)
            raise

    return output
