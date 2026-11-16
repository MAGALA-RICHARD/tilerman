
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.transform import from_origin
from rasterio.windows import Window, transform as window_transform

from orthophoto import tile_raster_by_pixels, tile_raster_by_distance
from merge_rasters import merge_raster_tiles


class TestRasterTilingAndMerging(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.height, self.width, self.bands = 23, 31, 5
        rows, cols = np.indices((self.height, self.width))
        self.data = np.stack([ (b + 1) * 10000 + rows * 100 + cols + 1 for b in range(self.bands)]).astype("uint16")

        self.transform = from_origin(500000, 4700000, 0.05, 0.05)
        self.original = self.write_raster("original.tif", self.data, self.transform)

    def write_raster(self, name, data, transform, nodata=0):
        path = self.root / name
        with rasterio.open(path, "w", driver="GTiff",
                           height=data.shape[1], width=data.shape[2],
                           count=data.shape[0], dtype=data.dtype,
                           crs="EPSG:32615", transform=transform,
                           nodata=nodata) as dst:
            dst.write(data)
        return path

    def split_by_pixels(self):
        folder = self.root / "pixel_tiles"
        folder.mkdir(exist_ok=True)
        result = tile_raster_by_pixels(
            source=self.original, destination=folder,
            width=12, height=10, overlap=3)
        return self.collect_tiles(folder, result)

    def split_by_distance(self):
        folder = self.root / "distance_tiles"
        folder.mkdir(exist_ok=True)
        result = tile_raster_by_distance(
            source=self.original, destination=folder,
            width=0.6, height=0.5, units="m",
            overlap=0.1, overlap_units="m" )
        return self.collect_tiles(folder, result)

    @staticmethod
    def collect_tiles(folder, result):
        if isinstance(result, (str, Path)):
            result = Path(result)
            if result.is_dir():
                paths = sorted([*result.rglob("*.tif"), *result.rglob("*.tiff")])
            else:
                paths = [result]
        elif result is not None:
            paths = [Path(p) for p in result]
        else:
            paths = sorted([*folder.rglob("*.tif"), *folder.rglob("*.tiff")])
        if not paths:
            raise AssertionError(f"No GeoTIFF tiles generated in {folder}")
        return paths

    def assert_tiles_match_original(self, tiles):
        with rasterio.open(self.original) as src:
            coverage = np.zeros((src.height, src.width), dtype=bool)

            for tile in tiles:
                with self.subTest(tile=tile.name), rasterio.open(tile) as dst:
                    self.assertEqual(dst.count, src.count)
                    self.assertEqual(dst.crs, src.crs)
                    self.assertEqual(dst.dtypes, src.dtypes)

                    x, y = (~src.transform) @ (dst.transform.c, dst.transform.f)
                    col, row = round(x), round(y)
                    self.assertAlmostEqual(x, col, places=5)
                    self.assertAlmostEqual(y, row, places=5)

                    self.assertEqual(
                        dst.transform,
                        src.transform @ Affine.translation(col, row) )

                    left, top = max(0, col), max(0, row)
                    right = min(src.width, col + dst.width)
                    bottom = min(src.height, row + dst.height)
                    self.assertGreater(right, left)
                    self.assertGreater(bottom, top)

                    w, h = right - left, bottom - top
                    src_win = Window(left, top, w, h)
                    tile_win = Window(left - col, top - row, w, h)

                    original_data = src.read(window=src_win)
                    tile_data = dst.read(window=tile_win)
                    original_mask = src.read_masks(window=src_win)
                    tile_mask = dst.read_masks(window=tile_win)

                    np.testing.assert_array_equal(
                        tile_mask, original_mask,
                        err_msg=f"Invalid pixel mask in {tile.name}")
                    valid = original_mask > 0
                    np.testing.assert_array_equal(
                        tile_data[valid], original_data[valid],
                        err_msg=f"Pixel values differ in {tile.name}")
                    coverage[top:bottom, left:right] = True

            self.assertTrue(coverage.all(), "Some original pixels were not covered by tiles")

    def assert_merge_matches_original(self, tiles, name):
        output = self.root / f"{name}_merged.tif"
        merged = merge_raster_tiles(
            tiles, output, reference=self.original, block_size=7)

        with rasterio.open(self.original) as src, rasterio.open(merged) as dst:
            self.assertEqual(dst.width, src.width)
            self.assertEqual(dst.height, src.height)
            self.assertEqual(dst.count, src.count)
            self.assertEqual(dst.crs, src.crs)
            self.assertEqual(dst.transform, src.transform)
            self.assertEqual(dst.dtypes, src.dtypes)
            self.assertEqual(dst.nodata, src.nodata)

            np.testing.assert_array_equal(
                dst.read_masks(), src.read_masks(),
                err_msg=f"{name}: merged masks differ")
            np.testing.assert_array_equal(
                dst.read(), src.read(),
                err_msg=f"{name}: merged pixels differ from original")

    def test_original_raster_size_and_values(self):
        with rasterio.open(self.original) as src:
            self.assertEqual((src.height, src.width, src.count),
                             (self.height, self.width, self.bands))
            np.testing.assert_array_equal(src.read(), self.data)

    def test_tile_by_pixels_and_merge(self):
        tiles = self.split_by_pixels()
        self.assertGreater(len(tiles), 1)
        self.assert_tiles_match_original(tiles)
        self.assert_merge_matches_original(tiles, "pixels")

    def test_tile_by_distance_and_merge(self):
        tiles = self.split_by_distance()
        self.assertGreater(len(tiles), 1)
        self.assert_tiles_match_original(tiles)
        self.assert_merge_matches_original(tiles, "distance")

    def test_overlapping_pixels_not_duplicated(self):
        transform = from_origin(500000, 4700000, 1, 1)
        original = self.write_raster(
            "overlap_original.tif", np.ones((1, 4, 6), dtype="uint16"), transform)
        first = np.full((1, 4, 4), 11, dtype="uint16")
        second = np.full((1, 4, 4), 99, dtype="uint16")
        first[0, 1, 3] = 0

        a = self.write_raster("overlap_a.tif", first, transform)
        b = self.write_raster(
            "overlap_b.tif", second, transform @ Affine.translation(2, 0))
        merged = merge_raster_tiles(
            [a, b], self.root / "overlap_merged.tif",
            reference=original, block_size=2 )

        expected = np.full((4, 6), 99, dtype="uint16")
        expected[:, :4] = 11
        expected[1, 3] = 99

        with rasterio.open(merged) as dst:
            self.assertEqual((dst.height, dst.width), (4, 6))
            self.assertEqual(dst.read(1).size, 24)
            np.testing.assert_array_equal(dst.read(1), expected)
            self.assertEqual(dst.read(1)[0, 3], 11)
            self.assertEqual(dst.read(1)[1, 3], 99)
            np.testing.assert_array_equal(
                dst.read_masks(1), np.full((4, 6), 255, dtype="uint8"))

    def test_merge_without_reference(self):
        transform = from_origin(500000, 4700000, 1, 1)
        a = self.write_raster(
            "union_a.tif", np.full((1, 3, 4), 7, dtype="uint16"), transform)
        b = self.write_raster(
            "union_b.tif", np.full((1, 3, 4), 9, dtype="uint16"),
            transform @ Affine.translation(2, 0) )
        merged = merge_raster_tiles([a, b], self.root / "union.tif")

        with rasterio.open(merged) as dst:
            self.assertEqual((dst.height, dst.width), (3, 6))
            self.assertEqual(dst.transform, transform)
            np.testing.assert_array_equal(
                dst.read(1), np.tile([7, 7, 7, 7, 9, 9], (3, 1)))

    def test_rejects_misaligned_tiles(self):
        transform = from_origin(500000, 4700000, 1, 1)
        a = self.write_raster("aligned.tif", np.ones((1, 3, 3), dtype="uint16"), transform)
        b = self.write_raster(
            "misaligned.tif", np.ones((1, 3, 3), dtype="uint16"),
            transform @ Affine.translation(0.5, 0))
        with self.assertRaisesRegex(ValueError, "not pixel-aligned"):
            merge_raster_tiles([a, b], self.root / "invalid.tif")

    def test_rejects_overwriting_input(self):
        with self.assertRaisesRegex(ValueError, "Output must not be one of the input tiles"):
            merge_raster_tiles([self.original], self.original)

    def test_rejects_invalid_block_size(self):
        with self.assertRaisesRegex(ValueError, "block_size must be positive"):
            merge_raster_tiles([self.original], self.root / "invalid_block.tif", block_size=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
