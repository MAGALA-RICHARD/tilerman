"""Orthophoto tile integrity tests using only the standard unittest framework."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import transform as window_transform

from src.geotiff_tiler import Config, dimensions, preview_grid, run, tile_by_distance, tile_by_pixels, tiles


class TestOrthphotoTile(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'orthophoto.tif'
        self.pixels = np.arange(5 * 7 * 11, dtype=np.uint16).reshape(5, 7, 11) + 100
        with rasterio.open(self.source, 'w', driver='GTiff', height=7, width=11,
                           count=5, dtype='uint16', crs='EPSG:32615',
                           transform=from_origin(500000, 4600000, 0.2, 0.2),
                           nodata=65535, compress='lzw') as ds:
            ds.write(self.pixels)
            ds.descriptions = ('Blue', 'Green', 'Red', 'Red edge', 'NIR')
            ds.update_tags(acquisition='2026-10-01', sensor='multispectral')
            for band in range(1, 6):
                ds.update_tags(band, wavelength_nm=str((475, 560, 668, 717, 840)[band - 1]))

    def assert_tile_matches_source(self, cfg):
        with rasterio.open(self.source) as src:
            expected = list(tiles(src, cfg))
            outputs = run(cfg)
            self.assertEqual(len(outputs), len(expected))
            for tile, filename in zip(expected, outputs):
                with rasterio.open(filename) as actual:
                    self.assertEqual(actual.crs, src.crs)
                    self.assertEqual(actual.transform, window_transform(tile.window, src.transform))
                    self.assertEqual(actual.res, src.res)
                    self.assertEqual(actual.count, src.count)
                    self.assertEqual(actual.dtypes, src.dtypes)
                    self.assertEqual(actual.nodata, src.nodata)
                    self.assertEqual(actual.descriptions, src.descriptions)
                    self.assertEqual(actual.tags()['acquisition'], '2026-10-01')
                    self.assertEqual(actual.tags()['sensor'], 'multispectral')
                    for band in range(1, 6):
                        self.assertEqual(actual.tags(band)['wavelength_nm'], src.tags(band)['wavelength_nm'])
                    self.assertEqual(actual.profile.get('compress'), src.profile.get('compress'))
                    x, y = int(tile.valid.col_off), int(tile.valid.row_off)
                    w, h = int(tile.valid.width), int(tile.valid.height)
                    dx, dy = int(x - tile.window.col_off), int(y - tile.window.row_off)
                    np.testing.assert_array_equal(actual.read()[:, dy:dy+h, dx:dx+w],
                                                  self.pixels[:, y:y+h, x:x+w])
                    if cfg.edges == 'pad':
                        mask = np.zeros((actual.height, actual.width), dtype='uint8')
                        mask[dy:dy+h, dx:dx+w] = 255
                        np.testing.assert_array_equal(actual.dataset_mask(), mask)
                        padded = actual.read()
                        self.assertTrue(np.all(padded[:, mask == 0] == src.nodata))
        return outputs

    def test_keep_partial_tiles_exact_data(self):
        self.assertEqual(len(self.assert_tile_matches_source(Config(self.source, self.root / 'keep', 5, 4))), 6)

    def test_overlap_preserves_pixels_and_transform(self):
        cfg = Config(self.source, self.root / 'overlap', 5, 4, overlap=2)
        self.assertEqual(len(self.assert_tile_matches_source(cfg)), 6)
        with rasterio.open(self.source) as src:
            self.assertEqual(list(tiles(src, cfg))[0].window.width, 7)

    def test_pad_edges_and_mask(self):
        cfg = Config(self.source, self.root / 'pad', 5, 4, edges='pad')
        outputs = self.assert_tile_matches_source(cfg)
        with rasterio.open(outputs[-1]) as ds:
            self.assertEqual((ds.width, ds.height), (5, 4))

    def test_discard_partial_tiles(self):
        cfg = Config(self.source, self.root / 'discard', 5, 4, edges='discard')
        self.assertEqual(len(self.assert_tile_matches_source(cfg)), 2)

    def test_distance_mode_and_feet_conversion(self):
        with rasterio.open(self.source) as ds:
            self.assertEqual(dimensions(ds, Config(self.source, self.root / 'unused', 1, 1, mode='distance'))[:2], (5, 5))
        outputs = tile_by_distance(self.source, self.root / 'distance', 1, 1)
        self.assertEqual(len(outputs), 6)
        with rasterio.open(self.source) as ds:
            self.assertEqual(dimensions(ds, Config(self.source, self.root / 'unused', 3.280839895, 3.280839895, mode='distance', units='ft'))[:2], (5, 5))

    def test_preview_does_not_create_tiles(self):
        output = self.root / 'preview'
        info = preview_grid(Config(self.source, output, 5, 4))
        self.assertEqual(info['count'], 6)
        self.assertEqual(info['tile_pixels'], (5, 4))
        self.assertFalse(output.exists())

    def test_api_accepts_strings_and_rejects_overwrite(self):
        output = self.root / 'string-paths'
        results = tile_by_pixels(str(self.source), str(output), 5, 4)
        self.assertEqual(len(results), 6)
        with self.assertRaises(FileExistsError):
            tile_by_pixels(str(self.source), str(output), 5, 4)

    def test_reject_invalid_overlap(self):
        with self.assertRaisesRegex(ValueError, 'Overlap'):
            run(Config(self.source, self.root / 'invalid', 5, 4, overlap=5))


if __name__ == '__main__':
    unittest.main()
