import logging
import shutil
from pathlib import Path
from functools import cache
import unittest
import numpy as np
import laspy

from tiles import tile_cloud_by_capacity, tile_cloud_by_length

LAZ_FILENAME =  Path('../tiler_test/synthetic.laz')
def sample_cloud(filename, n=10,):
    """Create a synthetic LAS point cloud containing exactly n points."""
    if n <= 0:
        raise ValueError("n must be greater than zero")
    base_path = Path(filename)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    path = str(base_path)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.zeros(3)
    las = laspy.LasData(header)
    rng = np.random.default_rng(42)
    las.x = rng.uniform(0, 100, n)
    las.y = rng.uniform(0, 100, n)
    las.z = rng.uniform(0, 20, n)
    las.red = rng.integers(0, 65536, n, dtype=np.uint16)
    las.green = rng.integers(0, 65536, n, dtype=np.uint16)
    las.blue = rng.integers(0, 65536, n, dtype=np.uint16)
    las.classification = rng.integers(0, 6, n, dtype=np.uint8)
    las.write(path)
    return path


def total_points(laz_file):
    return len(laspy.read(laz_file).points)

def creat_synthetic_cloud(file_name, size = 3266231):
    out = Path(file_name).resolve()
    if out.is_dir():
        raise ValueError("provide correct file path")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        print(f"\n Creating {out} with size: {size}")
        sample_cloud(out,  size)

def sample_cloud_by_length(path, nx=10, ny=10, spacing=1.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.zeros(3)
    xx, yy = np.meshgrid( np.arange(nx) * spacing,np.arange(ny) * spacing,)
    las = laspy.LasData(header)
    las.x = xx.ravel()
    las.y = yy.ravel()
    las.z = np.zeros(nx * ny)
    las.classification = np.arange(nx * ny, dtype=np.uint8) % 5
    las.write(str(path))
    return path

class TestTiles(unittest.TestCase):

    def setUp(self):

        """If real laz file does not exist, create it."""
        self.capacity = 300_000
        self.input_file = Path("../tiler_test/odm_georeferencing/odm_georeferenced_modely.laz").resolve()
        self.output_dir = Path("../tiler_test/test_tiles").resolve()
        if not self.input_file.exists():
            creat_synthetic_cloud(LAZ_FILENAME)
            self.input_file = Path(f'../tiler_test/{LAZ_FILENAME}')
        self.output_pattern = self.output_dir / "tile_#.laz"
        self.tp = total_points(self.input_file)


    def tearDown(self):
        if self.output_dir.exists():
            for file in self.output_dir.glob("tile_*.laz"):
                file.unlink()

    def test_total_points(self):
        self.assertIsInstance(self.tp, int)
        self.assertGreater(self.tp, 0)

    def test_tile_by_capacity(self):
        """test tile by capacity for expected tiles"""
        outputs = tile_cloud_by_capacity(
            self.input_file,
            self.output_pattern,
            capacity=self.capacity,)
        self.assertGreater(len(outputs), 0)
        expected_tiles = ( self.tp + self.capacity - 1 ) // self.capacity
        print(expected_tiles, 'expected_tiles')
        self.assertEqual(  len(outputs),
            expected_tiles,  )

    def test_each_tile_respects_capacity(self):
        outputs = tile_cloud_by_capacity(self.input_file, self.output_pattern, capacity=self.capacity, )
        for output in outputs:
            self.assertTrue(output.exists())
            points = total_points(output)
            self.assertGreater(points, 0)
            self.assertLessEqual(
                points,
                self.capacity,)

    def test_all_points_are_tiled(self):
        size = 10053
        tmp_file = self.output_dir / "tmp.laz"
        input =  sample_cloud(tmp_file, n=size)
        outputs = tile_cloud_by_capacity(input, self.output_pattern, capacity=50, )
        total =0
        points = []
        for output in outputs:
            tp = total_points(output)
            total += tp
            points.append(tp)
        self.assertEqual(size, total, msg="tiles points exceeds the total points of {}".format(size))
        self.assertEqual(points[-1], 3, 'last point is expected to be 3')
        tmp_file.unlink(missing_ok=True)



    def test_all_points_are_preserved(self):
        outputs = tile_cloud_by_capacity(
            self.input_file,
            self.output_pattern,
            capacity=self.capacity, )
        tiled_points = sum( total_points(file) for file in outputs)
        self.assertEqual(tiled_points, self.tp, )

    def test_output_names(self):
        outputs = tile_cloud_by_capacity(
            self.input_file,
            self.output_pattern,
            capacity=self.capacity,)

        for index, output in enumerate(outputs, start=1):
            self.assertEqual(
                output.name,
                f"tile_{index}.laz",)

    def test_capacity_larger_than_cloud(self):
        outputs = tile_cloud_by_capacity(
            self.input_file,
            self.output_pattern,
            capacity=self.tp + 1, )
        print(total_points(outputs[0]))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(total_points(outputs[0]), self.tp,)

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            tile_cloud_by_capacity(
                self.input_file,
                self.output_pattern,
                capacity=0, )

    def test_missing_hash(self):
        with self.assertRaises(ValueError):
            tile_cloud_by_capacity(
                self.input_file, self.output_dir / "tile.laz",capacity=self.capacity,)

    @classmethod
    def tearDownClass(cls):
        LAZ_FILENAME.unlink(missing_ok=True)
        pass

class TestTileCloudByLength(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("../tiler_test/unit_length").resolve()
        cls.test_dir.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.input_file = sample_cloud_by_length(
            self.test_dir / "input.las",
            nx=10,
            ny=10,
            spacing=1.0,
        )
        self.output_pattern = self.test_dir / "tile_#.las"

    def tearDown(self):
        for file in self.test_dir.glob("tile_*.las"):
            file.unlink()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_invalid_length(self):
        with self.assertRaises(ValueError):
            tile_cloud_by_length(
                self.input_file,
                self.output_pattern,
                length=0, )

    def test_tiles_are_created(self):
        outputs = tile_cloud_by_length(self.input_file,self.output_pattern, length=5.0, )
        self.assertGreater(len(outputs), 0)

        for output in outputs:
            self.assertTrue(output.exists())

    def test_expected_number_of_tiles(self):
        outputs = tile_cloud_by_length(self.input_file,self.output_pattern, length=5.0, )
        self.assertEqual(len(outputs), 4)

    def test_output_names(self):
        outputs = tile_cloud_by_length( self.input_file,self.output_pattern,length=5.0,)
        expected = [
            "tile_1.las",
            "tile_2.las",
            "tile_3.las",
            "tile_4.las",  ]
        self.assertEqual([file.name for file in outputs], expected, )

    def test_all_points_preserved_without_buffer(self):
        outputs = tile_cloud_by_length( self.input_file, self.output_pattern,  length=5.0,
            buffer=0.0, )
        tiled_points = sum(
            total_points(file)
            for file in outputs)
        self.assertEqual( tiled_points,total_points(self.input_file),)

    def test_buffer_creates_overlap(self):
        outputs = tile_cloud_by_length(
            self.input_file, self.output_pattern,length=5.0,buffer=1.0, )
        tiled_points = sum(total_points(file)  for file in outputs)
        self.assertGreater( tiled_points,
            total_points(self.input_file),)

    def test_single_large_tile(self):
        outputs = tile_cloud_by_length(self.input_file, self.output_pattern, length=100.0,)

        self.assertEqual(len(outputs), 1)
        self.assertEqual(
            total_points(outputs[0]),
            total_points(self.input_file),)

    def test_tile_extent_respects_length_without_buffer(self):
        outputs = tile_cloud_by_length(self.input_file,self.output_pattern,length=5.0,buffer=0.0,)
        for output in outputs:
            las = laspy.read(output)
            width = np.asarray(las.x).max() - np.asarray(las.x).min()
            height = np.asarray(las.y).max() - np.asarray(las.y).min()
            self.assertLess(width, 5.0)
            self.assertLess(height, 5.0)

    def test_points_are_not_lost(self):
        original = laspy.read(self.input_file)
        outputs = tile_cloud_by_length(
            self.input_file,
            self.output_pattern,
            length=5.0, )

        original_xyz = { tuple(row)
            for row in np.column_stack( (original.x, original.y, original.z) )}

        tiled_xyz = set()
        for output in outputs:
            las = laspy.read(output)
            tiled_xyz.update(tuple(row)
                for row in np.column_stack(
                    (las.x, las.y, las.z)))

        self.assertEqual(tiled_xyz, original_xyz)



if __name__ == "__main__":
    unittest.main(verbosity=2)
