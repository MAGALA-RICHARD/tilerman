from pathlib import Path
import laspy
import numpy as np
from math import ceil

def _part1by1(n):
    n = n.astype(np.uint64)
    n = (n | (n << 16)) & 0x0000FFFF0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F0F0F0F0F
    n = (n | (n << 2)) & 0x3333333333333333
    n = (n | (n << 1)) & 0x5555555555555555
    return n

def _morton_codes(x, y):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xn = (x - xmin) / max(xmax - xmin, 1e-12)
    yn = (y - ymin) / max(ymax - ymin, 1e-12)
    scale = (1 << 31) - 1
    xi = (xn * scale).astype(np.uint64)
    yi = (yn * scale).astype(np.uint64)
    return _part1by1(xi) | (_part1by1(yi) << 1)


def tile_cloud_by_capacity(laz_file, output_pattern, capacity=100_000):
    """
        Split a LAS/LAZ point cloud into spatially ordered tiles by point capacity.

        Points are ordered using Morton (Z-order) codes derived from their XY
        coordinates, then partitioned into sequential chunks containing at most
        ``capacity`` points. All point attributes are preserved in the output
        tiles.

        Parameters
        ----------
        laz_file : str | Path
            Path to the input LAS or LAZ point cloud.

        output_pattern : str | Path
            Output path pattern containing ``#`` as the tile index placeholder.
            For example, ``"tiles/tile_#.laz"`` produces ``tile_1.laz``,
            ``tile_2.laz``, and so on.

        capacity : int, default=100_000
            Maximum number of points allowed in each output tile. Must be greater
            than zero.

        Returns
        -------
        generator[Path]
            Paths to the generated tile files, in tile creation order.

        Raises
        ------
        ValueError
            If ``capacity`` is not greater than zero.
            If ``output_pattern`` does not contain ``#``.

        Notes
        -----
        The method preserves the input LAS header and all point dimensions.
        Spatial locality is maintained approximately through Morton ordering,
        but tile boundaries are determined by point count rather than fixed
        geometric dimensions.

        Examples
        --------
        >>> outputs = tile_by_capacity(
        ...     "cloud.laz",
        ...     "tiles/tile_#.laz",
        ...     capacity=300_000,
        ... )
        """
    laz_file = Path(laz_file)
    if capacity <= 0:
        raise ValueError("capacity must be greater than zero")
    if "#" not in str(output_pattern):
        raise ValueError("output_pattern must contain '#'")
    las = laspy.read(laz_file)
    x, y = np.asarray(las.x), np.asarray(las.y)
    morton = _morton_codes(x, y)
    order = np.argsort(morton)
    outputs = []
    for index, start in enumerate(range(0, len(order), capacity), start=1):
        ids = order[start:start + capacity]
        out = Path(str(output_pattern).replace("#", str(index)))
        out.parent.mkdir(parents=True, exist_ok=True)
        tile = laspy.LasData(las.header)
        tile.points = las.points[ids]
        tile.write(str(out.resolve()))
        yield out



def tile_cloud_by_length(input_file, output_pattern, length, buffer=0.0):
    """
        Split a LAS/LAZ point cloud into square XY tiles of a fixed side length.

        The XY extent of the input cloud is divided into a regular grid of square
        cells. Points falling within each cell are written to a separate tile.
        An optional buffer expands each tile boundary, allowing neighboring tiles
        to contain overlapping points.

        Parameters
        ----------
        input_file : str | Path
            Path to the input LAS or LAZ point cloud.

        output_pattern : str | Path
            Output path pattern containing ``#`` as the tile index placeholder.
            For example, ``"tiles/tile_#.laz"`` produces ``tile_1.laz``,
            ``tile_2.laz``, and so on.

        length : float
            Side length of each square tile, in the coordinate units of the input
            point cloud. Must be greater than zero.

        buffer : float, default=0.0
            Distance by which each tile boundary is expanded in all XY directions.
            A positive buffer creates overlap between adjacent tiles.

        Returns
        -------
        list[Path]
            Paths to the generated tile files, in tile creation order.

        Raises
        ------
        ValueError
            If ``length`` is not greater than zero.

        Notes
        -----
        Without a buffer, each point belongs to a single tile because the tile
        boundaries use half-open intervals. With a positive buffer, points near
        tile boundaries may appear in multiple output tiles.
        The method preserves the input LAS header and all point dimensions.
        Examples
        --------
        >>> outputs = tile_cloud_by_length(
        ...     "cloud.laz",
        ...     "tiles/tile_#.laz",
        ...     length=10.0,
        ...     buffer=0.5,
        ... )
        """
    input_file = Path(input_file)
    output_pattern = str(output_pattern)

    if length <= 0:
        raise ValueError("length must be greater than zero")

    las = laspy.read(input_file)
    x, y = np.asarray(las.x), np.asarray(las.y)

    xmin, ymin = x.min(), y.min()
    xmax, ymax = x.max(), y.max()

    outputs = []
    index = 1
    xs = np.arange(xmin, xmax, length)
    ys = np.arange(ymin, ymax, length)
    for x0 in xs:
        for y0 in ys:
            x1, y1 = x0 + length, y0 + length
            mask = (
                (x >= x0 - buffer) & (x < x1 + buffer) &
                (y >= y0 - buffer) & (y < y1 + buffer))
            if not np.any(mask):
                continue
            out = Path(output_pattern.replace("#", str(index)))
            out.parent.mkdir(parents=True, exist_ok=True)
            tile = laspy.LasData(las.header)
            tile.points = las.points[mask]
            tile.write(str(out))
            outputs.append(out)
            index += 1

    return outputs

def number_of_points(las_file):
    """Return the number of points without loading the full point cloud."""
    with laspy.open(las_file) as reader:
        return reader.header.point_count


def tile_cloud_by_number(laz_file, k):
    """Split a point cloud into approximately k tiles. Uses tile_cloud_by_capacity. under the hood"""

    if k <= 0:
        raise ValueError("k must be greater than zero")
    laz_file = Path(laz_file)
    points = number_of_points(laz_file)
    capacity = ceil(points / k)
    output_pattern = laz_file.with_name(f"{laz_file.stem}_tile_#{laz_file.suffix}")
    return tile_cloud_by_capacity(laz_file, output_pattern, capacity=capacity)