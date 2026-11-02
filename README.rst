Tilerman
========

Tilerman is a lightweight Python package for splitting LAS and LAZ point clouds
into smaller spatial tiles.

It supports two tiling strategies:

* **Capacity-based tiling** — splits a point cloud into spatially ordered chunks
  containing no more than a specified number of points.
* **Length-based tiling** — splits a point cloud into square XY tiles with a
  specified side length and optional overlap buffer.

Tilerman preserves the LAS header and existing point attributes when writing
output tiles.


Installation
============

Requirements
------------

Tilerman requires Python 3.12 or newer.

The main dependencies are:

* ``laspy``
* ``lazrs``
* ``numpy``

``lazrs`` provides LASzip-compatible LAZ compression and decompression support.


Install from Source
-------------------

Clone or download the project, then install it from the project root:

.. code-block:: bash

   pip install .

For development, install it in editable mode:

.. code-block:: bash

   pip install -e .

If you are using a virtual environment, activate it before installation.


Project Structure
=================

A typical project layout is:

.. code-block:: text

   tilerman/
   ├── pyproject.toml
   ├── README.rst
   ├── src/
   │   └── tilerman/
   │       ├── __init__.py
   │       └── tiles.py
   └── tests/
       └── test_tiles.py


Usage
=====

Import the tiling functions:

.. code-block:: python

   from tilerman.tiles import tile_by_capacity as  tile_cloud_by_capacity, tile_cloud_by_length as tile_cloud_by_length


Capacity-Based Tiling
---------------------

``tile_by_capacity`` divides a point cloud into spatially ordered tiles with a
maximum number of points per tile.

Points are ordered using Morton or Z-order indexing so that points that are
close spatially tend to remain close in the output sequence.

Example:

.. code-block:: python

   from tilerman.tiles import tile_by_capacity

   outputs = tile_by_capacity(
       "data/cloud.laz",
       "tiles/tile_#.laz",
       capacity=300_000,
   )

   for output in outputs:
       print(output)

This may produce:

.. code-block:: text

   tiles/tile_1.laz
   tiles/tile_2.laz
   tiles/tile_3.laz
   tiles/tile_4.laz

The ``#`` symbol in the output pattern is replaced automatically with the tile
number.

For example:

.. code-block:: python

   tile_by_capacity(
       "cloud.laz",
       "output/cloud_#.laz",
       capacity=100_000,
   )

could produce:

.. code-block:: text

   output/cloud_1.laz
   output/cloud_2.laz
   output/cloud_3.laz


Length-Based Tiling
-------------------

``tile_cloud_by_length`` divides the XY extent of a point cloud into square
tiles of a specified side length.

Example:

.. code-block:: python

   from tilerman.tiles import tile_cloud_by_length

   outputs = tile_cloud_by_length(
       "data/cloud.laz",
       "tiles/tile_#.laz",
       length=10.0,
   )

If the point cloud uses projected coordinates in metres, ``length=10.0``
creates tiles that are approximately 10 m by 10 m.


Using a Buffer
~~~~~~~~~~~~~~

An optional buffer can be used to create overlap between neighboring tiles.

.. code-block:: python

   outputs = tile_cloud_by_length(
       "data/cloud.laz",
       "tiles/tile_#.laz",
       length=10.0,
       buffer=0.5,)

In this example, every tile is expanded by 0.5 coordinate units in each XY
direction.

Because buffered tiles overlap, points near tile boundaries may appear in more
than one output tile.


LAS and LAZ Support
===================

Tilerman supports both ``.las`` and ``.laz`` files.

LAS files can be read directly by ``laspy``.

LAZ files require a compression backend. Tilerman installs ``lazrs`` through:

.. code-block:: text

   laspy[lazrs]

If LAZ support is unavailable, it can also be installed manually:

.. code-block:: bash

   pip install lazrs


Output Directories
==================

Output directories are created automatically when they do not exist.

For example:

.. code-block:: python

   tile_by_capacity(
       "cloud.laz",
       "results/tiles/tile_#.laz",
       capacity=250_000,)

will automatically create:

.. code-block:: text

   results/
   └── tiles/


Point Attributes
================

Tilerman writes subsets of the original LAS point records directly to each
tile.

Existing point attributes are therefore preserved, including attributes such
as:

* XYZ coordinates
* classification
* RGB values
* intensity
* return number
* GPS time
* extra LAS dimensions

provided that those dimensions exist in the original point cloud.


Running Tests
=============

Tilerman uses Python's built-in ``unittest`` framework.

Run all tests from the project root with:

.. code-block:: bash

   python -m unittest discover -s tests -p "test_*.py" -v

Run a specific test module with:

.. code-block:: bash

   python -m unittest tests.test_tiles -v


Example
=======

The following example performs both types of tiling:

.. code-block:: python

   from tilerman.tiles import tile_by_capacity, tile_cloud_by_length


   capacity_tiles = tile_by_capacity(
       "data/model.laz",
       "output/capacity/tile_#.laz",
       capacity=300_000,
   )

   spatial_tiles = tile_cloud_by_length(
       "data/model.laz",
       "output/spatial/tile_#.laz",
       length=5.0,
       buffer=0.25,
   )

   print(f"Capacity tiles: {len(capacity_tiles)}")
   print(f"Spatial tiles: {len(spatial_tiles)}")


Choosing a Tiling Method
========================

Use ``tile_by_capacity`` when the primary goal is to limit the number of points
processed at one time.

This is useful for:

* memory-limited workflows
* parallel processing
* machine-learning inference
* point-cloud batch processing

Use ``tile_cloud_by_length`` when physical tile dimensions are important.

This is useful for:

* spatial analysis
* local feature extraction
* fixed-area processing
* neighborhood-based point-cloud analysis
* geospatial workflows requiring predictable tile dimensions


Coordinate Systems
==================

For meaningful length-based tiling, the point cloud should normally use a
projected coordinate reference system.

For example, when coordinates are expressed in metres:

.. code-block:: python

   length=10.0

represents a 10 m tile side length.

Using geographic longitude and latitude coordinates directly is generally not
recommended for fixed-distance tiling because degrees are angular units rather
than constant physical distances.


License
=======

Add the project license information here when the package license is finalized.
```