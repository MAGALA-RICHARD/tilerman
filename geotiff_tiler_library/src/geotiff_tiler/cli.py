"""Optional command-line interface for the geotiff_tiler library."""
import argparse
from pathlib import Path
from .orthophoto import Config, run, preview_grid

def main():
    p = argparse.ArgumentParser(description='Window-based lossless GeoTIFF tiling')
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    p.add_argument('--mode', choices=('pixels', 'distance'), default='pixels')
    p.add_argument('--width', type=float, default=512)
    p.add_argument('--height', type=float, default=512)
    p.add_argument('--units', choices=('m', 'ft'), default='m')
    p.add_argument('--overlap', type=float, default=0)
    p.add_argument('--overlap-units', choices=('pixels', 'map', 'm', 'ft'), default='pixels')
    p.add_argument('--edges', choices=('keep', 'pad', 'discard'), default='keep')
    p.add_argument('--pad-value', type=float)
    p.add_argument('--compress', choices=('source', 'none', 'lzw', 'deflate', 'zstd'), default='source')
    p.add_argument('--naming', default='Tile_Row{row:03d}_Col{col:03d}.tif')
    p.add_argument('--folders', choices=('single', 'row', 'grid'), default='single')
    p.add_argument('--preview', action='store_true')
    a = p.parse_args()
    cfg = Config(a.source, a.destination, a.width, a.height, a.mode, a.units, a.overlap,
                 a.overlap_units, a.edges, a.pad_value, a.compress, a.naming, a.folders)
    if a.preview:
        print(preview_grid(cfg))
    else:
        result = run(cfg, progress=lambda n, total, eta: print(f'\r{n}/{total} tiles | ETA {eta:.0f}s', end='', flush=True))
        print(f'\nCreated {len(result)} tiles in {a.destination}')

if __name__ == '__main__':
    main()
