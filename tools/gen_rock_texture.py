#!/usr/bin/env python3
"""Generate assets/textures/rock_procedural.png — a seamless (tileable),
subtle, noise-blended grey-brown rock/cliff texture for the triplanar side
texture. Pure numpy fBm on a PERIODIC lattice (each octave's lattice wraps
modulo its own cell count, so the image tiles edge-to-edge with no seams)
plus low-frequency crack modulation. Committed asset: run this only to
regenerate/tune.

Usage: .venv/bin/python tools/gen_rock_texture.py [--size 512] [--seed 11]
"""
import argparse
import numpy as np
from pathlib import Path


def _smooth(t):
    return t * t * (3 - 2 * t)


def periodic_fbm(rng, size, octaves=6, base_cells=4, gain=0.55, lacunarity=2):
    """fBm over a size x size pixel grid; bilinear sampling of per-octave
    value lattices with WRAP indexing -> the result is exactly periodic."""
    xs = np.arange(size) / size                     # 0..1 (endpoint excluded)
    fx = xs[None, :] * 0 + np.linspace(0, 1, size, endpoint=False)[:, None]
    out = np.zeros((size, size), dtype=np.float64)
    total, amp = 0.0, 1.0
    for o in range(octaves):
        cells = base_cells * (lacunarity ** o)
        g = rng.random((cells, cells))
        f = np.linspace(0, 1, size, endpoint=False) * cells
        i = np.floor(f).astype(int)
        t = _smooth(f - i)
        i0, i1 = i % cells, (i + 1) % cells         # periodic wrap
        # separable: interpolate rows then columns
        w0x, w1x = (1 - t)[:, None], t[:, None]     # axis-0 weights
        w0y, w1y = (1 - t)[None, :], t[None, :]     # axis-1 weights
        lat = (w0x * w0y * g[np.ix_(i0, i0)]
               + w1x * w0y * g[np.ix_(i1, i0)]
               + w0x * w1y * g[np.ix_(i0, i1)]
               + w1x * w1y * g[np.ix_(i1, i1)])
        out += lat * amp
        total += amp
        amp *= gain
    return out / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent.parent
                                        / 'assets' / 'textures' / 'rock_procedural.png'))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    s = args.size

    grain = periodic_fbm(rng, s, octaves=8, base_cells=2, gain=0.68)   # rock grain
    broad = periodic_fbm(rng, s, octaves=3, base_cells=2, gain=0.6)    # tonal drift
    crack = periodic_fbm(rng, s, octaves=5, base_cells=3)

    # subtle dark cracks: thin ridges where crack noise crosses a low band
    # (narrower band + higher power -> fissure lines, not blobs)
    cr = np.clip(1.0 - np.abs(crack - 0.42) * 14.0, 0.0, 1.0) ** 3
    lum = 0.60 + 0.42 * (grain - 0.5) * 2 + 0.14 * (broad - 0.5) * 2
    lum *= (1.0 - 0.35 * cr)
    lum = np.clip(lum, 0.0, 1.0)

    # neutral grey with a faint warm-brown cast (matches the fog/valley palette)
    base = np.array([0.50, 0.47, 0.42])
    rgb = base[None, None, :] * lum[:, :, None]
    # fine speckle so close-up tiling reads as grain, not mush
    speck = rng.random((s, s))
    rgb += (speck - 0.5)[..., None] * 0.09
    rgb = np.clip(rgb, 0.0, 1.0)

    img8 = (rgb * 255).astype(np.uint8)
    from PIL import Image
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img8, 'RGB').save(out)
    print(f'wrote {out} ({s}x{s})')
    # seam self-check: wrap-continuity mean 8-bit diff across the edge vs
    # a typical interior neighbour step (should be the same order)
    d_col = np.abs(img8[:, 0].astype(int) - img8[:, -1].astype(int)).mean()
    d_row = np.abs(img8[0].astype(int) - img8[-1].astype(int)).mean()
    interior = np.abs(img8[:, 1:].astype(int) - img8[:, :-1].astype(int)).mean()
    print(f'seam delta: col {d_col:.1f}, row {d_row:.1f}, interior step {interior:.1f}')


if __name__ == '__main__':
    main()
