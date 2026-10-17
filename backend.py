#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backend.py — GPU/CPU backend selection (bottom layer: zero internal imports).

Exposes: cp (cupy or the numpy shim), np, to_cpu(), and _cfg_early (the early
config read the FORCE_CPU check needs, reused by config.py).
"""
import numpy as np
import os
import sys

# --- 0. GPU/CPU backend selection ---
# Use CuPy (GPU) if there's enough free VRAM; otherwise fall back to NumPy.
cp = None
_FORCE_CPU = False
# --cpu on the command line or FORCE_CPU=1 in the environment bypasses the
# CuPy VRAM probe completely (quick CPU testing without touching config).
_cfg_early = None
try:
    import tomllib as _tl
    _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml') if '__file__' in dir() else 'config.toml'
    if os.path.exists(_p):
        with open(_p, 'rb') as _fh:
            _cfg_early = _tl.load(_fh)
        _FORCE_CPU = bool(_cfg_early.get('game', {}).get('force_cpu', False))
except Exception:
    pass
if '--cpu' in sys.argv or os.environ.get('FORCE_CPU', '') == '1':
    _FORCE_CPU = True
    sys.argv = [a for a in sys.argv if a != '--cpu']   # keep ursina's arg parsing clean
if not _FORCE_CPU:
    try:
        import cupy as _cupy_probe
        # mem_info is (free, total) — index [0] is the free amount. (An old
        # version read [1] = TOTAL, which made the "> 512 MB free" check
        # almost always true and the log line misleading; the probe
        # allocation below is what actually gated GPU use.)
        _free_mb = _cupy_probe.cuda.Device(0).mem_info[0] // (1024 * 1024)
        # probe by actually allocating ~ what the sim needs — mem_info can be
        # stale/generous while other processes (llama.cpp) fill VRAM on demand
        _probe = _cupy_probe.zeros(64_000_000, dtype=_cupy_probe.float32)  # 256 MB
        _cupy_probe.cuda.Device(0).synchronize()
        del _probe
        _cupy_probe.get_default_memory_pool().free_all_blocks()
        if _free_mb > 512:
            cp = _cupy_probe
            print(f"[backend] CuPy GPU (free VRAM: {_free_mb} MB)")
        else:
            print(f"[backend] NumPy CPU (only {_free_mb} MB VRAM free)")
    except Exception as e:
        print(f"[backend] NumPy CPU (cupy unavailable: {type(e).__name__})")
else:
    print("[backend] NumPy CPU (forced by config: game.force_cpu)")

if cp is None:
    # Shim: pretend numpy is cupy. numpy supports everything the sim uses
    # (argsort/where/linalg/float32). Only .get() is missing (C type, can't
    # be monkey-patched), so all reads go through to_cpu() below. (No fake
    # cp.cuda attribute: `cp = np` makes that an in-place mutation of the
    # shared numpy module, visible to every other importer.)
    cp = np

def to_cpu(arr):
    """cupy array -> numpy via .get(); numpy array passes through."""
    return arr.get() if hasattr(arr, 'get') else arr
