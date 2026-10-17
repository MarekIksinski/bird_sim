# -*- coding: utf-8 -*-
"""
math3d.py — pure NumPy batched 3D quaternion utilities (layer 3: no state).

Convention (verified numerically against entity.quaternion/e.forward):
_q_rot_b's +w*uv form matches Panda3D's rotation exactly. Control deltas
are built from BODY axes and RIGHT-multiply the attitude
(q_new = q * dq_body), which is the identical rotation as left-multiplying
the delta built from the same axes' world images.
"""
import numpy as np

def _q_mul_b(a, b):
    """Batched quaternion multiply, [x,y,z,w] rows, (N,4) x (N,4) -> (N,4)."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz], axis=1)

def _q_from_aa_b(axes, deg):
    """Batched axis-angle quats: axes (N,3) unit-ish, deg (N,) -> (N,4) [x,y,z,w]."""
    r = np.radians(deg) * 0.5
    s = np.sin(r)[:, None]; c = np.cos(r)[:, None]
    n = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / np.maximum(n, 1e-9)
    return np.concatenate([axes * s, c], axis=1)

def _q_rot_b(q, v):
    """Batched rotate vectors v (N,3) by quaternions q (N,4)."""
    u = q[:, :3]; w = q[:, 3:4]
    uv = np.cross(u, v); uuv = np.cross(u, uv)
    return v + 2 * (w * uv + uuv)

def _slerp_np(a, b, t):
    """Shortest-arc slerp of unit quats [x,y,z,w]. Falls back to nlerp when
    the arc is tiny (sin(theta)~0) and renormalizes degenerate inputs."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    if na < 1e-9:
        return b / max(np.linalg.norm(b), 1e-9)
    a = a / na
    if np.dot(a, b) < 0.0:
        b = -b                          # shortest arc: never a 180deg snap
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 0.9995:                      # near-identical: lerp+renorm is stable
        q = a + (b - a) * t
    else:
        th = np.arccos(c); s = np.sin(th)
        q = a * np.sin((1 - t) * th) / s + b * np.sin(t * th) / s
    return q / max(np.linalg.norm(q), 1e-9)
