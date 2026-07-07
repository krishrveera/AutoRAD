"""Synthetic centerline builder for turn-segmentation tests.

Generates ``(points, s, yaw_deg, track_length)`` arrays in the same format
:func:`game.turns.build_centerline` produces, from a list of segment specs:

* ``('straight', length_m)``
* ``('arc', radius_m, angle_deg, 'left' | 'right')``

Convention matches CARLA as used by :mod:`game.turns`: a right turn is one
where yaw (degrees) increases along the track (``RIGHT_TURN_SIGN = +1``).
"""

import numpy as np


def build_track(segments, step=1.0):
    pts, yaws = [], []
    x, y, yaw = 0.0, 0.0, 0.0

    for seg in segments:
        if seg[0] == "straight":
            n = max(1, int(round(seg[1] / step)))
            for _ in range(n):
                pts.append((x, y))
                yaws.append(yaw)
                x += step * np.cos(np.radians(yaw))
                y += step * np.sin(np.radians(yaw))
        elif seg[0] == "arc":
            _, radius, angle_deg, direction = seg
            arc_len = radius * np.radians(angle_deg)
            n = max(1, int(round(arc_len / step)))
            dyaw = angle_deg / n * (1 if direction == "right" else -1)
            for _ in range(n):
                pts.append((x, y))
                yaws.append(yaw)
                x += step * np.cos(np.radians(yaw))
                y += step * np.sin(np.radians(yaw))
                yaw += dyaw
        else:
            raise ValueError(f"Unknown segment spec: {seg}")

    points = np.asarray(pts, dtype=float)
    seg_len = np.hypot(*np.diff(points, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    track_length = float(s[-1] + np.hypot(*(points[0] - points[-1])))
    return points, s, np.asarray(yaws, dtype=float), track_length


def stadium(radius=100.0, straight=200.0, direction="right", step=1.0):
    """Closed stadium loop: two straights joined by two semicircles."""
    return build_track([
        ("straight", straight),
        ("arc", radius, 180.0, direction),
        ("straight", straight),
        ("arc", radius, 180.0, direction),
    ], step=step)
