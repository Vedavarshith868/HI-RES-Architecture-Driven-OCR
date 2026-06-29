"""Skew-robust drop-in replacement for PaddleOCR's `sorted_boxes`.

PaddleOCR orders detected boxes into reading order with (tools/infer/
predict_system.py @ main):

    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    # then a bubble pass that only swaps neighbours within abs(dy) < 10 px

The fixed 10-pixel row threshold assumes the page is upright. Once it is tilted,
a single text line spans far more than 10 px in y (a 1000-px-wide line at 3°
already spans ~52 px, at 10° ~176 px), so boxes from one visual line are split
across several "rows" and the reading order interleaves.

This version estimates the dominant text angle from the detected boxes, computes
the row/column sort keys in the *deskewed* frame, and uses a row tolerance
proportional to text height. On an upright page (theta ~= 0) it reduces to the
original top-to-bottom / left-to-right order, so it is backward compatible for
the common case and only changes behaviour on skewed input.

Pure NumPy, no extra dependencies. Scope: in-plane skew of left-to-right
scripts; it does not attempt multi-column segmentation or RTL.
"""
from __future__ import annotations

import numpy as np

_MAX_SKEW_RAD = 0.2618  # clamp the angle estimate to +/-15 degrees


def reading_order_indices(dt_boxes) -> list[int]:
    """Return indices of `dt_boxes` in skew-robust reading order."""
    n = len(dt_boxes)
    if n <= 1:
        return list(range(n))
    boxes = np.asarray(dt_boxes, dtype=np.float64)
    p0, p1, p3 = boxes[:, 0], boxes[:, 1], boxes[:, 3]

    # dominant skew: median angle of each box's top edge (p1 - p0)
    theta = float(np.median(np.arctan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])))
    theta = max(-_MAX_SKEW_RAD, min(_MAX_SKEW_RAD, theta))

    # sort keys in the deskewed frame (rotate the top-left point by -theta)
    cos, sin = np.cos(-theta), np.sin(-theta)
    ky = p0[:, 0] * sin + p0[:, 1] * cos          # deskewed y (row)
    kx = p0[:, 0] * cos - p0[:, 1] * sin          # deskewed x (column)

    heights = np.hypot(p3[:, 0] - p0[:, 0], p3[:, 1] - p0[:, 1])
    row_tol = max(10.0, 0.5 * float(np.median(heights)))  # height-relative, not fixed 10px

    order = sorted(range(n), key=lambda i: ky[i])
    out: list[int] = []
    line = [order[0]]
    for prev, cur in zip(order, order[1:]):
        if ky[cur] - ky[prev] <= row_tol:         # same visual line in deskewed space
            line.append(cur)
        else:
            out.extend(sorted(line, key=lambda j: kx[j]))
            line = [cur]
    out.extend(sorted(line, key=lambda j: kx[j]))
    return out


def sorted_boxes(dt_boxes):
    """Skew-robust drop-in for PaddleOCR's sorted_boxes.

    Args:
        dt_boxes (ndarray): detected boxes, shape (N, 4, 2).
    Returns:
        list: the same boxes reordered into reading order.
    """
    return [dt_boxes[i] for i in reading_order_indices(dt_boxes)]
