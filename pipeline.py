"""Pure geometry for the OCR pipeline: reading order, line clustering, cropping.

Everything in this module is deterministic and depends only on numpy/cv2,
so it can be unit-tested without loading any model.

Coordinate convention: image coordinates, x right, y DOWN. Quads are 4x2
arrays ordered [top-left, top-right, bottom-right, bottom-left] after
passing through `order_points`.

Reading-order algorithm (per page):
  1. Estimate global skew as the median angle of the detected boxes'
     horizontal edges (wide boxes only, when enough exist).
  2. Rotate all box corners by the negative skew -> "deskewed space" where
     text lines are horizontal. Only coordinates are rotated, never pixels.
  3. Conservatively split into column blocks: a vertical cut requires a
     projection gap wider than `gap_factor * median box height` with no box
     crossing it, enough boxes on both sides, and both sides spanning most
     of the block height. Recurses (depth-limited) for 3+ columns.
  4. Within each block, cluster boxes into lines: a box joins the line whose
     vertical band it overlaps by >= `overlap_thresh` of the smaller height.
  5. Order lines by band center y; order boxes within a line by center x.
  6. Optionally split very long lines into chunks whose aspect ratio TrOCR
     can handle (it squeezes every crop to a 384x384 square).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = [
    "order_points",
    "quad_size",
    "estimate_skew",
    "rotate_points",
    "split_columns",
    "cluster_lines",
    "reading_order",
    "chunk_line",
    "merge_quads",
    "expand_quad",
    "perspective_crop",
    "assemble_text",
    "annotate",
    "compose_transcript",
    "Line",
]


# --------------------------------------------------------------------------
# basic quad utilities
# --------------------------------------------------------------------------

def order_points(quad) -> np.ndarray:
    """Return the 4 points ordered [TL, TR, BR, BL].

    Valid for quads rotated less than ~45 degrees, which is all this
    pipeline handles (page-level 90/180 rotation is an upstream
    orientation problem, not a reading-order problem).
    """
    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    idx = np.lexsort((pts[:, 1], pts[:, 0]))  # by x, ties by y
    left, right = pts[idx[:2]], pts[idx[2:]]
    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    return np.array([tl, tr, br, bl])


def quad_size(quad) -> tuple[float, float]:
    """(width, height) as the mean of opposite edge lengths of an ordered quad."""
    q = np.asarray(quad, dtype=np.float64)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2.0
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2.0
    return float(w), float(h)


def rotate_points(pts: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate points by theta_deg around the origin (standard math rotation;
    with y-down image coords a positive theta maps a +theta-skewed page back
    to horizontal when applied as -theta)."""
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    rot = np.array([[c, -s], [s, c]])
    return np.asarray(pts, dtype=np.float64) @ rot.T


def estimate_skew(quads: list[np.ndarray], max_abs_deg: float = 30.0,
                  wide_factor: float = 1.5) -> float:
    """Median angle (degrees) of box top/bottom edges.

    Prefers wide boxes (width >= wide_factor * height) because their edge
    direction reliably follows the text baseline; a near-square box around a
    single short word carries almost no angle information. Returns 0.0 when
    the estimate exceeds max_abs_deg — that signals a rotated page, which
    coordinate deskew must not attempt to fix.
    """
    angles, is_wide = [], []
    for quad in quads:
        q = order_points(quad)
        w, h = quad_size(q)
        if w < 2 or h < 2:
            continue
        v = ((q[1] - q[0]) + (q[2] - q[3])) / 2.0  # mean of top and bottom edge
        angles.append(np.degrees(np.arctan2(v[1], v[0])))
        is_wide.append(w >= wide_factor * h)
    if not angles:
        return 0.0
    angles = np.array(angles)
    wide = angles[np.array(is_wide)]
    med = float(np.median(wide if wide.size >= 3 else angles))
    return med if abs(med) <= max_abs_deg else 0.0


# --------------------------------------------------------------------------
# reading order
# --------------------------------------------------------------------------

@dataclass
class Line:
    """One text line: member box indices ordered left-to-right, plus the
    line's vertical band [top, bottom] in deskewed coordinates."""
    members: list[int]
    top: float
    bottom: float
    chunks: list[list[int]] = field(default_factory=list)

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2.0


def _band(quad_d: np.ndarray) -> tuple[float, float]:
    """Vertical band of a deskewed quad: mean of the 2 upper / 2 lower ys.
    Using means (not min/max) makes the band robust to one stray corner."""
    ys = np.sort(np.asarray(quad_d)[:, 1])
    return float(ys[:2].mean()), float(ys[2:].mean())


def split_columns(quads_d: list[np.ndarray], indices: list[int],
                  gap_factor: float = 2.0, min_side: int = 3,
                  min_y_coverage: float = 0.55, _depth: int = 0) -> list[list[int]]:
    """Conservative column split on deskewed quads. Returns groups of indices
    ordered left-to-right; [indices] unchanged when no confident cut exists.

    A cut requires: an x-projection gap (no box crosses it at any y) wider
    than gap_factor * median box height, at least min_side boxes per side,
    and each side spanning >= min_y_coverage of the block's height — so a
    single line with a big word gap can never trigger a phantom column.
    """
    if _depth >= 2 or len(indices) < 2 * min_side:
        return [indices]

    heights = [_band(quads_d[i])[1] - _band(quads_d[i])[0] for i in indices]
    med_h = float(np.median(heights))
    if med_h <= 0:
        return [indices]

    # merge the boxes' x-intervals; gaps between merged intervals are
    # x-ranges no box touches
    spans = sorted((float(quads_d[i][:, 0].min()), float(quads_d[i][:, 0].max()))
                   for i in indices)
    merged = [list(spans[0])]
    tol = 0.25 * med_h
    for lo, hi in spans[1:]:
        if lo <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    if len(merged) < 2:
        return [indices]

    gaps = [(merged[k + 1][0] - merged[k][1], (merged[k][1] + merged[k + 1][0]) / 2.0)
            for k in range(len(merged) - 1)]
    gaps = [g for g in gaps if g[0] >= gap_factor * med_h]
    if not gaps:
        return [indices]
    cut_x = max(gaps)[1]  # widest qualifying gap

    centers_x = {i: float(quads_d[i][:, 0].mean()) for i in indices}
    left = [i for i in indices if centers_x[i] < cut_x]
    right = [i for i in indices if centers_x[i] >= cut_x]
    if len(left) < min_side or len(right) < min_side:
        return [indices]

    def y_extent(group):
        bands = [_band(quads_d[i]) for i in group]
        return max(b for _, b in bands) - min(t for t, _ in bands)

    total = y_extent(indices)
    if total <= 0 or y_extent(left) < min_y_coverage * total \
            or y_extent(right) < min_y_coverage * total:
        return [indices]

    return (split_columns(quads_d, left, gap_factor, min_side, min_y_coverage, _depth + 1)
            + split_columns(quads_d, right, gap_factor, min_side, min_y_coverage, _depth + 1))


def cluster_lines(quads_d: list[np.ndarray], indices: list[int],
                  overlap_thresh: float = 0.4) -> list[Line]:
    """Group deskewed boxes into text lines by vertical-band overlap.

    A box joins the existing line it overlaps most, provided
    intersection / min(box_height, line_height) >= overlap_thresh.
    Normalizing by the smaller height keeps boxes with deep descenders or
    tall capitals (height outliers) attached to their true line.
    """
    items = []
    for i in indices:
        top, bottom = _band(quads_d[i])
        items.append((i, top, bottom))
    items.sort(key=lambda t: ((t[1] + t[2]) / 2.0, t[0]))

    lines: list[Line] = []
    for i, top, bottom in items:
        h = max(bottom - top, 1e-6)
        best, best_ov = None, overlap_thresh
        for line in lines:
            inter = min(bottom, line.bottom) - max(top, line.top)
            ov = inter / max(min(h, line.height), 1e-6)
            if ov >= best_ov:
                best, best_ov = line, ov
        if best is None:
            lines.append(Line(members=[i], top=top, bottom=bottom))
        else:
            n = len(best.members) + 1
            best.members.append(i)
            best.top += (top - best.top) / n        # running mean of bands
            best.bottom += (bottom - best.bottom) / n
    return lines


def reading_order(quads, column_split: bool = True,
                  overlap_thresh: float = 0.4,
                  gap_factor: float = 2.0) -> tuple[list[Line], float]:
    """Full reading order. Returns (lines, skew_deg); lines are in reading
    order, each line's members ordered left-to-right. Indices refer to the
    input quad list."""
    quads = [np.asarray(q, dtype=np.float64).reshape(4, 2) for q in quads]
    if not quads:
        return [], 0.0

    ordered = [order_points(q) for q in quads]
    theta = estimate_skew(ordered)
    deskewed = [rotate_points(q, -theta) for q in ordered]

    all_idx = list(range(len(quads)))
    groups = split_columns(deskewed, all_idx, gap_factor=gap_factor) \
        if column_split else [all_idx]

    result: list[Line] = []
    for group in groups:  # groups arrive left-to-right; read a column fully first
        lines = cluster_lines(deskewed, group, overlap_thresh=overlap_thresh)
        lines.sort(key=lambda l: l.center_y)
        for line in lines:
            line.members.sort(key=lambda i: (float(deskewed[i][:, 0].mean()), i))
        result.extend(lines)
    return result, theta


def chunk_line(line: Line, quads_d: list[np.ndarray],
               aspect_cap: float = 16.0) -> list[list[int]]:
    """Split a line's members (already x-ordered) into chunks whose merged
    width/height stays under aspect_cap.

    TrOCR resizes every crop to a 384x384 square; beyond ~16:1 the glyphs
    get squeezed into unreadability, so very long lines must be recognized
    in pieces.
    """
    chunks: list[list[int]] = []
    cur: list[int] = []
    cur_lo = cur_hi = 0.0
    h = max(line.height, 1e-6)
    for i in line.members:
        lo, hi = float(quads_d[i][:, 0].min()), float(quads_d[i][:, 0].max())
        if not cur:
            cur, cur_lo, cur_hi = [i], lo, hi
            continue
        new_lo, new_hi = min(cur_lo, lo), max(cur_hi, hi)
        if (new_hi - new_lo) / h > aspect_cap:
            chunks.append(cur)
            cur, cur_lo, cur_hi = [i], lo, hi
        else:
            cur, cur_lo, cur_hi = cur + [i], new_lo, new_hi
    if cur:
        chunks.append(cur)
    line.chunks = chunks
    return chunks


# --------------------------------------------------------------------------
# cropping
# --------------------------------------------------------------------------

def merge_quads(quads: list[np.ndarray]) -> np.ndarray:
    """Smallest rotated rectangle covering all quads, as an ordered quad
    (original image space)."""
    pts = np.vstack([np.asarray(q, dtype=np.float64).reshape(-1, 2) for q in quads])
    rect = cv2.minAreaRect(pts.astype(np.float32))
    return order_points(cv2.boxPoints(rect))


def expand_quad(quad: np.ndarray, pad_frac: float = 0.05) -> np.ndarray:
    """Push the ordered quad's corners outward along its own edge directions
    by pad_frac * height. No clipping to the image: the crop uses
    BORDER_REPLICATE, so slightly out-of-image corners are harmless and the
    quad stays a true parallelogram-ish shape."""
    q = np.asarray(quad, dtype=np.float64).copy()
    _, h = quad_size(q)
    pad = pad_frac * max(h, 1.0)
    ux = (q[1] - q[0]) + (q[2] - q[3])
    uy = (q[3] - q[0]) + (q[2] - q[1])
    nx = ux / max(np.linalg.norm(ux), 1e-6)
    ny = uy / max(np.linalg.norm(uy), 1e-6)
    q[0] += -nx * pad - ny * pad
    q[1] += +nx * pad - ny * pad
    q[2] += +nx * pad + ny * pad
    q[3] += -nx * pad + ny * pad
    return q


def perspective_crop(img: np.ndarray, quad, pad_frac: float = 0.05,
                     allow_rot90: bool = False) -> np.ndarray | None:
    """Rectify one ordered quad into an axis-aligned crop.

    Returns None for degenerate boxes instead of crashing warpPerspective.
    allow_rot90 rotates strongly vertical crops (h >= 2w) upright — pass it
    only for boxes known not to be single tall characters like 'I'.
    """
    q = expand_quad(order_points(quad), pad_frac).astype(np.float32)
    w = int(round(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))))
    h = int(round(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))))
    if w < 2 or h < 2:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(q, dst)
    warped = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    if allow_rot90 and h >= 2 * w:
        # np.rot90 returns a negative-stride VIEW; torch.from_numpy (in the
        # TrOCR processor) rejects those, so force a contiguous copy.
        warped = np.ascontiguousarray(np.rot90(warped))  # CCW, matching PaddleOCR
    return warped


# --------------------------------------------------------------------------
# output assembly / debug rendering
# --------------------------------------------------------------------------

def assemble_text(lines: list[Line], line_texts: list[str],
                  para_gap_factor: float = 1.0,
                  column_breaks: set[int] | None = None) -> str:
    """Join per-line texts in reading order. Inserts a blank line when the
    vertical gap to the previous line exceeds para_gap_factor * median line
    height (a paragraph break), or at a column boundary."""
    assert len(lines) == len(line_texts)
    if not lines:
        return ""
    med_h = float(np.median([max(l.height, 1e-6) for l in lines]))
    out: list[str] = []
    for k, (line, text) in enumerate(zip(lines, line_texts)):
        if k > 0:
            gap = line.top - lines[k - 1].bottom
            if (column_breaks and k in column_breaks) or gap > para_gap_factor * med_h:
                out.append("")
        out.append(text)
    return "\n".join(out)


def annotate(img_rgb: np.ndarray, quads: list[np.ndarray],
             lines: list[Line]) -> np.ndarray:
    """Debug overlay: each box outlined and numbered with its reading order,
    colored per line. Makes ordering mistakes visible at a glance."""
    out = img_rgb.copy()
    palette = [(46, 204, 113), (52, 152, 219), (231, 76, 60), (241, 196, 15),
               (155, 89, 182), (26, 188, 156), (230, 126, 34), (149, 165, 166)]
    n = 0
    scale = max(out.shape[0], out.shape[1]) / 1500.0
    fs = max(0.5, 0.9 * scale)
    th = max(1, int(round(2 * scale)))
    for li, line in enumerate(lines):
        color = palette[li % len(palette)]
        for i in line.members:
            n += 1
            q = np.asarray(quads[i], dtype=np.int32).reshape(-1, 2)
            cv2.polylines(out, [q], isClosed=True, color=color, thickness=th)
            pos = (int(q[:, 0].min()), max(int(q[:, 1].min()) - 4, 12))
            cv2.putText(out, str(n), pos, cv2.FONT_HERSHEY_SIMPLEX, fs,
                        color, th, cv2.LINE_AA)
    return out


_PALETTE = [(46, 204, 113), (52, 152, 219), (231, 76, 60), (241, 196, 15),
            (155, 89, 182), (26, 188, 156), (230, 126, 34), (192, 57, 43)]


def _load_font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "arial.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)  # Pillow >= 10
    except TypeError:
        return ImageFont.load_default()


def _wrap(draw, text: str, font, max_w: float) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    out, cur = [], words[0]
    for w in words[1:]:
        if draw.textlength(cur + " " + w, font=font) <= max_w:
            cur += " " + w
        else:
            out.append(cur)
            cur = w
    out.append(cur)
    return out


def compose_transcript(img_rgb: np.ndarray, line_boxes: list[list[np.ndarray]],
                       line_texts: list[str], panel_frac: float = 0.6) -> np.ndarray:
    """One image: the page with per-line numbered, color-coded boxes on the
    left, and the matching numbered transcript on the right.

    line_boxes[k] are the detected quads of line k; line_texts[k] is its text.
    The number and color on each box match its transcript entry, so you can
    read the page and the recognized text side by side and immediately see
    any mismatch. Pure numpy/PIL/cv2 — no models involved."""
    from PIL import Image, ImageDraw

    h, w = img_rgb.shape[:2]
    scale = max(h, w) / 1500.0
    th = max(1, int(round(2 * scale)))
    fs_cv = max(0.5, 1.0 * scale)

    # --- left: draw boxes, label each line at its leftmost box ---
    left = img_rgb.copy()
    for li, boxes in enumerate(line_boxes):
        color = _PALETTE[li % len(_PALETTE)]
        anchor = None
        for box in boxes:
            q = np.asarray(box, dtype=np.int32).reshape(-1, 2)
            cv2.polylines(left, [q], isClosed=True, color=color, thickness=th)
            x0, y0 = int(q[:, 0].min()), int(q[:, 1].min())
            if anchor is None or x0 < anchor[0]:
                anchor = (x0, y0)
        if anchor is not None:
            cv2.putText(left, str(li + 1), (anchor[0], max(anchor[1] - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_cv, color, th, cv2.LINE_AA)

    # --- right: numbered transcript panel ---
    panel_w = max(360, int(w * panel_frac))
    fsize = max(16, int(round(0.020 * h)))
    line_h = int(fsize * 1.35)
    pad = fsize
    font = _load_font(fsize)
    num_font = _load_font(fsize)

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    indent = int(max((probe.textlength(f"{k + 1}.", font=num_font)
                      for k in range(max(len(line_texts), 1))), default=0) + 0.6 * fsize)
    text_w = max(panel_w - 2 * pad - indent, 8 * fsize)

    wrapped = []
    for li, txt in enumerate(line_texts):
        segs = _wrap(probe, txt if txt.strip() else "(blank)", font, text_w)
        wrapped.append((li, _PALETTE[li % len(_PALETTE)], segs))

    panel_h = pad + sum(len(segs) * line_h + int(0.5 * line_h)
                        for _, _, segs in wrapped) + pad
    canvas_h = max(h, panel_h)

    panel = Image.new("RGB", (panel_w, canvas_h), (255, 255, 255))
    pd = ImageDraw.Draw(panel)
    y = pad
    for li, color, segs in wrapped:
        pd.text((pad, y), f"{li + 1}.", font=num_font, fill=color)
        for seg in segs:
            pd.text((pad + indent, y), seg, font=font, fill=(20, 20, 25))
            y += line_h
        y += int(0.5 * line_h)

    left_canvas = np.full((canvas_h, w, 3), 255, dtype=np.uint8)
    left_canvas[:h, :w] = left
    sep = np.full((canvas_h, max(2, th), 3), 220, dtype=np.uint8)
    return np.concatenate([left_canvas, sep, np.asarray(panel)], axis=1)
