"""Unit tests for pipeline.py (pure geometry — no models needed).

Run:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline  # noqa: E402


def make_box(x, y, w, h):
    """Axis-aligned quad [TL, TR, BR, BL]."""
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)


def rotate_page(quads, theta_deg, center=(0, 0)):
    """Rotate quads by +theta around center — simulates a skewed photo."""
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    rot = np.array([[c, -s], [s, c]])
    ctr = np.asarray(center, dtype=np.float64)
    return [(np.asarray(q) - ctr) @ rot.T + ctr for q in quads]


def flat_order(lines):
    return [i for line in lines for i in line.members]


class TestOrderPoints(unittest.TestCase):
    def test_scrambled_square(self):
        quad = [[100, 10], [10, 10], [10, 50], [100, 50]]  # TR, TL, BL, BR
        out = pipeline.order_points(quad)
        np.testing.assert_allclose(
            out, [[10, 10], [100, 10], [100, 50], [10, 50]])

    def test_slightly_rotated(self):
        quad = make_box(50, 50, 200, 40)
        (rotated,) = rotate_page([quad], 10, center=(150, 70))
        out = pipeline.order_points(rotated[[2, 0, 3, 1]])  # scramble rows
        np.testing.assert_allclose(out, rotated, atol=1e-9)


class TestReadingOrder(unittest.TestCase):
    def test_shuffled_grid(self):
        # 3 lines x 3 words, fed in shuffled order
        quads, expected = [], []
        for r in range(3):
            for c in range(3):
                quads.append(make_box(50 + 220 * c, 40 + 70 * r, 180, 40))
        rng = np.random.default_rng(7)
        perm = rng.permutation(9)
        shuffled = [quads[i] for i in perm]
        lines, theta = pipeline.reading_order(shuffled)
        self.assertEqual(len(lines), 3)
        self.assertAlmostEqual(theta, 0.0, places=5)
        # mapping back: reading order must equal row-major original order
        recovered = [int(perm[i]) for i in flat_order(lines)]
        self.assertEqual(recovered, list(range(9)))

    def test_skew_where_naive_y_sort_fails(self):
        # Long lines + 8 deg skew: the rightmost word of line 1 sits LOWER in
        # the photo than the leftmost word of line 2, so sorting boxes by
        # center-y interleaves the lines. Deskewed clustering must not.
        quads = []
        for r in range(3):           # 3 lines
            for c in range(5):       # 5 words, total width ~1100 px
                quads.append(make_box(40 + 220 * c, 100 + 70 * r, 180, 40))
        skewed = rotate_page(quads, 8, center=(600, 200))

        # sanity: prove the naive method actually fails on this input
        centers = [np.mean(q, axis=0) for q in skewed]
        naive = sorted(range(15), key=lambda i: (centers[i][1], centers[i][0]))
        self.assertNotEqual(naive, list(range(15)),
                            "test fixture too easy: naive sort accidentally works")

        lines, theta = pipeline.reading_order(skewed)
        self.assertEqual(len(lines), 3)
        self.assertAlmostEqual(theta, 8.0, delta=0.5)
        self.assertEqual(flat_order(lines), list(range(15)))

    def test_two_columns(self):
        # 2 columns x 4 lines; column gap (160 px) >> box height (30 px)
        quads, left_idx, right_idx = [], [], []
        for col, x0 in enumerate((40, 500)):
            for r in range(4):
                (left_idx if col == 0 else right_idx).append(len(quads))
                quads.append(make_box(x0, 50 + 60 * r, 300, 30))
        lines, _ = pipeline.reading_order(quads)
        order = flat_order(lines)
        self.assertEqual(order[:4], left_idx, "left column must be read first")
        self.assertEqual(order[4:], right_idx)

    def test_single_column_with_wide_word_gap_not_split(self):
        # One line has a huge inner gap; other lines bridge that x-range, so
        # no column split may happen.
        quads = [
            make_box(40, 50, 120, 30), make_box(600, 50, 120, 30),  # gappy line
            make_box(40, 110, 680, 30),                              # full-width
            make_box(40, 170, 680, 30),
            make_box(40, 230, 120, 30), make_box(600, 230, 120, 30),
        ]
        lines, _ = pipeline.reading_order(quads)
        self.assertEqual(len(lines), 4)
        self.assertEqual(flat_order(lines), [0, 1, 2, 3, 4, 5])

    def test_descender_height_outlier_same_line(self):
        # box B is 45% taller (deep descenders) but clearly the same line;
        # box C starts where B's descenders reach but overlaps B only 10px
        a = make_box(40, 100, 200, 40)    # band 100..140
        b = make_box(260, 102, 200, 58)   # band 102..160 (descenders)
        c = make_box(40, 150, 200, 40)    # band 150..190 -> next line
        lines, _ = pipeline.reading_order([a, b, c])
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].members, [0, 1])
        self.assertEqual(lines[1].members, [2])

    def test_empty(self):
        lines, theta = pipeline.reading_order([])
        self.assertEqual(lines, [])
        self.assertEqual(theta, 0.0)

    def test_single_box(self):
        lines, _ = pipeline.reading_order([make_box(10, 10, 100, 30)])
        self.assertEqual(flat_order(lines), [0])


class TestChunking(unittest.TestCase):
    def test_long_line_is_chunked(self):
        # 10 adjacent words, 100x30 each -> merged aspect 1000/30 = 33 > 16
        quads = [make_box(40 + 100 * c, 50, 96, 30) for c in range(10)]
        lines, _ = pipeline.reading_order(quads)
        self.assertEqual(len(lines), 1)
        deskewed = [pipeline.order_points(q) for q in quads]
        chunks = pipeline.chunk_line(lines[0], deskewed, aspect_cap=16.0)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual([i for ch in chunks for i in ch], list(range(10)),
                         "chunking must preserve order")
        for ch in chunks:
            lo = min(deskewed[i][:, 0].min() for i in ch)
            hi = max(deskewed[i][:, 0].max() for i in ch)
            self.assertLessEqual((hi - lo) / lines[0].height, 16.0 + 1e-6)

    def test_short_line_single_chunk(self):
        quads = [make_box(40, 50, 200, 40), make_box(260, 50, 200, 40)]
        lines, _ = pipeline.reading_order(quads)
        deskewed = [pipeline.order_points(q) for q in quads]
        chunks = pipeline.chunk_line(lines[0], deskewed)
        self.assertEqual(chunks, [[0, 1]])


class TestMergeAndCrop(unittest.TestCase):
    def test_merge_quads_covers_members_tightly(self):
        words = [make_box(40 + 150 * c, 100, 130, 36) for c in range(4)]
        skewed = rotate_page(words, 5, center=(300, 118))
        merged = pipeline.merge_quads(skewed)
        w, h = pipeline.quad_size(merged)
        self.assertGreater(w, 550)            # spans all words
        self.assertLess(h, 36 * 1.4)          # but stays one line tall
        # every member corner inside (or on) the merged rect
        merged_i = merged.astype(np.float32)
        for q in skewed:
            for pt in q:
                d = cv2.pointPolygonTest(merged_i, (float(pt[0]), float(pt[1])), True)
                self.assertGreater(d, -1.5)

    def test_crop_rectifies_rotated_text(self):
        # paint a black bar on white, rotated 12 deg; crop must come back
        # axis-aligned with dark pixels in the middle row, white at corners
        img = np.full((400, 600, 3), 255, dtype=np.uint8)
        quad = make_box(150, 180, 300, 40)
        (rq,) = rotate_page([quad], 12, center=(300, 200))
        cv2.fillPoly(img, [rq.astype(np.int32)], (0, 0, 0))
        crop = pipeline.perspective_crop(img, rq, pad_frac=0.10)
        self.assertIsNotNone(crop)
        ch, cw = crop.shape[:2]
        self.assertGreater(cw / ch, 4.0)  # wide line stays wide
        mid = crop[ch // 2, cw // 2].mean()
        corner = crop[1, 1].mean()
        self.assertLess(mid, 60)          # painted bar
        self.assertGreater(corner, 200)   # padded margin is page-white

    def test_degenerate_quad_returns_none(self):
        line = np.array([[10, 10], [200, 10], [200, 10], [10, 10]], dtype=np.float64)
        self.assertIsNone(pipeline.perspective_crop(
            np.zeros((100, 300, 3), np.uint8), line))

    def test_vertical_crop_rot90(self):
        img = np.zeros((300, 200, 3), np.uint8)
        tall = make_box(50, 30, 40, 200)
        upright = pipeline.perspective_crop(img, tall, allow_rot90=True)
        self.assertGreater(upright.shape[1], upright.shape[0])
        self.assertTrue(upright.flags["C_CONTIGUOUS"],
                        "rot90 crop must be contiguous for torch.from_numpy")
        kept = pipeline.perspective_crop(img, tall, allow_rot90=False)
        self.assertGreater(kept.shape[0], kept.shape[1])


class TestAssemble(unittest.TestCase):
    def test_paragraph_gap(self):
        l1 = pipeline.Line([0], top=100, bottom=140)
        l2 = pipeline.Line([1], top=150, bottom=190)   # gap 10 -> same para
        l3 = pipeline.Line([2], top=260, bottom=300)   # gap 70 > med_h 40
        text = pipeline.assemble_text([l1, l2, l3], ["one", "two", "three"])
        self.assertEqual(text, "one\ntwo\n\nthree")

    def test_empty(self):
        self.assertEqual(pipeline.assemble_text([], []), "")


class TestComposeTranscript(unittest.TestCase):
    def test_composite_shape_and_no_mutation(self):
        img = np.full((300, 500, 3), 255, np.uint8)
        before = img.copy()
        boxes = [[make_box(20, 30, 200, 30)], [make_box(20, 90, 260, 30)]]
        out = pipeline.compose_transcript(img, boxes, ["first line", "second line"])
        self.assertTrue(np.array_equal(img, before), "must not mutate input")
        self.assertGreaterEqual(out.shape[1], img.shape[1], "panel widens the image")
        self.assertGreaterEqual(out.shape[0], img.shape[0])
        self.assertEqual(out.shape[2], 3)
        # the right-hand panel must contain dark (text) pixels on white
        panel = out[:, img.shape[1]:]
        self.assertTrue((panel < 80).any(), "transcript text not rendered")

    def test_handles_blank_text_and_empty_boxes(self):
        img = np.full((120, 200, 3), 255, np.uint8)
        out = pipeline.compose_transcript(img, [[]], [""])
        self.assertEqual(out.shape[2], 3)
        self.assertGreaterEqual(out.shape[1], img.shape[1])


class TestAnnotate(unittest.TestCase):
    def test_overlay_draws_in_order(self):
        img = np.full((200, 400, 3), 255, np.uint8)
        quads = [make_box(20, 30, 100, 30), make_box(150, 30, 100, 30)]
        lines, _ = pipeline.reading_order(quads)
        out = pipeline.annotate(img, quads, lines)
        self.assertEqual(out.shape, img.shape)
        self.assertFalse(np.array_equal(out, img), "overlay must draw something")
        self.assertTrue(np.array_equal(img, np.full((200, 400, 3), 255, np.uint8)),
                        "annotate must not mutate the input image")


if __name__ == "__main__":
    unittest.main(verbosity=2)
