"""Isolated proof for the PaddleOCR skew-robustness fix.

Question: does PaddleOCR's `sorted_boxes` reading-order break under mild page
skew, and does a skew-aware ordering fix it — *holding everything else equal*?

Method (the cleanest possible isolation): for each page we run PaddleOCR
detection ONCE and recognize every detected box ONCE, then assemble the page
text two different ways and score CER against the ground truth:

  * `paddle_sorted_order`  — PaddleOCR's exact current logic
    (tools/infer/predict_system.py @ main): sort by (y, x) with a hardcoded
    10-pixel row threshold.
  * `skew_robust_order`    — estimate page skew, cluster lines in deskewed
    space, then order top-to-bottom / left-to-right.

Detection and recognition are identical for both; only the ordering differs, so
any CER gap is purely reading order. This mirrors the one-function change the PR
proposes for `sorted_boxes`.

Run on a GPU Colab where PaddleOCR already works:
    python skew_sortfix_proof.py
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent
for p in (_ROOT, _ROOT / "multilingual"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pipeline
import evaluate as E
from detector import Detector
import ml_engine as ML
import ml_evaluate as M

# ---- configuration -------------------------------------------------------
SKEW_LANGS = ["zh", "ja", "de", "es", "fr"]
PPOCR_LANGS = {"zh": "ch", "ja": "japan", "de": "german", "es": "es", "fr": "fr"}
SKEW_N = 8                       # pages per language
SKEW_ANGLES = [0, 3, 6, 10]     # degrees
DET_MODEL = "PP-OCRv6_medium_det"
REC_MODEL = "PP-OCRv6_medium_rec"
SEED = 42


# ---- the two orderings under test ----------------------------------------
def paddle_sorted_order(boxes: list[np.ndarray]) -> list[int]:
    """PaddleOCR's exact sorted_boxes logic, returning the index order.

    Verbatim port of tools/infer/predict_system.py @ main: sort by (top-y, x)
    then a local bubble pass that only swaps neighbours within 10 px of y."""
    n = len(boxes)
    idx = sorted(range(n), key=lambda i: (boxes[i][0][1], boxes[i][0][0]))
    for i in range(n - 1):
        for j in range(i, -1, -1):
            a, b = boxes[idx[j + 1]], boxes[idx[j]]
            if abs(a[0][1] - b[0][1]) < 10 and a[0][0] < b[0][0]:
                idx[j], idx[j + 1] = idx[j + 1], idx[j]
            else:
                break
    return idx


def skew_robust_order(boxes: list[np.ndarray]) -> list[int]:
    """Skew-aware reading order: deskew -> line-cluster -> ltr within line."""
    quads = np.asarray(boxes, dtype=np.float64)
    lines, _theta = pipeline.reading_order(quads)
    return [i for line in lines for i in line.members]


# ---- helpers -------------------------------------------------------------
def rotate_image(img: np.ndarray, deg: float) -> np.ndarray:
    if not deg:
        return img
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, mat, (w, h), borderValue=(255, 255, 255),
                          flags=cv2.INTER_LINEAR)


def ensure_xfund(lang: str) -> str:
    d = _ROOT / f"xfund_{lang}"
    d.mkdir(exist_ok=True)
    base = "https://github.com/doc-analysis/XFUND/releases/download/v1.0"
    for fn in (f"{lang}.val.json", f"{lang}.val.zip"):
        dst = d / fn
        if not dst.exists():
            print(f"  downloading {fn}...", flush=True)
            urllib.request.urlretrieve(f"{base}/{fn}", dst)
    with zipfile.ZipFile(d / f"{lang}.val.zip") as z:
        z.extractall(d)
    return str(d)


def main() -> int:
    norm = E.NormCfg()
    rng = np.random.default_rng(SEED)
    rows = []  # (lang, angle, buggy_cer, fixed_cer)

    for xlang in SKEW_LANGS:
        plang = PPOCR_LANGS[xlang]
        sep = "" if plang in ML._NO_SPACE_LANGS else " "
        print(f"\n=== {xlang} ({plang}) ===", flush=True)
        samples = M.load_xfund(ensure_xfund(xlang), plang, split="val", n=SKEW_N)

        detector = Detector(DET_MODEL)
        recognizer = ML.PaddleRecognizer(REC_MODEL)

        for deg in SKEW_ANGLES:
            b_edits = b_ref = f_edits = f_ref = 0
            for s in samples:
                sign = rng.choice([-1, 1])
                img = rotate_image(s.image_rgb(), deg * sign)

                boxes = detector(img)
                if len(boxes) == 0:
                    # no detection -> both orderings produce empty text
                    e, r = E.cer_counts(s.gt, "", norm)
                    b_edits += e; b_ref += r; f_edits += e; f_ref += r
                    continue
                boxes = [b for b in boxes]

                # recognize each detected box exactly once
                crops = [pipeline.perspective_crop(img, pipeline.order_points(b),
                                                   pad_frac=0.0, interp=cv2.INTER_CUBIC)
                         for b in boxes]
                keep = [i for i, c in enumerate(crops) if c is not None]
                texts_raw = recognizer([crops[i] for i in keep])
                texts = {ki: t for ki, t in zip(keep, texts_raw)}

                def assemble(order):
                    return sep.join(texts[i] for i in order
                                    if i in texts and texts[i])

                buggy = assemble(paddle_sorted_order(boxes))
                fixed = assemble(skew_robust_order(boxes))

                e, r = E.cer_counts(s.gt, buggy, norm); b_edits += e; b_ref += r
                e, r = E.cer_counts(s.gt, fixed, norm); f_edits += e; f_ref += r

            b_cer = b_edits / max(b_ref, 1)
            f_cer = f_edits / max(f_ref, 1)
            rows.append((xlang, deg, b_cer, f_cer))
            print(f"  skew {deg:2d}deg:  sorted_boxes {b_cer:.1%}   "
                  f"skew-robust {f_cer:.1%}   (Δ {b_cer - f_cer:+.1%})", flush=True)

        del detector, recognizer

    # ---- CSV + figure ----
    import csv
    with open(_ROOT / "skew_sortfix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["language", "skew_angle", "sorted_boxes_cer", "skew_robust_cer"])
        w.writerows(rows)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(SKEW_LANGS),
                                 figsize=(3.6 * len(SKEW_LANGS), 4.3), sharey=True)
        for ax, lang in zip(axes, SKEW_LANGS):
            r = [x for x in rows if x[0] == lang]
            angs = [x[1] for x in r]
            ax.plot(angs, [x[2] for x in r], "s--", color="#f97316", lw=2,
                    label="sorted_boxes (current)")
            ax.plot(angs, [x[3] for x in r], "o-", color="#2563eb", lw=2,
                    label="skew-robust (proposed)")
            ax.set_title(lang); ax.set_xlabel("page skew (deg)"); ax.grid(alpha=0.3)
            if ax is axes[0]:
                ax.set_ylabel("CER (lower is better)")
        axes[0].legend(fontsize=8)
        fig.suptitle("PaddleOCR reading order under skew: sorted_boxes vs "
                     "skew-robust (detection + recognition held identical)", y=1.02)
        plt.tight_layout()
        plt.savefig(_ROOT / "skew_sortfix.png", dpi=130, bbox_inches="tight")
        print("\nsaved skew_sortfix.png + skew_sortfix.csv")
    except Exception as ex:  # headless / no matplotlib
        print(f"\nsaved skew_sortfix.csv (figure skipped: {ex})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
