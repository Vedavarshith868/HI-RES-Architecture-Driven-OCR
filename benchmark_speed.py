"""Inference-speed benchmark: HI-RES pipeline vs the PP-OCRv5 *server* pipeline.

Both systems transcribe the SAME images on the SAME machine, so the timings are
directly comparable. HI-RES = PP-OCRv5 detection -> reading-order reconstruction
-> TrOCR-large recognition. The baseline = PaddleOCR's built-in PP-OCRv5 server
det+rec pipeline (the strongest-accuracy baseline, ~28% CER on GNHK).

Because both use the same detector, the timing gap is the recognizer plus the
(negligible) reading-order geometry. This benchmark makes that explicit by
splitting HI-RES into detect / order+crop / recognize, so you can say *why* one
is faster rather than just *which*.

Run on a GPU runtime (Colab T4): TrOCR is GPU-bound, and PP-OCRv5 server rec is
unstable on CPU. The first --warmup images are excluded (graph/JIT/CUDA warmup).

    python benchmark_speed.py --images gnhk/test --n 30
    python benchmark_speed.py --images pages --n 20 --warmup 3 --beams 1
    python benchmark_speed.py --images pages --n 20 --no-baseline   # HI-RES only
"""
from __future__ import annotations

import argparse
import statistics as stats
import time
from pathlib import Path

import cv2
import numpy as np

import pipeline
from evaluate import IMAGE_EXTS
from ocr_engine import (ASPECT_CAP, MERGE_HEIGHT_GUARD, DET_MODEL_NAME,
                        Detector, Recognizer)


def _cuda_sync() -> None:
    """Flush pending GPU work so wall-clock timing reflects real compute."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def list_images(folder: str | Path, n: int | None = None) -> list[Path]:
    folder = Path(folder)
    imgs = [p for p in sorted(folder.rglob("*"))
            if p.suffix.lower() in IMAGE_EXTS
            and not any(t in p.stem for t in ("_panel", "_overlay"))]
    return imgs[:n] if n else imgs


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_crops(img_rgb: np.ndarray, quads: np.ndarray) -> list[np.ndarray]:
    """Reproduce OcrEngine.run's crop construction *without* the debug overlay /
    transcript rendering, so we time exactly the production recognition path:
    reading order -> line chunking -> perspective crops."""
    lines, theta = pipeline.reading_order(quads)
    ordered = [pipeline.order_points(q) for q in quads]
    deskewed = [pipeline.rotate_points(q, -theta) for q in ordered]
    med_h = float(np.median([pipeline.quad_size(q)[1] for q in ordered]))
    crops: list[np.ndarray] = []
    for line in lines:
        for chunk in pipeline.chunk_line(line, deskewed, aspect_cap=ASPECT_CAP):
            merged = pipeline.merge_quads([ordered[i] for i in chunk])
            crop = None
            if pipeline.quad_size(merged)[1] <= MERGE_HEIGHT_GUARD * med_h:
                crop = pipeline.perspective_crop(img_rgb, merged)
            if crop is not None:
                crops.append(crop)
            else:
                for i in chunk:
                    h = pipeline.quad_size(ordered[i])[1]
                    c = pipeline.perspective_crop(img_rgb, ordered[i],
                                                  allow_rot90=h > 2.2 * med_h)
                    if c is not None:
                        crops.append(c)
    return crops


class HiResTimer:
    """Times HI-RES per stage: detect, order+crop, recognize."""

    def __init__(self, beams: int = 1, device: str | None = None):
        self.detector = Detector()
        self.recognizer = Recognizer(device=device)
        self.beams = beams
        self.device = str(self.recognizer.device)

    def __call__(self, img_rgb: np.ndarray) -> dict:
        t0 = time.perf_counter()
        quads = self.detector(img_rgb)
        _cuda_sync()
        t1 = time.perf_counter()
        crops = build_crops(img_rgb, quads) if len(quads) else [img_rgb]
        t2 = time.perf_counter()
        self.recognizer(crops, num_beams=self.beams)
        _cuda_sync()
        t3 = time.perf_counter()
        return {"detect": t1 - t0, "order+crop": t2 - t1, "recognize": t3 - t2,
                "total": t3 - t0, "boxes": int(len(quads))}


class PaddleServerTimer:
    """Times PaddleOCR's built-in PP-OCRv5 server det+rec pipeline (end-to-end)."""

    def __init__(self):
        from paddleocr import PaddleOCR
        self.ocr = None
        last = None
        for kw in (
            dict(lang="en", enable_mkldnn=False,
                 text_detection_model_name="PP-OCRv5_server_det",
                 text_recognition_model_name="PP-OCRv5_server_rec",
                 use_doc_orientation_classify=False, use_doc_unwarping=False,
                 use_textline_orientation=False),
            dict(lang="en", enable_mkldnn=False),
            dict(lang="en"),
        ):
            try:
                self.ocr = PaddleOCR(**kw)
                break
            except (TypeError, ValueError) as e:
                last = e
                continue
        if self.ocr is None:
            raise RuntimeError(f"could not init PaddleOCR server pipeline: {last}")

    def __call__(self, img_rgb: np.ndarray) -> dict:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        t0 = time.perf_counter()
        if hasattr(self.ocr, "predict"):
            self.ocr.predict(bgr)
        else:
            self.ocr.ocr(bgr)
        _cuda_sync()
        return {"total": time.perf_counter() - t0}


def _agg(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return stats.fmean(values), stats.median(values)


def run_timer(timer, images: list[np.ndarray], warmup: int, label: str) -> list[dict]:
    print(f"  warming up {label} ({warmup} img)...", flush=True)
    for img in images[:warmup]:
        timer(img)
    print(f"  timing {label} ({len(images) - warmup} img)...", flush=True)
    return [timer(img) for img in images[warmup:]]


def report(hires: list[dict], paddle: list[dict] | None,
           device: str, beams: int, warmup: int) -> str:
    out: list[str] = []
    n = len(hires)
    det = [r["detect"] for r in hires]
    order = [r["order+crop"] for r in hires]
    rec = [r["recognize"] for r in hires]
    tot = [r["total"] for r in hires]
    boxes = [r["boxes"] for r in hires]

    out.append(f"HI-RES  (PP-OCRv5 det + reading-order + TrOCR-large, beams={beams}, "
               f"device={device}) — n={n}, warmup={warmup}")
    for name, vals in (("detect", det), ("order+crop", order),
                       ("recognize", rec), ("total", tot)):
        mean, med = _agg(vals)
        out.append(f"    {name:<11} mean {mean:7.3f}s   median {med:7.3f}s")
    tmean, _ = _agg(tot)
    out.append(f"    throughput {1.0 / tmean:5.2f} img/s   "
               f"(avg {stats.fmean(boxes):.0f} boxes/page, "
               f"reading-order overhead {1000 * _agg(order)[0]:.1f} ms/page)")

    if paddle:
        pt = [r["total"] for r in paddle]
        pmean, pmed = _agg(pt)
        out.append("")
        out.append(f"PP-OCRv5 server  (built-in det+rec) — n={len(paddle)}")
        out.append(f"    {'total':<11} mean {pmean:7.3f}s   median {pmed:7.3f}s")
        out.append(f"    throughput {1.0 / pmean:5.2f} img/s")

        out.append("")
        rec_mean = _agg(rec)[0]
        slow = tmean / pmean if pmean else float("nan")
        rec_share = 100 * rec_mean / tmean if tmean else 0.0
        faster = "slower" if slow >= 1 else "faster"
        out.append(f"HI-RES is {slow:.1f}x {faster} than PP-OCRv5 server "
                   f"({tmean:.3f}s vs {pmean:.3f}s per page).")
        out.append(f"Recognition is {rec_share:.0f}% of HI-RES time; "
                   f"detection is shared, reading-order is "
                   f"{1000 * _agg(order)[0]:.1f} ms — negligible.")
    return "\n".join(out)


def save_csv(hires: list[dict], paddle: list[dict] | None, path: str) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "img_index", "detect", "order_crop",
                    "recognize", "total", "boxes"])
        for i, r in enumerate(hires):
            w.writerow(["hires", i, r["detect"], r["order+crop"],
                        r["recognize"], r["total"], r["boxes"]])
        for i, r in enumerate(paddle or []):
            w.writerow(["ppocrv5-server", i, "", "", "", r["total"], ""])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of images (recursed)")
    ap.add_argument("--n", type=int, default=30, help="cap images (incl. warmup)")
    ap.add_argument("--warmup", type=int, default=2, help="warmup images, excluded")
    ap.add_argument("--beams", type=int, default=1, help="TrOCR beams (1 = greedy)")
    ap.add_argument("--device", default=None, help="torch device for TrOCR")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the PP-OCRv5 server pipeline (HI-RES only)")
    ap.add_argument("--csv", default="speed_benchmark.csv")
    args = ap.parse_args()

    paths = list_images(args.images, args.n)
    if len(paths) <= args.warmup:
        print(f"need > {args.warmup} images, found {len(paths)} in {args.images}")
        return 1
    print(f"loaded {len(paths)} images from {args.images}")
    images = [read_rgb(p) for p in paths]  # decode once; excluded from timing

    print("init HI-RES (detector + TrOCR)...", flush=True)
    hires_timer = HiResTimer(beams=args.beams, device=args.device)
    device = hires_timer.device
    hires = run_timer(hires_timer, images, args.warmup, "HI-RES")

    paddle = None
    if not args.no_baseline:
        try:
            print("init PP-OCRv5 server pipeline...", flush=True)
            paddle = run_timer(PaddleServerTimer(), images, args.warmup,
                               "PP-OCRv5 server")
        except Exception as e:
            print(f"  (baseline skipped: {type(e).__name__}: {e})")

    print("\n" + report(hires, paddle, device, args.beams, args.warmup))
    save_csv(hires, paddle, args.csv)
    print(f"\nper-image timings -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
