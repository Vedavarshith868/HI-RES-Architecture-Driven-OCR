"""Inference-speed benchmark: HI-RES pipeline vs a PaddleOCR baseline.

Both systems transcribe the SAME images on the SAME machine, so the timings are
directly comparable. HI-RES = PP-OCRv5 detection -> reading-order reconstruction
-> TrOCR-large recognition. The baseline = PaddleOCR's built-in det+rec pipeline,
defaulting to **PP-OCRv6 medium** (released June 2026: light, ~5x CPU speedup,
beats PP-OCRv5_server on accuracy). Pick another with --baseline, e.g. v5-server
or v6-small. PP-OCRv5 *server* OOMs / native-crashes on Colab, which is why the
lighter v6 default is preferred.

HI-RES is split into detect / order+crop / recognize, so you can say *why* one is
faster, not just *which*. Run on a GPU runtime (Colab T4): TrOCR is GPU-bound. The
first --warmup images are excluded (graph/JIT/CUDA warmup).

    python benchmark_speed.py --images gnhk/test --n 30
    python benchmark_speed.py --images pages --n 20 --baseline v5-server
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


# Baseline presets: (text_detection_model_name, text_recognition_model_name).
# None -> let PaddleOCR pick its default, which is **PP-OCRv6 medium** as of
# June 2026 (light, ~5x CPU speedup, beats PP-OCRv5_server on accuracy, and
# avoids the PP-OCRv5 *server* RAM blow-up / native crash on Colab).
BASELINE_PRESETS = {
    "v6": (None, None),
    "v6-medium": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
    "v6-small": ("PP-OCRv6_small_det", "PP-OCRv6_small_rec"),
    "v6-tiny": ("PP-OCRv6_tiny_det", "PP-OCRv6_tiny_rec"),
    "v5-server": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
    "v5-mobile": ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"),
}
# accept underscores too (v5_server == v5-server), a common typo
BASELINE_PRESETS.update({k.replace("-", "_"): v
                         for k, v in list(BASELINE_PRESETS.items()) if "-" in k})


class PaddleBaselineTimer:
    """Times PaddleOCR's built-in det+rec pipeline end-to-end.

    With no model names it uses PaddleOCR's current default (PP-OCRv6 medium as of
    June 2026) — lightweight and stable, unlike the PP-OCRv5 *server* models which
    OOM / native-crash on Colab. Pass det_model/rec_model (see BASELINE_PRESETS) to
    pin a specific tier or an older model."""

    def __init__(self, det_model: str | None = None, rec_model: str | None = None):
        from paddleocr import PaddleOCR
        base = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, enable_mkldnn=False)
        named = dict(base)
        if det_model:
            named["text_detection_model_name"] = det_model
        if rec_model:
            named["text_recognition_model_name"] = rec_model
        self.ocr = None
        last = None
        for kw in (named, base, dict(use_textline_orientation=False), {}):
            try:
                self.ocr = PaddleOCR(**kw)
                break
            except (TypeError, ValueError) as e:
                last = e
                continue
        if self.ocr is None:
            raise RuntimeError(f"could not init PaddleOCR baseline: {last}")

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
           device: str, beams: int, warmup: int,
           baseline_label: str = "PP-OCRv5 server") -> str:
    out: list[str] = []
    n = len(hires)
    det = [r["detect"] for r in hires]
    order = [r["order+crop"] for r in hires]
    rec = [r["recognize"] for r in hires]
    tot = [r["total"] for r in hires]
    boxes = [r["boxes"] for r in hires]

    if device == "cpu":
        out.append("!! WARNING: TrOCR-large ran on CPU — it is an autoregressive 558M model,\n"
                   "   ~50x slower on CPU than GPU. These HI-RES timings are a CPU worst case;\n"
                   "   use a T4 runtime for a representative recognize time (~1-2 s/page).")
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
        out.append(f"{baseline_label}  (built-in det+rec) — n={len(paddle)}")
        out.append(f"    {'total':<11} mean {pmean:7.3f}s   median {pmed:7.3f}s")
        out.append(f"    throughput {1.0 / pmean:5.2f} img/s")

        out.append("")
        rec_mean = _agg(rec)[0]
        slow = tmean / pmean if pmean else float("nan")
        rec_share = 100 * rec_mean / tmean if tmean else 0.0
        faster = "slower" if slow >= 1 else "faster"
        out.append(f"HI-RES is {slow:.1f}x {faster} than {baseline_label} "
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
                    help="skip the PaddleOCR baseline (HI-RES only)")
    ap.add_argument("--baseline", choices=list(BASELINE_PRESETS), default="v6",
                    help="PaddleOCR baseline (default v6 = PP-OCRv6 medium; "
                         "v5-server is heavy/OOM-prone on Colab)")
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

    baseline_label = f"PP-OCR[{args.baseline}]"
    det_m, rec_m = BASELINE_PRESETS[args.baseline]
    paddle = None
    if not args.no_baseline:
        try:
            print(f"init {baseline_label} pipeline...", flush=True)
            paddle = run_timer(PaddleBaselineTimer(det_m, rec_m), images,
                               args.warmup, baseline_label)
        except Exception as e:
            print(f"  (baseline skipped: {type(e).__name__}: {e})")

    print("\n" + report(hires, paddle, device, args.beams, args.warmup,
                        baseline_label))
    save_csv(hires, paddle, args.csv)
    print(f"\nper-image timings -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
