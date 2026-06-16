"""Multilingual OCR evaluation: HI-RES (PP-OCRv5 det + reading-order + PP-OCRv5
rec) vs the stock PaddleOCR(lang=...) pipeline, scored on the SAME pages.

Dataset: XFUND (multilingual forms — ZH/JA/ES/FR/IT/DE/PT — CC-BY-4.0, from
microsoft/unilm), which ships per-segment text + boxes, so we get document-level
ground truth in reading order. Metrics (CER / WER / WordAcc / sec-per-img) are
imported from evaluate.py, so they are defined identically to the English
benchmark.

CJK note: Chinese/Japanese/Korean are scored space-free (whitespace removed from
both GT and prediction) — the standard character-level CER for scripts without
inter-word spaces. WER is not meaningful there and should be ignored; CER is the
headline. Latin languages keep spaces, so both CER and WER are meaningful.

Get the data (one folder per language):
    # download {lang}.{split}.json + the matching images from XFUND, unzip here
    xfund_zh/zh.val.json + xfund_zh/zh_val_0.jpg ...

    python multilingual/ml_evaluate.py --xfund xfund_zh --lang ch
    python multilingual/ml_evaluate.py --xfund xfund_fr --lang fr --gt-order geom
    python multilingual/ml_evaluate.py --xfund xfund_ja --lang ja --n 40 --no-baseline
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# import the root harness + the sibling engine
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate import (IMAGE_EXTS, NormCfg, Sample, evaluate_system,
                      format_table, save_csv)
from ml_engine import MultilingualOcrEngine, _NO_SPACE_LANGS, rec_model_for


# --------------------------------------------------------------------------
# XFUND loading
# --------------------------------------------------------------------------

def _xfund_docs(data) -> list[dict]:
    """XFUND json: usually {'documents': [...]}, sometimes a bare list."""
    if isinstance(data, dict):
        for key in ("documents", "data"):
            if isinstance(data.get(key), list):
                return data[key]
    if isinstance(data, list):
        return data
    return []


def _entity_box(ent: dict) -> list[float]:
    b = ent.get("box") or [0, 0, 0, 0]
    return [float(x) for x in b[:4]] if len(b) >= 4 else [0, 0, 0, 0]


def _doc_text(doc: dict, gt_order: str) -> str:
    """Page ground truth from a document's segments.

    gt_order='list' keeps the dataset's annotation order (the human reading
    order). gt_order='geom' re-sorts segments by (center-y, x) — a geometry
    reference, useful to check ordering on its own terms."""
    ents = [e for e in doc.get("document", []) if str(e.get("text", "")).strip()]
    if gt_order == "geom":
        ents.sort(key=lambda e: ((_entity_box(e)[1] + _entity_box(e)[3]) / 2.0,
                                 _entity_box(e)[0]))
    return "\n".join(str(e["text"]) for e in ents)


def _find_image(folder: Path, fname: str | None) -> Path | None:
    if not fname:
        return None
    cand = folder / fname
    if cand.is_file():
        return cand
    stem = Path(fname).stem
    for p in folder.rglob(Path(fname).name):
        if p.is_file():
            return p
    for ext in IMAGE_EXTS:
        for p in folder.rglob(stem + ext):
            if p.is_file():
                return p
    return None


def load_xfund(folder: str | Path, lang: str, split: str = "val",
               gt_order: str = "list", n: int | None = None) -> list[Sample]:
    folder = Path(folder)
    no_space = lang in _NO_SPACE_LANGS
    jsons = sorted(folder.rglob(f"*{split}*.json")) or sorted(folder.rglob("*.json"))
    samples: list[Sample] = []
    for jf in jsons:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for doc in _xfund_docs(data):
            img = _find_image(folder, (doc.get("img") or {}).get("fname")
                              or doc.get("id"))
            if img is None:
                continue
            gt = _doc_text(doc, gt_order)
            if no_space:
                gt = re.sub(r"\s+", "", gt)
            if gt.strip():
                samples.append(Sample(name=Path(img).stem, image_path=img, gt=gt))
            if n is not None and len(samples) >= n:
                return samples
    return samples


# --------------------------------------------------------------------------
# predictors
# --------------------------------------------------------------------------

def hires_predict(engine: MultilingualOcrEngine, no_space: bool):
    def _f(img):
        text = engine.run(img, make_visuals=False)["text"]
        return re.sub(r"\s+", "", text) if no_space else text
    return _f


def _init_paddle(lang: str, det_model: str | None = None,
                 rec_model: str | None = None):
    """Stock PaddleOCR full pipeline (PIR+oneDNN guard applied).

    Pass det_model/rec_model to force the SAME models HI-RES uses — that turns the
    comparison into a controlled one where the only differences are HI-RES's
    reading order, keep-every-box, and crop method (a fair test of the layer).
    With both None, PaddleOCR loads its (heavier) defaults for `lang`."""
    from paddleocr import PaddleOCR
    base = dict(enable_mkldnn=False, use_doc_orientation_classify=False,
                use_doc_unwarping=False, use_textline_orientation=False)
    named = dict(base)
    if det_model:
        named["text_detection_model_name"] = det_model
    if rec_model:
        named["text_recognition_model_name"] = rec_model
    attempts = ([named] if (det_model or rec_model) else []) + [
        dict(base, lang=lang), base, dict(lang=lang), {}]
    last = None
    for kw in attempts:
        try:
            return PaddleOCR(**kw)
        except (TypeError, ValueError) as e:
            last = e
            continue
    raise RuntimeError(f"could not init PaddleOCR(lang={lang!r}): {last}")


def _paddle_texts(ocr, bgr) -> list[str]:
    if hasattr(ocr, "predict"):
        try:
            out: list[str] = []
            for r in ocr.predict(bgr):
                texts = r.get("rec_texts") if hasattr(r, "get") else None
                if texts:
                    out.extend(texts)
            return out
        except Exception:
            pass
    out = []
    for page in ocr.ocr(bgr) or []:
        for line in page or []:
            try:
                out.append(line[1][0])
            except (IndexError, TypeError):
                pass
    return out


def builtin_predict(lang: str, no_space: bool, det_model: str | None = None,
                    rec_model: str | None = None):
    ocr = _init_paddle(lang, det_model, rec_model)

    def _f(img):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        texts = _paddle_texts(ocr, bgr)
        if no_space:
            return re.sub(r"\s+", "", "".join(texts))
        return "\n".join(texts)
    return _f


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xfund", required=True, help="folder with {lang}.{split}.json + images")
    ap.add_argument("--lang", required=True, help="PP-OCR language code (ch, ja, fr, de, ...)")
    ap.add_argument("--split", default="val", help="XFUND split substring (default: val)")
    ap.add_argument("--gt-order", choices=["list", "geom"], default="list",
                    help="ground-truth reading order: dataset list order (default) or geometric")
    ap.add_argument("--n", type=int, default=None, help="cap number of pages")
    ap.add_argument("--det-model", default="PP-OCRv5_mobile_det",
                    help="PP-OCR detection model for HI-RES (mobile = fast on CPU)")
    ap.add_argument("--rec-model", default=None, help="override PP-OCR rec model name")
    ap.add_argument("--controlled", action="store_true",
                    help="give stock PaddleOCR the SAME det+rec models as HI-RES, "
                         "isolating the reading-order layer (a fair, apples-to-apples test)")
    ap.add_argument("--no-baseline", action="store_true", help="skip stock PaddleOCR")
    ap.add_argument("--strip-punct", action="store_true")
    ap.add_argument("--csv", default="ml_eval_results.csv")
    args = ap.parse_args()

    no_space = args.lang in _NO_SPACE_LANGS
    samples = load_xfund(args.xfund, args.lang, split=args.split,
                         gt_order=args.gt_order, n=args.n)
    print(f"loaded {len(samples)} XFUND pages from {args.xfund} "
          f"[lang={args.lang}, split~{args.split}, gt-order={args.gt_order}]")
    if not samples:
        print("No (image, json) pairs found — check the folder and --split.")
        return 1

    norm = NormCfg(strip_punct=args.strip_punct)
    rec = args.rec_model or rec_model_for(args.lang)
    print(f"HI-RES models: det={args.det_model} rec={rec}"
          + ("  | stock: SAME models (controlled)" if args.controlled
             else "  | stock: PaddleOCR defaults"))
    engine = MultilingualOcrEngine(lang=args.lang, det_model=args.det_model,
                                   rec_model=args.rec_model)

    scores = [evaluate_system(f"hires-ml[{args.lang}]", samples,
                              hires_predict(engine, no_space), norm=norm, progress=True)]
    if not args.no_baseline:
        det_m = args.det_model if args.controlled else None
        rec_m = rec if args.controlled else None
        try:
            scores.append(evaluate_system(
                f"paddle-stock[{args.lang}]", samples,
                builtin_predict(args.lang, no_space, det_m, rec_m),
                norm=norm, progress=True))
        except Exception as e:
            print(f"  (stock PaddleOCR skipped: {type(e).__name__}: {e})")

    print("\n" + format_table(scores))
    if no_space:
        print("\n(CJK: scored space-free; WER is not meaningful — read CER.)")
    save_csv(scores, args.csv)
    print(f"per-page breakdown -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
