"""OCR evaluation harness: CER / WER + an order-free recognition diagnostic,
plus baseline runners so several systems are scored on the *same* data.

Metrics
-------
- **CER** (Character Error Rate) = Levenshtein(ref, hyp) / len(ref), aggregated
  at corpus level (sum of edits / sum of ref chars — the standard way). Lower
  is better; can exceed 100% when a system hallucinates. This is the headline.
- **WER** (Word Error Rate) = same, on whitespace tokens.
- **WordAcc** = order-free word accuracy = fraction of reference words present
  in the prediction regardless of position (multiset intersection / n_ref).
  Higher is better. This separates recognition from reading order WITHOUT the
  paradoxes of a char-level "order-invariant CER": sorting words/chars to remove
  order misaligns recognition errors and can exceed the document CER, so such a
  decomposition is ill-posed on real data. WordAcc is a clean, bounded [0,1]
  signal — high WordAcc with high CER => ordering/segmentation issue; low
  WordAcc => genuine recognition issue. It is NOT subtracted from CER.

A "system" is just a function image_rgb -> text (and optionally -> list of
lines, for the order diagnostic). Their pipeline, Tesseract, EasyOCR and
PaddleOCR are all wired up below behind lazy imports.

CLI:
    # your own labeled folder
    python evaluate.py --data DIR
    python evaluate.py --data DIR --baselines tesseract,easyocr,paddleocr --csv out.csv
    # a public dataset (Hugging Face) — IAM test lines, quick 200-sample check:
    python evaluate.py --hf iam-lines --n 200
    python evaluate.py --hf iam-lines            # full IAM test (2,920 lines)

--data layout: each image `foo.jpg` has its ground-truth transcript in a sibling
text file `foo.txt` (or `foo.gt.txt`), one physical line per line, in reading
order. Feed page images for document-level scores, or single-line crops for
line-level scores.

--hf: a preset in HF_DATASETS (currently 'iam-lines' = Teklia/IAM-line) or any
hub dataset id with an image column + a text column (auto-detected). Line-level
datasets are scored on the recognizer alone (no detection/ordering), directly
comparable to published TrOCR CER (~2.9% on IAM test).
"""

from __future__ import annotations

import argparse
import re
import string
import time
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# edit distance + normalization
# --------------------------------------------------------------------------

def levenshtein(a, b) -> int:
    """Edit distance between two sequences (strings for CER, token lists for WER)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def normalize(s: str, lowercase: bool = False, strip_punct: bool = False,
              collapse_ws: bool = True) -> str:
    """Text normalization applied to BOTH ref and hyp before scoring.

    Default collapses runs of whitespace (incl. newlines) to single spaces so
    detection/segmentation formatting differences don't inflate CER, while
    leaving case and punctuation intact (standard CER). Flags let you relax it.
    """
    if lowercase:
        s = s.lower()
    if strip_punct:
        s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip() if collapse_ws else s.strip()
    return s


@dataclass
class NormCfg:
    lowercase: bool = False
    strip_punct: bool = False
    collapse_ws: bool = True

    def __call__(self, s: str) -> str:
        return normalize(s, self.lowercase, self.strip_punct, self.collapse_ws)


def cer_counts(ref: str, hyp: str, norm: NormCfg) -> tuple[int, int]:
    r, h = norm(ref), norm(hyp)
    return levenshtein(r, h), len(r)


def wer_counts(ref: str, hyp: str, norm: NormCfg) -> tuple[int, int]:
    r, h = norm(ref).split(), norm(hyp).split()
    return levenshtein(r, h), len(r)


def word_acc_counts(ref: str, hyp: str, norm: NormCfg) -> tuple[int, int]:
    """Order-free recognition signal: (matched, n_ref_words) where `matched` is
    the multiset intersection of reference and predicted words.

    word accuracy = matched / n_ref_words = fraction of ground-truth words the
    system produced *regardless of position*. This isolates recognition quality
    from reading order without the paradoxes of a char-level "order-invariant
    CER" (sorting characters/words misaligns recognition errors and can exceed
    the document CER). High word-accuracy with high CER => ordering/segmentation
    problem; low word-accuracy => genuine recognition problem."""
    from collections import Counter
    r = Counter(norm(ref).split())
    h = Counter(norm(hyp).split())
    matched = sum((r & h).values())
    return matched, sum(r.values())


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

@dataclass
class Sample:
    name: str
    gt: str
    image_path: Path | None = None
    image: object | None = None  # in-memory PIL.Image or RGB np.ndarray (optional)

    def image_rgb(self):
        import cv2
        import numpy as np
        if self.image is not None:
            arr = np.asarray(self.image)
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            return np.ascontiguousarray(arr)
        bgr = cv2.imdecode(np.fromfile(self.image_path, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not decode {self.image_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_pairs(folder: str | Path) -> list[Sample]:
    """Local folder: each image has a sibling .txt / .gt.txt ground truth."""
    folder = Path(folder)
    samples: list[Sample] = []
    for img in sorted(folder.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        if any(t in img.stem for t in ("_panel", "_overlay")):  # skip our outputs
            continue
        gt_path = None
        for cand in (img.with_suffix(".gt.txt"), img.with_suffix(".txt"),
                     img.with_name(img.stem + ".gt.txt")):
            if cand.is_file():
                gt_path = cand
                break
        if gt_path is None:
            print(f"  (skip {img.name}: no ground-truth .txt beside it)")
            continue
        samples.append(Sample(name=img.name, image_path=img,
                              gt=gt_path.read_text(encoding="utf-8")))
    return samples


# verified-good presets; `level='line'` -> score the recognizer directly,
# `level='page'` -> score the full detection+ordering+recognition pipeline.
HF_DATASETS = {
    "iam-lines": dict(path="Teklia/IAM-line", image_col="image", text_col="text",
                      split="test", level="line",
                      note="IAM test lines; compare to TrOCR-large CER ~2.9%"),
    "iam-sentences": dict(path="alpayariyak/IAM_Sentences", image_col="image",
                          text_col="text", split="train", level="page",
                          note="IAM sentences as MULTI-LINE images; tests the full "
                               "pipeline. Recognition CER is optimistic (TrOCR trained "
                               "on IAM); detection+ordering test is valid."),
}

_IMG_COL_HINTS = ("image", "img", "im", "picture")
_TXT_COL_HINTS = ("text", "label", "transcription", "transcript", "sentence",
                  "gt", "ground_truth", "caption")


def _auto_col(row: dict, hints) -> str:
    keys = list(row.keys())
    for h in hints:
        for k in keys:
            if k.lower() == h:
                return k
    for h in hints:
        for k in keys:
            if h in k.lower():
                return k
    raise KeyError(f"could not auto-detect column from {keys} (hints={hints}); "
                   f"pass image_col=/text_col= explicitly")


def load_hf(name_or_path: str, split: str | None = None, n: int | None = None,
            image_col: str | None = None, text_col: str | None = None,
            config: str | None = None, streaming: bool = True) -> list[Sample]:
    """Load image+text pairs from a Hugging Face dataset.

    `name_or_path` may be a preset key in HF_DATASETS (e.g. 'iam-lines') or any
    hub dataset id exposing an image column and a text column. `n` caps the
    number of samples (streaming, so it won't download the whole set for a
    quick check). Columns and split are auto-filled from the preset / detected.
    """
    from datasets import load_dataset

    preset = HF_DATASETS.get(name_or_path, {})
    path = preset.get("path", name_or_path)
    image_col = image_col or preset.get("image_col")
    text_col = text_col or preset.get("text_col")
    config = config or preset.get("config")
    split = split or preset.get("split", "test")

    ds = load_dataset(path, name=config, split=split, streaming=streaming)
    samples: list[Sample] = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        ic = image_col or _auto_col(row, _IMG_COL_HINTS)
        tc = text_col or _auto_col(row, _TXT_COL_HINTS)
        samples.append(Sample(name=f"{path}:{split}:{i}", gt=str(row[tc]),
                              image=row[ic]))
    return samples


def is_line_level(name_or_path: str) -> bool:
    return HF_DATASETS.get(name_or_path, {}).get("level") == "line"


def build_iam_pages(n_pages: int = 20, lines_per_page: tuple[int, int] = (4, 8),
                    max_skew_deg: float = 3.0, gap_px: int = 22, margin_px: int = 50,
                    max_indent_px: int = 60, seed: int = 0,
                    source_split: str = "test") -> list[Sample]:
    """Synthesize multi-line 'pages' by stacking real IAM line images, with
    random per-line indent and a small page skew. The skew makes naive
    top-to-bottom ordering fail, so a low CER here means `pipeline.reading_order`
    survived; WordAcc stays high since the stacking doesn't change recognition.

    Recognition CER is still IAM-optimistic (TrOCR trained on IAM); the point of
    this set is to measure detection + line clustering + ordering."""
    import cv2
    import numpy as np
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    it = iter(load_dataset("Teklia/IAM-line", split=source_split, streaming=True))
    pages: list[Sample] = []
    for p in range(n_pages):
        k = int(rng.integers(lines_per_page[0], lines_per_page[1] + 1))
        rows, texts = [], []
        for _ in range(k):
            row = next(it)
            rows.append(np.asarray(row["image"].convert("L")))
            texts.append(row["text"])
        indents = [int(rng.integers(0, max_indent_px + 1)) for _ in rows]
        width = margin_px * 2 + max(r.shape[1] + indents[i] for i, r in enumerate(rows))
        height = margin_px * 2 + sum(r.shape[0] for r in rows) + gap_px * (k - 1)
        canvas = np.full((height, width), 255, np.uint8)
        y = margin_px
        for r, ind in zip(rows, indents):
            x = margin_px + ind
            canvas[y:y + r.shape[0], x:x + r.shape[1]] = r
            y += r.shape[0] + gap_px
        if max_skew_deg:
            ang = float(rng.uniform(-max_skew_deg, max_skew_deg))
            m = cv2.getRotationMatrix2D((width / 2, height / 2), ang, 1.0)
            canvas = cv2.warpAffine(canvas, m, (width, height),
                                    borderValue=255, flags=cv2.INTER_LINEAR)
        pages.append(Sample(name=f"iam-page-{p:03d}", gt="\n".join(texts),
                            image=cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)))
    return pages


def _gnhk_words(data) -> list[dict]:
    """GNHK json is normally a list of word dicts; tolerate a dict wrapper."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    return []


def _gnhk_reading_order_text(words: list[dict], drop_special: bool = True,
                             keep_types: set[str] | None = None) -> str:
    """Reconstruct a page's reading-order transcript from GNHK word annotations
    (each: 'text', 'polygon' {x0,y0..x3,y3}, 'line_idx', 'type'). Groups by
    line_idx, orders lines by mean-y, words within a line by x.

    `%...%` tokens (%NA%/%math%/%SC%/...) mark non-transcribable content and are
    dropped by default — same as the GNHK recognition benchmark. `keep_types`
    optionally restricts to certain region types (e.g. {'H'} for handwritten),
    once you've seen the type codes in the real data."""
    lines: dict[int, list[tuple[float, float, str]]] = {}
    for w in words:
        t = str(w.get("text", "")).strip()
        if not t or (drop_special and t.startswith("%") and t.endswith("%")):
            continue
        if keep_types is not None and w.get("type") not in keep_types:
            continue
        poly = w.get("polygon", {}) or {}
        xs = [poly.get(f"x{i}", 0) for i in range(4)]
        ys = [poly.get(f"y{i}", 0) for i in range(4)]
        lines.setdefault(int(w.get("line_idx", -1)), []).append(
            (min(xs), min(ys), t))
    ordered = sorted(lines.values(),
                     key=lambda ws: sum(y for _, y, _ in ws) / len(ws))
    return "\n".join(" ".join(t for _, _, t in sorted(ws, key=lambda e: e[0]))
                     for ws in ordered)


def load_gnhk(folder: str | Path, drop_special: bool = True,
              keep_types: set[str] | None = None) -> list[Sample]:
    """Load GNHK (real-photo handwriting) pages from a locally downloaded folder
    (https://goodnotes.com/gnhk — agree to the CC-BY-4.0 terms, then unzip the
    train/test zips here). Expects image files with sibling .json word
    annotations; GT is reconstructed in reading order from line_idx."""
    import json
    folder = Path(folder)
    samples: list[Sample] = []
    for jf in sorted(folder.rglob("*.json")):
        img = next((c for ext in IMAGE_EXTS
                    for c in [jf.with_suffix(ext)] if c.is_file()), None)
        if img is None:
            continue
        try:
            words = _gnhk_words(json.loads(jf.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
        gt = _gnhk_reading_order_text(words, drop_special, keep_types)
        if gt.strip():
            samples.append(Sample(name=jf.stem, image_path=img, gt=gt))
    return samples


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

@dataclass
class Score:
    system: str
    n: int = 0
    cer_edits: int = 0
    cer_ref: int = 0
    wer_edits: int = 0
    wer_ref: int = 0
    wa_matched: int = 0
    wa_total: int = 0
    seconds: float = 0.0
    per_item: list[dict] = field(default_factory=list)

    @property
    def cer(self) -> float:
        return self.cer_edits / max(self.cer_ref, 1)

    @property
    def wer(self) -> float:
        return self.wer_edits / max(self.wer_ref, 1)

    @property
    def word_acc(self) -> float | None:
        # order-free fraction of reference words recovered (higher is better)
        return self.wa_matched / self.wa_total if self.wa_total else None

    @property
    def sec_per_img(self) -> float:
        return self.seconds / max(self.n, 1)


def evaluate_system(name: str, samples: list[Sample], predict_text,
                    predict_lines=None, norm: NormCfg | None = None,
                    progress: bool = False, word_metrics: bool = True) -> Score:
    """Score one system. predict_text(image_rgb)->str. WordAcc is computed
    order-free from the text, so predict_lines is not required (kept for
    backward compatibility, ignored).

    word_metrics=False skips WER and WordAcc — set it for **space-free CJK**,
    where there are no word boundaries: stripping spaces makes the whole page one
    token, so WER collapses to 100% and WordAcc to 0% (meaningless). Only CER
    (character-level) is valid there; the table then shows WER/WordAcc as '—'.

    progress=True prints a line per image (running CER + per-image seconds) so a
    slow run — e.g. heavy PaddleOCR models on a CPU runtime — shows it is working,
    not hung."""
    norm = norm or NormCfg()
    sc = Score(system=name)
    total = len(samples)
    for s in samples:
        img = s.image_rgb()
        t0 = time.perf_counter()
        hyp = predict_text(img)
        dt = time.perf_counter() - t0

        ce, cr = cer_counts(s.gt, hyp, norm)
        sc.n += 1
        sc.cer_edits += ce; sc.cer_ref += cr
        sc.seconds += dt
        item = {"system": name, "image": s.name, "cer": ce / max(cr, 1),
                "ref_chars": cr, "seconds": round(dt, 3)}
        if word_metrics:
            we, wr = wer_counts(s.gt, hyp, norm)
            wm, wt = word_acc_counts(s.gt, hyp, norm)
            sc.wer_edits += we; sc.wer_ref += wr
            sc.wa_matched += wm; sc.wa_total += wt
            item.update(wer=we / max(wr, 1), word_acc=wm / max(wt, 1))
        sc.per_item.append(item)
        if progress:
            print(f"  [{name}] {sc.n}/{total} {s.name}: cer={sc.cer:.3f} "
                  f"({dt:.1f}s)", flush=True)
    return sc


# --------------------------------------------------------------------------
# systems under test (lazy imports — none required unless used)
# --------------------------------------------------------------------------

def recognizer_predictor(recognizer=None, **rec_kwargs):
    """predict_text using ONLY the TrOCR recognizer (no detection/ordering).
    Use on single-line images (e.g. IAM lines) for a recognizer-quality number
    directly comparable to published TrOCR CER."""
    if recognizer is None:
        from handwritten.engine import Recognizer
        recognizer = Recognizer(**rec_kwargs)
    return lambda img: recognizer([img])[0]


def evaluate_recognizer(name: str, samples: list[Sample], recognizer=None,
                        norm: NormCfg | None = None, batch_size: int = 16) -> Score:
    """Fast batched recognizer-only scoring for line datasets — runs the whole
    set through TrOCR in batches instead of one image at a time."""
    norm = norm or NormCfg()
    if recognizer is None:
        from handwritten.engine import Recognizer
        recognizer = Recognizer()
    imgs = [s.image_rgb() for s in samples]
    t0 = time.perf_counter()
    hyps: list[str] = []
    for i in range(0, len(imgs), batch_size):
        hyps.extend(recognizer(imgs[i:i + batch_size], batch_size=batch_size))
    dt = time.perf_counter() - t0

    sc = Score(system=name, seconds=dt)
    for s, hyp in zip(samples, hyps):
        ce, cr = cer_counts(s.gt, hyp, norm)
        we, wr = wer_counts(s.gt, hyp, norm)
        wm, wt = word_acc_counts(s.gt, hyp, norm)
        sc.n += 1
        sc.cer_edits += ce; sc.cer_ref += cr
        sc.wer_edits += we; sc.wer_ref += wr
        sc.wa_matched += wm; sc.wa_total += wt
        sc.per_item.append({"system": name, "image": s.name, "cer": ce / max(cr, 1),
                            "wer": we / max(wr, 1), "word_acc": wm / max(wt, 1),
                            "ref_chars": cr})
    return sc


def pipeline_predictors(engine=None, **engine_kwargs):
    """Return (predict_text, predict_lines) for THIS project's pipeline."""
    if engine is None:
        from handwritten.engine import OcrEngine
        engine = OcrEngine(**engine_kwargs)
    cache: dict[int, dict] = {}

    def _run(img):
        key = id(img)
        if key not in cache:
            cache.clear()
            cache[key] = engine.run(img)
        return cache[key]

    return (lambda img: _run(img)["text"],
            lambda img: [l["text"] for l in _run(img)["lines"]])


def baseline_predict(name: str):
    """Return predict_text for a baseline OCR system (lazy, raises if missing)."""
    name = name.lower()
    if name == "tesseract":
        import pytesseract
        from PIL import Image
        return lambda img: pytesseract.image_to_string(Image.fromarray(img))
    if name == "easyocr":
        import easyocr
        reader = easyocr.Reader(["en"], gpu=True)
        return lambda img: "\n".join(reader.readtext(img, detail=0, paragraph=True))
    if name == "paddleocr":
        import cv2
        from paddleocr import PaddleOCR
        # enable_mkldnn=False avoids paddle's PIR+oneDNN crash
        # (ConvertPirAttribute2RuntimeAttribute), same fix Detector uses.
        ocr = None
        for kw in (dict(lang="en", enable_mkldnn=False, use_doc_orientation_classify=False,
                        use_doc_unwarping=False, use_textline_orientation=False),
                   dict(lang="en", enable_mkldnn=False),
                   dict(use_angle_cls=False, lang="en", enable_mkldnn=False),  # 2.x
                   dict(lang="en")):
            try:
                ocr = PaddleOCR(**kw)
                break
            except TypeError:
                continue
        if ocr is None:
            raise RuntimeError("could not initialize PaddleOCR")
        def _run(img):
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if hasattr(ocr, "predict"):                      # 3.x: list of dict-like
                try:
                    out = []
                    for r in ocr.predict(bgr):
                        texts = r.get("rec_texts") if hasattr(r, "get") else None
                        if texts:
                            out.extend(texts)
                    return "\n".join(out)
                except Exception:
                    pass
            res = ocr.ocr(bgr)                               # 2.x: [[[box,(text,score)],...]]
            out = []
            for page in res or []:
                for line in page or []:
                    try:
                        out.append(line[1][0])
                    except (IndexError, TypeError):
                        pass
            return "\n".join(out)
        return _run
    raise ValueError(f"unknown baseline: {name}")


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def format_table(scores: list[Score]) -> str:
    # Standard OCR metrics: CER and WER (both corpus-level Levenshtein, lower is better).
    # WordAcc is kept in the CSV for per-image diagnostics but not shown in the table —
    # it is order-free and not a standard benchmark metric.
    head = f"{'system':<20}{'CER':>8}{'WER':>8}{'sec/img':>9}{'n':>5}"
    rows = [head, "-" * len(head)]
    for s in sorted(scores, key=lambda x: x.cer):
        wer = f"{s.wer:>8.1%}" if s.wer_ref else f"{'—':>8}"
        rows.append(f"{s.system:<20}{s.cer:>8.1%}{wer}{s.sec_per_img:>9.2f}{s.n:>5}")
    return "\n".join(rows)


def save_csv(scores: list[Score], path: str | Path) -> None:
    import csv
    rows = [it for s in scores for it in s.per_item]
    if not rows:
        return
    keys = ["system", "image", "cer", "wer", "word_acc", "ref_chars", "seconds"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="folder of images + sibling .txt ground truth")
    src.add_argument("--hf", help=f"HF dataset: preset ({', '.join(HF_DATASETS)}) or any hub id")
    ap.add_argument("--split", default="test", help="HF split (default: test)")
    ap.add_argument("--n", type=int, default=None, help="cap number of HF samples (quick check)")
    ap.add_argument("--config", default=None, help="HF dataset config/subset name")
    ap.add_argument("--mode", choices=["auto", "recognizer", "pipeline"], default="auto",
                    help="auto: recognizer for line datasets, full pipeline otherwise")
    ap.add_argument("--baselines", default="", help="comma list: tesseract,easyocr,paddleocr")
    ap.add_argument("--csv", default="eval_results.csv")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--strip-punct", action="store_true")
    args = ap.parse_args()

    if args.hf:
        samples = load_hf(args.hf, split=args.split, n=args.n, config=args.config)
        line_level = is_line_level(args.hf)
        print(f"loaded {len(samples)} samples from HF {args.hf} [{args.split}]")
    else:
        samples = load_pairs(args.data)
        line_level = False
        print(f"loaded {len(samples)} samples from {args.data}")
    if not samples:
        print("No (image, ground-truth) pairs found. See module docstring for layout.")
        return 1

    norm = NormCfg(lowercase=args.lowercase, strip_punct=args.strip_punct)
    use_recognizer = args.mode == "recognizer" or (args.mode == "auto" and line_level)

    scores: list[Score] = []
    if use_recognizer:
        print("scoring recognizer only (no detection/ordering)")
        scores.append(evaluate_recognizer("trocr-recognizer", samples, norm=norm))
    else:
        pt, pl = pipeline_predictors()
        scores.append(evaluate_system("this-pipeline", samples, pt, pl, norm))

    for b in [x.strip() for x in args.baselines.split(",") if x.strip()]:
        try:
            scores.append(evaluate_system(b, samples, baseline_predict(b), norm=norm))
        except Exception as e:  # missing dep / runtime error in a baseline
            print(f"  (skip baseline {b}: {type(e).__name__}: {e})")

    print("\n" + format_table(scores))
    save_csv(scores, args.csv)
    print(f"\nper-image breakdown -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
