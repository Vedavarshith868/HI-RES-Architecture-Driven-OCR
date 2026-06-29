# HI-RES — Architecture-Driven OCR

A **training-free** OCR system that improves accuracy by *re-engineering the
pipeline* instead of retraining models. A deterministic **reading-order
reconstruction** layer (deskew → column split → line clustering → left-to-right)
is inserted between an off-the-shelf detector and recognizer, so the transcript
reads in the correct order even on skewed, multi-column, or tabular pages.

**[▶ Live demo](https://huggingface.co/spaces/imperiusrex/hi-res-ocr)**  ·  **[Open-source contribution → PaddleOCR PR #18189](https://github.com/PaddlePaddle/PaddleOCR/pull/18189)**

![architecture](assets/pipeline.svg)

## Two pipelines, one reading-order core

| | Recognizer | Best for |
|---|---|---|
| **`handwritten/`** | TrOCR-large | English handwriting |
| **`printed/`** | PP-OCRv6 (~50 languages) | printed documents, forms, invoices (with table reconstruction) |

Both share `core/` (text detection + the reading-order geometry) and differ only
in the recognizer.

## Results

- **Handwriting** — matches PaddleOCR v5-Server; **2× lower CER than EasyOCR**,
  **15% fewer errors than v5-Mobile** (GNHK benchmark).
- **Multilingual printed** — **2.3 pp lower CER** and **2.9× CPU throughput** vs
  stock PaddleOCR via batched recognition (5 languages, 800 pages).
- **Skew robustness** — stock CER nearly doubles under real-world page rotation
  while HI-RES stays flat (**~19× more robust**, 800 pages / 5 languages).
- **Upstreamed** the reading-order fix to PaddleOCR (★84k): the stock
  `sorted_boxes` routine mis-orders text under mild skew — **+10–20% CER across
  7 XFUND languages** — fixed in [PR #18189](https://github.com/PaddlePaddle/PaddleOCR/pull/18189)
  (see [`paddleocr_pr/`](paddleocr_pr/)).

## Repository layout

```
core/            shared engine — pipeline.py (reading-order geometry) + detector.py
handwritten/     English handwriting pipeline (TrOCR)
printed/         multilingual printed pipeline (PP-OCRv6) + table reconstruction
paddleocr_pr/    the upstream fix contributed to PaddleOCR (PR #18189)
evaluate.py      CER / WER evaluation harness
app.py           Gradio demo (both pipelines, two tabs)
tests/           pure-geometry + metrics unit tests (no model downloads)
```

## Quickstart

```bash
pip install -r requirements.txt
python app.py                 # launch the two-tab demo locally
python -m pytest tests/       # run the unit tests
```

The first run downloads the detection/recognition models automatically.

## How reading order is reconstructed

1. **Estimate skew** from the median angle of detected box edges.
2. **Deskew** the box coordinates (pixels are never rotated).
3. **Split columns** conservatively (a cut needs a full-height whitespace gap).
4. **Cluster lines** by vertical-band overlap; order lines top→bottom, boxes
   left→right.
5. Recognize every detected box (PaddleOCR drops low-confidence boxes; HI-RES
   keeps them), then assemble in reading order. Column-aligned blocks are
   rendered as tables from the box geometry alone — no table model.
