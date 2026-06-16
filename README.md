# HI-RES — Architecture-Driven Handwriting OCR

Full-page **English handwriting** OCR built by composing task-specific models with a
deterministic **reading-order reconstruction** stage — turning a single-line recognizer
into a system that transcribes whole pages in the correct order.

![Pipeline](assets/pipeline.svg)

## Why this exists

Conventional OCR leaves a gap for full-page handwriting:

- **Generic OCR engines** (Tesseract, PaddleOCR) handle printed text well but are weak on handwriting.
- **Handwriting models** (Microsoft TrOCR) are strong on handwriting but only transcribe a **single cropped line** — not a page.

HI-RES bridges that gap: **PaddleOCR PP-OCRv5 detection → a custom reading-order reconstruction module → TrOCR-large line recognition → reassembled text.** The reading-order module is the engineered core — it recovers the true reading sequence from detected box geometry, so the output isn't a bag of randomly-ordered lines.

## How it works

1. **Detection** — PP-OCRv5 finds text regions and returns quadrilaterals.
2. **Reading-order reconstruction** (`pipeline.py`, pure NumPy/OpenCV, fully unit-tested):
   - page-skew estimation from box edge angles,
   - line clustering by vertical-band overlap,
   - left-to-right ordering within each line,
   - conservative column splitting for multi-column pages.
   Order is computed **only from coordinates**, so it is provably independent of the detector's output order.
3. **Recognition** — each line crop (perspective-rectified, padded) is read by TrOCR-large; long lines are chunked so they fit the model's input aspect ratio.
4. **Reassembly** — line texts are joined in reading order, with paragraph breaks inferred from vertical gaps.

## Results

Benchmarked on **GNHK** (172 real-world "handwriting-in-the-wild" photos), every system scored on the **same images** with one metric (corpus-level CER/WER, lower is better):

| System | CER |
|---|---|
| **HI-RES (this pipeline)** | **29.5%** |
| PP-OCRv5 server | 28.3% |
| PP-OCRv5 mobile | 34.5% |
| EasyOCR | 55.4% |
| Tesseract | 76.1% |

→ **HI-RES matches PP-OCRv5 server** (the strongest baseline) while composing a stock single-line recognizer (TrOCR-large) with a deterministic reading-order stage — no end-to-end re-training. All baselines run their built-in pipeline (detection + recognition); HI-RES uses PP-OCRv5 detection paired with TrOCR recognition, which is the contribution of the reading-order module.

> Note: GNHK is deliberately hard (camera-captured, unconstrained handwriting), and the recognizer is **stock** TrOCR-large (trained on IAM, not fine-tuned on GNHK) — so absolute CER has clear headroom. The result is strong *relative to deployable baselines*; fine-tuning the recognizer is the main accuracy lever (see Roadmap).

## Inference speed

HI-RES and PP-OCRv5 server reach nearly the same accuracy (29.5% vs 28.3% CER), so the practical question becomes throughput. `benchmark_speed.py` times both on the **same images**, splitting HI-RES into its three stages so the bottleneck is explicit:

```bash
python benchmark_speed.py --images gnhk/test --n 30          # HI-RES vs PP-OCRv5 server
python benchmark_speed.py --images pages --n 20 --no-baseline # HI-RES only
```

Both systems share the PP-OCRv5 detector, so the timing gap is the **recognizer**: TrOCR-large is an autoregressive decoder (heavier, handwriting-grade), while PP-OCRv5 server rec is a CTC head (lighter). The **reading-order stage is pure geometry — sub-millisecond per page** — so it is never the bottleneck; any HI-RES slowdown buys the stronger handwriting recognizer. Run on a GPU runtime for a fair comparison (TrOCR is GPU-bound). The script prints per-stage means, throughput (img/s), and the HI-RES↔PP-OCRv5 slowdown factor.

One-click, self-contained notebook (embeds the source, synthesizes test pages, no setup): **[`speed_benchmark.ipynb`](speed_benchmark.ipynb)** — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vedavarshith868/HI-RES-Architecture-Driven-OCR/blob/main/speed_benchmark.ipynb)

## Multilingual extension

A separate pipeline — [`multilingual/`](multilingual/README.md) — applies the same reading-order stage to **multilingual document OCR**, where PaddleOCR PP-OCRv5 handles *both* detection and recognition:

```
PP-OCRv5 server detection → reading-order reconstruction → PP-OCRv5 recognition (per language)
```

It recognizes **every** detected box (the stock pipeline silently drops low-confidence ones) and emits them in geometric reading order, then is scored against stock `PaddleOCR(lang=…)` on **XFUND** (Latin + CJK forms) for CER/WER and speed in one run. Left-to-right scripts (Latin/CJK/Indic); see [`multilingual/README.md`](multilingual/README.md) for details.

One-click, self-contained notebook (embeds the source, auto-downloads XFUND): **[`multilingual_ocr.ipynb`](multilingual_ocr.ipynb)** — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vedavarshith868/HI-RES-Architecture-Driven-OCR/blob/main/multilingual_ocr.ipynb)

## Quickstart

```bash
pip install -r requirements.txt
```

The TrOCR weights (~2 GB) download automatically from the Hugging Face Hub on first run — nothing to commit or place manually.

```bash
# Gradio web UI
python app.py

# One image from the CLI (writes <name>_ocr.txt and a side-by-side panel image)
python app.py --image page.jpg
python app.py --image page.jpg --beams 4      # slightly higher accuracy, slower
```

A GPU is recommended for the recognizer; it falls back to CPU.

### Try it on Colab

Open `colab_ocr_debug.ipynb` in Google Colab (T4 GPU). It is self-contained: installs deps, runs the geometry tests, lets you upload your own photos, shows a numbered boxes-plus-transcript view, and includes the full evaluation harness.

## Evaluation

`evaluate.py` is a from-scratch harness scoring any system on the same data:

```bash
# your own labelled folder (image + sibling .txt ground truth)
python evaluate.py --data eval_data --baselines tesseract,paddleocr

# a public dataset from the Hugging Face Hub (IAM test lines)
python evaluate.py --hf iam-lines --n 200
```

Metrics: **CER / WER** (the headline) plus **WordAcc** — an order-free word accuracy that separates *recognition* errors from *reading-order* errors (high WordAcc + high CER ⇒ ordering issue; low WordAcc ⇒ recognition issue).

## Project structure

```
pipeline.py            # reading-order geometry (deskew, clustering, cropping) — NumPy/OpenCV only
ocr_engine.py          # Detector (PaddleOCR) + Recognizer (TrOCR) + OcrEngine.run()
app.py                 # Gradio UI + CLI
evaluate.py            # CER/WER + WordAcc harness, dataset loaders, baseline runners
benchmark_speed.py     # inference speed: HI-RES vs PP-OCRv5 server, per-stage timing
multilingual/          # separate PP-OCR det+rec pipeline with HI-RES ordering (Latin+CJK)
make_colab_notebook.py # generates colab_ocr_debug.ipynb
colab_ocr_debug.ipynb  # self-contained Colab notebook (handwriting pipeline + eval)
make_speed_notebook.py # generates speed_benchmark.ipynb
speed_benchmark.ipynb  # self-contained Colab: HI-RES vs PP-OCRv5 speed
make_multilingual_notebook.py # generates multilingual_ocr.ipynb
multilingual_ocr.ipynb # self-contained Colab: multilingual pipeline vs stock PaddleOCR
tests/                 # geometry + metric unit tests (no model needed) + end-to-end smoke tests
docs/ANALYSIS.md       # engineering audit of the original prototype and what was fixed/verified
```

## Limitations & roadmap

1. **Fine-tune the recognizer** on handwriting (GNHK/IAM/synthetic) — the biggest accuracy lever; converts a stock recognizer into a domain-tuned one.
2. **Document orientation** — pages rotated 90°/180° need an upfront orientation classifier.
3. **Mobile/web deployment** — TrOCR-large is the heavy component; a TrOCR-base or quantized/ONNX export is the path to on-device latency.
4. Tables, dense multi-column layouts, and non-Latin scripts are out of current scope.

## Acknowledgements

- [Microsoft TrOCR](https://huggingface.co/microsoft/trocr-large-handwritten) (recognition)
- [PaddleOCR PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR) (detection)
- [GNHK dataset](https://github.com/GoodNotes/GNHK-dataset) (evaluation, CC-BY-4.0)
