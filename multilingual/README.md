# Multilingual OCR — PP-OCR detection + recognition with HI-RES reading order

A second, **independent** pipeline (no link to the handwriting work) that brings the
HI-RES reading-order stage to **multilingual printed/document OCR**, where both
detection and recognition are done by PaddleOCR PP-OCRv5 models.

```
PP-OCRv5 server detection  ─►  reading-order reconstruction (pipeline.py)  ─►  PP-OCRv5 recognition (per language)
```

## Why insert HI-RES between PaddleOCR's own detector and recognizer?

PaddleOCR already ships an end-to-end multilingual pipeline, so the gain has to be
concrete. Two mechanisms:

1. **No dropped words.** The built-in pipeline discards detected boxes whose
   recognition confidence falls below an internal threshold — those words simply
   vanish from the output. We run a strong detector (PP-OCRv5 **server** det) and
   recognize **every** detected box, so faint/odd words still appear.
2. **Correct reading order.** Boxes are emitted in a geometry-derived order
   (deskew → line clustering → left-to-right within a line → conservative column
   split), computed purely from coordinates, instead of the detector's raw output
   order. On skewed phone photos and multi-column forms this is the difference
   between a readable transcript and a shuffled bag of lines.

The recognizer is unchanged PaddleOCR, so this is **not** a recognition-accuracy
trick — it is a *detection-recall + ordering* layer around PP-OCR.

## Scope

Left-to-right scripts only: **Latin** (en/fr/de/es/it/pt…), **CJK** (zh/ja/ko),
**Indic** (Devanagari/Tamil/Telugu…). Right-to-left (Arabic/Hebrew) is out of scope
— the ordering sorts left-to-right within each line.

CJK is scored **space-free** (the standard character-level CER for scripts with no
inter-word spaces); WER is not meaningful for CJK and should be ignored.

## Files

```
multilingual/ml_engine.py     # MultilingualOcrEngine: PP-OCR det → reading order → PP-OCR rec
multilingual/ml_evaluate.py   # XFUND loader + HI-RES vs stock PaddleOCR (CER/WER/WordAcc/speed)
```

Both reuse `../pipeline.py` (the reading-order geometry), `../detector.py`'s
`Detector` (the paddle-only detection wrapper — **no TrOCR/handwriting dependency**),
and `../evaluate.py`'s metrics — so the numbers are defined identically to the
English benchmark.

## Use

```python
from multilingual.ml_engine import MultilingualOcrEngine
import cv2, numpy as np

eng = MultilingualOcrEngine(lang="ch")          # or "ja", "fr", "de", ...
img = cv2.cvtColor(cv2.imread("form.jpg"), cv2.COLOR_BGR2RGB)
out = eng.run(img)
print(out["text"])                               # transcript in reading order
# out["overlay"], out["composite"] are debug visualizations; out["seconds"] is timing
```

## Evaluate (XFUND)

XFUND (CC-BY-4.0, from [microsoft/unilm](https://github.com/doc-analysis/XFUND))
is multilingual forms with per-segment text + boxes. Download one language's
`{lang}.{split}.json` plus its images into a folder, then:

```bash
python multilingual/ml_evaluate.py --xfund xfund_zh --lang ch
python multilingual/ml_evaluate.py --xfund xfund_fr --lang fr
python multilingual/ml_evaluate.py --xfund xfund_ja --lang ja --gt-order geom
```

This scores **hires-ml** against **stock PaddleOCR(lang=…)** on the same pages and
prints CER / WER / WordAcc / **sec-per-img**, so accuracy *and* speed come out of
one run. `--gt-order list` (default) uses the dataset's annotation order as ground
truth; `--gt-order geom` uses a geometric reading order.

## Run it on Colab (GPU)

**Easiest:** open the one-click, self-contained notebook
[`multilingual_ocr.ipynb`](../multilingual_ocr.ipynb) — it embeds the source and
auto-downloads XFUND, no clone or manual data steps.

Or assemble it yourself from the repo:

```python
# 1. deps
!pip -q install paddleocr paddlepaddle datasets opencv-python-headless

# 2. code
!git clone -q https://github.com/Vedavarshith868/HI-RES-Architecture-Driven-OCR
%cd HI-RES-Architecture-Driven-OCR

# 3. XFUND (Chinese val) — text json + images
!mkdir -p xfund_zh && cd xfund_zh && \
  wget -q https://github.com/doc-analysis/XFUND/releases/download/v1.0/zh.val.json && \
  wget -q https://github.com/doc-analysis/XFUND/releases/download/v1.0/zh.val.zip && \
  unzip -q -o zh.val.zip

# 4. evaluate: HI-RES multilingual vs stock PaddleOCR, accuracy + speed in one table
!python multilingual/ml_evaluate.py --xfund xfund_zh --lang ch --n 40
```

> The exact PP-OCRv5 language rec-model names are resolved at load time;
> `ml_engine.PaddleRecognizer` falls back to `PP-OCRv5_server_rec` if a specific
> one is unavailable in the installed PaddleOCR build. Pass `--rec-model NAME` to
> force a particular model.
