# Results

Raw per-page data backing every number below lives in [`results/`](results/) — nothing
here is aggregated without the source CSV to check it against.

## Handwriting (English) — TrOCR-large + reading order vs PaddleOCR v5-Server

Real, out-of-domain photographed handwriting from **GNHK**, 172 pages, corpus-level
CER/WER. Source: [`results/handwriting_gnhk.csv`](results/handwriting_gnhk.csv).

| system | CER | WER | sec/img |
|---|---|---|---|
| **HI-RES** (TrOCR-large + reading order) | **29.3%** | **53.0%** | 1.16 |
| PaddleOCR v5-Server | 29.5% | 67.5% | 0.97 |

HI-RES matches PaddleOCR's server-tier model on character accuracy and is meaningfully
better on word accuracy (WER), at a slower per-image cost — TrOCR's autoregressive
decoder is inherently heavier than a CTC head. See the pitch on recognizer choice for why
that trade was made deliberately.

## Multilingual printed text — PP-OCRv6, HI-RES vs stock PaddleOCR

XFUND validation pages, 5 languages, 40 pages each (200 total), same PP-OCRv6 detection
and recognition models on both sides — only detection-confidence handling and reading
order differ. Source: `results/multilingual_{ch,es,fr,german,ja}.csv` (raw per-page).

| language | HI-RES CER | stock CER | Δ (pp) | HI-RES img/s | stock img/s |
|---|---|---|---|---|---|
| ch | 38.23% | 39.58% | +1.35 | 0.596 | 0.555 |
| es | 13.09% | 14.29% | +1.20 | 0.372 | 0.494 |
| fr | 27.12% | 33.56% | +6.45 | 0.444 | 0.525 |
| german | 10.37% | 11.90% | +1.52 | 0.458 | 0.530 |
| ja | 30.42% | 31.24% | +0.82 | 0.456 | 0.531 |
| **pooled (n=200)** | **20.23%** | **22.83%** | **+2.60** | **0.455** | **0.526** |

HI-RES is consistently more accurate (recognizes every detected box; stock PaddleOCR
silently drops low-confidence detections, which can look faster but loses text). In this
controlled, same-model comparison HI-RES runs at ~0.86x stock's throughput — the accuracy
gain here trades against speed, not for it.

## CPU throughput — PP-OCRv5-mobile, same models on both sides

Actual CPU runs (not GPU), 3 languages, 12 pages each, PP-OCRv5-mobile detection +
recognition on both HI-RES and stock PaddleOCR. Source:
[`results/multilingual_cpu_v5mobile.csv`](results/multilingual_cpu_v5mobile.csv).

| language | HI-RES sec/img | stock sec/img | speedup | HI-RES CER | stock CER |
|---|---|---|---|---|---|
| ja | 16.20 | 39.95 | 2.47x | 32.8% | 37.9% |
| ch | 12.12 | 37.96 | 3.13x | 35.4% | 36.6% |
| es | 16.81 | 41.92 | 2.49x | 16.6% | 16.0% |
| **average** | | | **~2.7x** | | |

On CPU, with identical detection/recognition models on both sides, HI-RES runs
**2.5–3.1x faster per page** (avg ~2.7x) — this is the batching-and-orchestration effect
described in the project write-up. CER is better for ja/ch and roughly tied for es on
this small sample (n=12/language); it isn't the accuracy claim — the multilingual PP-OCRv6
comparison above (n=200) is.

## Skew robustness — the PaddleOCR PR evidence

The proof behind [PaddlePaddle/PaddleOCR#18189](https://github.com/PaddlePaddle/PaddleOCR/pull/18189):
detection + recognition held identical, only the box-ordering function swapped. Full
XFUND validation split, 7 languages, ~50 pages/language, skew 0/3/6/10°.
Source: [`results/paddleocr_pr_skew_proof.csv`](results/paddleocr_pr_skew_proof.csv),
figure: [`results/paddleocr_pr_skew_proof.png`](results/paddleocr_pr_skew_proof.png).

| language | CER @0° (stock → fixed) | CER @10° (stock → fixed) |
|---|---|---|
| zh | 43.6% → 40.5% | 61.0% → 41.5% |
| ja | 32.8% → 31.5% | 44.7% → 31.8% |
| de | 11.6% → 9.8% | 27.9% → 10.5% |
| es | 14.4% → 12.1% | 32.2% → 12.6% |
| fr | 32.8% → 30.9% | 40.8% → 30.8% |
| it | 23.8% → 22.7% | 34.2% → 22.9% |
| pt | 31.5% → 30.4% | 40.1% → 30.4% |

At 0° the two orderings are within 1–3 CER (backward compatible). By 10° tilt, stock's
fixed 10px row threshold degrades by +8 to +18 CER in every language; the fix stays flat.

## Full HI-RES pipeline vs stock under skew

Same skew sweep, but comparing the *complete* HI-RES pipeline (including column
splitting) against stock PaddleOCR end-to-end, rather than isolating just the ordering
function. Source: [`results/hires_vs_stock_skew.csv`](results/hires_vs_stock_skew.csv).

| language | HI-RES CER @0°→10° | stock CER @0°→10° |
|---|---|---|
| zh | 38.2% → 39.2% | 39.6% → 58.2% |
| ja | 30.4% → 30.2% | 31.2% → 43.1% |
| de | 10.4% → 11.0% | 11.9% → 28.3% |
| es | 13.1% → 13.3% | 14.3% → 33.1% |
| fr | 27.1% → 29.3% | 33.6% → 40.4% |

HI-RES stays essentially flat across the whole skew range; stock's CER climbs sharply as
tilt increases.
