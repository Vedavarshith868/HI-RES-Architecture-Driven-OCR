# Rigorous analysis of the original OCR pipeline

Date: 2026-06-13. Everything below was verified empirically on this machine —
nothing was taken on trust. The original `app.py` is preserved unchanged as
`legacy_app.py`; line numbers refer to it.

## Verdict on the architecture

The architecture itself — **detect text boxes → rectify-crop each box →
recognize per box → reassemble in reading order** — is sound. It is the same
design used by PP-OCR, docTR, Surya, MMOCR and essentially every production
OCR system. It was not the reason results were unreliable.

What was actually broken: **the reading-order step was never implemented.**
The whole "spatial reordering" was one line, `cropped_images.reverse()`
(legacy_app.py:61). The stored coordinates were never used for ordering, no
line grouping existed, and no left-to-right ordering within a line existed.

Why `reverse()` ever looked right: PaddleOCR's DB postprocess collects boxes
from `cv2.findContours`, which tends to return contours bottom-to-top.
Reversing therefore *often* yields top-to-bottom on simple one-column images —
but that ordering is an implementation detail with no contract, and it falls
apart the moment a physical line is split into several boxes (routine for
handwriting), the page is skewed, or there are columns. The observed
"random order" was exactly this.

## Findings

Severity: CRITICAL = wrong output, HIGH = broken artifact/silent quality loss,
MED = correctness-or-performance defect, LOW = robustness gap.

| # | Where | Severity | Finding | Status |
|---|-------|----------|---------|--------|
| 1 | legacy_app.py:61 | CRITICAL | Reading order = `list.reverse()` on detector output order (undefined). No coordinate use, no line clustering, no x-sort. | Rewritten: deskew → column split → line clustering → x-sort (`pipeline.reading_order`), covered by unit tests incl. a case where naive y-sort provably fails. |
| 2 | legacy_app.py:78 | CRITICAL | Output joins every box with `\n`: word fragments of one handwritten line appear as separate, arbitrarily ordered lines. | Boxes are clustered into lines; fragments joined with spaces; paragraph gaps emit blank lines. |
| 3 | local_trocr_model/ | HIGH | Weights file named `model-001.safetensors` — a name `from_pretrained` never looks for. The local dir could not load at all, so the app silently re-downloaded 2.2 GB from the Hub on every cold start. | Renamed to `model.safetensors`; verified loadable (smoke test). Header parse: 636 tensors, 558.2M params fp32 — a complete TrOCR-large checkpoint. |
| 4 | local_trocr_model/generation_config.json | HIGH | Missing `decoder_start_token_id: 2` (present in both your Hub copy and Microsoft's original). A local load would silently fall back to `<s>`(0) as the start token and degrade generation. | Added; smoke test confirms it resolves to 2. |
| 5 | legacy_app.py:32 | MED | Gradio delivers **RGB**; PaddleOCR's `predict()` expects cv2-style **BGR**. Channel order silently wrong for detection. | `cv2.cvtColor(..., COLOR_RGB2BGR)` before detection. |
| 6 | config/generation_config | MED | `use_cache: false` shipped in Microsoft's checkpoint (training leftover): generation re-runs the whole decoder per token. | `use_cache=True` passed at generate time. (Measured on CPU greedy short lines: small gain — the ViT encoder dominates; matters more with beams/long lines.) |
| 7 | legacy_app.py:65–70 | MED | Crops recognized one at a time — no batching. | Batched recognition (default 8/batch); fp16 on CUDA. |
| 8 | legacy_app.py:44–58 | MED | 4-point perspective crop math itself is **correct** (verified against the standard `get_rotate_crop_image` and by unit test), but: no degenerate-size guard (`width`/`height` can be 0 → crash) and the reference's vertical-text `rot90` step was dropped. | Guards + opt-in `rot90` for strongly vertical boxes; crop now pads ~5% so ascenders/descenders aren't clipped (TrOCR was trained with margins). |
| 9 | legacy_app.py:36 | LOW | `dt_scores` ignored (no confidence filter) and `polys.tolist()` assumes ndarray — crashes if a paddleocr version returns a list. | Score filter (≥0.30) + `np.asarray` normalization + >4-point polygon reduction to min-area quads. |
| 10 | recognition design | MED | Per-box recognition starves TrOCR's language-model decoder of context: it reads single words/fragments. | Same-line fragments are merged into one line crop (min-area rect) before recognition, with a height guard (≤1.8× median member height) so a bad cluster can't swallow two stacked lines. |
| 11 | recognition design | MED | TrOCR squeezes every crop to a 384×384 square. A full-width handwritten line (aspect ≳16:1) becomes unreadably squeezed — an inherent TrOCR constraint the old code never handled. | Long lines are chunked to ≤16:1 aspect per recognition call, order preserved (`pipeline.chunk_line`, tested). |
| 12 | legacy_app.py:68 | LOW | `max_new_tokens=64` can truncate long merged lines. | 96 + aspect chunking bounds the text per crop. |
| 13 | requirements.txt | MED | Fully unpinned. `from paddleocr import TextDetection` only exists in paddleocr 3.x — an environment that resolves 2.x crashes at import. | Floors pinned; verified against paddleocr 3.7.0 / paddlepaddle 3.3.1 / transformers 5.12.0 / torch 2.7.1. |
| 14 | legacy_app.py:8 | LOW | `from spaces import GPU` crashes anywhere the HF-Spaces package is missing. | Guarded import with no-op fallback decorator. |
| 15 | legacy_app.py:72–76 | LOW | Whole-page fallback feeds a multi-line page to a single-line model. | Kept as explicit fallback, now flagged in a structured `note` field. |
| 16 | environment (found during verification) | HIGH (env) | paddlepaddle 3.3.1 + paddleocr 3.7.0 on Windows CPU crashes in the PIR/oneDNN executor (`ConvertPirAttribute2RuntimeAttribute not support`). Not a bug in this repo, but it makes detection unusable here by default. | MKL-DNN auto-disabled on Windows (env override `OCR_DET_MKLDNN=1`), plus a one-shot self-heal retry if the crash signature appears elsewhere. Cost: CPU detection is slow on Windows (~78 s for a 1554×990 page with the server det model). Linux/Spaces keeps oneDNN. |

## Identity of the model (verified, not assumed)

`imperiusrex/Handwritten_model` is a re-save of `microsoft/trocr-large-handwritten`
(identical architecture and dims; 558.2M params, fp32, output projection tied
to input embeddings — the tensor count and parameter total match exactly; no
fine-tuning info in the repo). Practical consequences:

- Recognition quality is stock TrOCR-large-handwritten: trained on IAM
  English handwriting **lines**. Strong on cursive/print English sentences;
  known weaknesses: digits, punctuation-dense text, ALL-CAPS, and domain
  shift to messy phone photos.
- There is no benefit in downloading from your Hub repo vs. the (now fixed)
  local copy — the new code prefers the local dir and falls back to the Hub.

## What was verified, and how

1. **18 geometry unit tests** (`tests/test_geometry.py`) — all pass. Includes:
   a skewed-page fixture which *first proves* naive center-y sorting produces
   the wrong order, then asserts the deskewed clustering recovers the exact
   order; two-column ordering; a wide in-line gap that must NOT trigger a
   column split; descender height outliers staying on their line; crop
   rectification of rotated text (pixel-level assertions); degenerate boxes
   returning None instead of crashing; chunking order preservation.
2. **Recognition smoke test** (`tests/smoke_recognition.py`) — local model
   loads after the repair, start token = 2, and a rendered handwriting-style
   sentence is read back **100% correct**.
3. **End-to-end smoke test** (`tests/smoke_e2e.py`) — a 6-line, 2-paragraph
   handwriting-font page rotated 3.5°: skew estimated −3.18°, all six lines
   recognized character-perfect, in order, with the paragraph break placed
   correctly. Debug overlay: `tests/e2e_overlay.png`.

## Honest calibration vs. the stated goal

"As good as ChatGPT/Gemini" needs splitting into two claims:

- **Reading order & layout**: now deterministic and testable. For typical
  single/two-column pages this component is solved to the same practical
  level VLMs reach, at ~zero compute cost. Remaining gap: pages rotated
  90/180° (needs an orientation classifier up front) and exotic layouts
  (tables, marginalia).
- **Recognition accuracy**: stock TrOCR-large-handwritten is competitive on
  IAM-style English handwriting (~2.9 CER on IAM test in its paper), but a
  frontier VLM is still better on hard real-world handwriting because it
  leverages page-level context. To close that gap cheaply, the highest-value
  moves in order: (1) build an eval harness (IAM/GNHK, CER/WER) so every
  change is measured, not vibed; (2) fine-tune TrOCR on broader handwriting
  (GNHK, CVL, synthetic); (3) try beam width 4 (now a UI slider);
  (4) add document/text-line orientation handling.

Without (1), no parity claim is meaningful — that's the next thing to build.
