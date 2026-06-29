"""Gradio demo: HI-RES OCR — two tabs, both loaded once at startup.

Tab 1 — Handwritten (English): PaddleOCR detection -> deterministic reading
order (deskew + column split + line clustering) -> TrOCR-large recognition.

Tab 2 — Printed / Multilingual: PP-OCRv6 detection -> the same reading-order
layer -> PP-OCRv6 recognition (~50 languages). Column-aligned blocks are
rendered as Markdown tables.

The two pipelines share the detector + reading-order layer and differ only in
the recognizer. They load once into memory and stay warm.
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import gradio as gr

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from printed.tabularize import layout_to_markdown, _md_text  # noqa: E402

# ---- load both engines once at startup -----------------------------------
HW = None
HW_ERR = None
ML = None
ML_ERR = None


def _load_engines():
    global HW, HW_ERR, ML, ML_ERR
    try:
        from handwritten.engine import OcrEngine
        print("Loading handwritten (TrOCR-large) engine ...", flush=True)
        HW = OcrEngine()  # auto device: GPU if available, else CPU
        print("  handwritten engine ready.", flush=True)
    except Exception:
        HW_ERR = traceback.format_exc()
        print("Handwritten engine FAILED:\n" + HW_ERR, flush=True)
    try:
        from printed.engine import MultilingualOcrEngine
        print("Loading multilingual (PP-OCRv6) engine ...", flush=True)
        ML = MultilingualOcrEngine(
            lang="en",
            det_model="PP-OCRv6_medium_det",
            rec_model="PP-OCRv6_medium_rec",
        )
        print("  multilingual engine ready.", flush=True)
    except Exception:
        ML_ERR = traceback.format_exc()
        print("Multilingual engine FAILED:\n" + ML_ERR, flush=True)


_load_engines()


# ---- inference callbacks -------------------------------------------------
def _ml_token_lines(res):
    """Convert engine tokens to (text, x0, x1) lines + median text height (px)."""
    out, heights = [], []
    for line in res.get("token_lines", []):
        toks = []
        for tk in line:
            box = np.asarray(tk["box"], dtype=float)
            xs, ys = box[:, 0], box[:, 1]
            heights.append(float(ys.max() - ys.min()))
            toks.append((tk["text"], float(xs.min()), float(xs.max())))
        out.append(toks)
    lh = float(np.median(heights)) if heights else None
    return out, lh


def run_handwritten(image):
    if image is None:
        return "Please upload an image.", None
    if HW is None:
        return f"Handwriting engine failed to load:\n\n```\n{HW_ERR}\n```", None
    try:
        res = HW.run(np.asarray(image), num_beams=1)
    except Exception:
        return f"Recognition error:\n\n```\n{traceback.format_exc()}\n```", None
    # TrOCR reads whole lines -> no per-token columns; show reading-order text.
    return _md_text(res.get("text", "")) or "_(no text found)_", res.get("overlay")


def run_multilingual(image, script, multicol):
    if image is None:
        return "Please upload an image.", None
    if ML is None:
        return f"Multilingual engine failed to load:\n\n```\n{ML_ERR}\n```", None
    ML.word_sep = "" if script.startswith("CJK") else " "
    try:
        res = ML.run(np.asarray(image), column_split=bool(multicol))
    except Exception:
        return f"Recognition error:\n\n```\n{traceback.format_exc()}\n```", None
    token_lines, line_height = _ml_token_lines(res)
    md = layout_to_markdown(token_lines, res.get("text", ""), line_height=line_height)
    return md or "_(no text found)_", res.get("overlay")


# ---- UI ------------------------------------------------------------------
with gr.Blocks(title="HI-RES OCR") as demo:
    gr.Markdown(
        "# HI-RES OCR\n"
        "Detection → deterministic reading-order reconstruction "
        "(deskew + column split + line clustering) → recognition.\n\n"
        "**Pick the tab that matches your input** — handwriting vs printed text "
        "(not language): printed English belongs in the multilingual tab."
    )

    with gr.Tab("Handwritten (English)"):
        gr.Markdown(
            "TrOCR-large + HI-RES pipeline — **English only**. Runs on free CPU, "
            "so a full page may take ~20–60s. Please be patient."
        )
        with gr.Row():
            hw_in = gr.Image(type="numpy", label="Upload handwriting",
                             sources=["upload", "clipboard"])
            hw_overlay = gr.Image(label="Detected reading order", interactive=False)
        gr.Markdown("**Recognized text**")
        hw_out = gr.Markdown()
        gr.Button("Run OCR", variant="primary").click(
            run_handwritten, inputs=hw_in, outputs=[hw_out, hw_overlay]
        )

    with gr.Tab("Printed / Multilingual"):
        gr.Markdown(
            "PP-OCRv6 — printed text in ~50 languages. Column-aligned content is "
            "rendered as a table. Set the script toggle so word-spacing matches."
        )
        with gr.Row():
            ml_in = gr.Image(type="numpy", label="Upload document",
                             sources=["upload", "clipboard"])
            ml_overlay = gr.Image(label="Detected reading order", interactive=False)
        ml_script = gr.Radio(
            ["Latin / Indic (use spaces)", "CJK (no spaces)"],
            value="Latin / Indic (use spaces)", label="Script",
        )
        ml_multicol = gr.Checkbox(
            value=False,
            label="Multi-column layout (newspaper / 2-column) — leave OFF for "
                  "forms, invoices and single-column pages",
        )
        gr.Markdown("**Recognized text / tables**")
        ml_out = gr.Markdown()
        gr.Button("Run OCR", variant="primary").click(
            run_multilingual,
            inputs=[ml_in, ml_script, ml_multicol],
            outputs=[ml_out, ml_overlay],
        )

if __name__ == "__main__":
    demo.queue().launch()
