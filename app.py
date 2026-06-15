"""Handwritten-text OCR app: PaddleOCR detection + reading order + TrOCR.

Run as a Gradio app:      python app.py
Run once on an image:     python app.py --image page.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # HF Spaces ZeroGPU decorator; harmless no-op elsewhere
    from spaces import GPU
except ImportError:
    def GPU(fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn

from ocr_engine import OcrEngine

engine = OcrEngine()


@GPU
def recognize(image, merge_segments: bool = True, num_beams: int = 1):
    if image is None:
        return "Please upload an image.", None
    result = engine.run(image, merge_segments=merge_segments, num_beams=int(num_beams))
    text = result["text"]
    if note := result.get("note"):
        text = f"[{note}]\n{text}"
    return text, result["composite"]


def build_interface():
    import gradio as gr  # lazy so the CLI works without gradio installed

    return gr.Interface(
        fn=recognize,
        inputs=[
            gr.Image(type="numpy", label="Handwritten page"),
            gr.Checkbox(value=True, label="Merge line segments before recognition"),
            gr.Slider(1, 5, value=1, step=1, label="Beam width (slower, slightly more accurate)"),
        ],
        outputs=[
            gr.Textbox(label="Recognized text", lines=12, show_copy_button=True),
            gr.Image(label="Page + transcript (boxes numbered, text beside)"),
        ],
        title="Handwritten Text Recognition",
        description="PaddleOCR PP-OCRv5 detection + deterministic reading-order "
                    "reconstruction + TrOCR-large recognition.",
        flagging_mode="never",
    )


def run_cli(image_path: str, merge: bool, beams: int) -> None:
    import cv2
    import numpy as np

    path = Path(image_path)
    img_bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"could not read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    result = engine.run(img_rgb, merge_segments=merge, num_beams=beams)

    print(result["text"])
    secs = result["seconds"]
    print(f"\n--- skew {result['skew_deg']:.1f} deg | detect {secs['detect']:.2f}s "
          f"| recognize {secs['recognize']:.2f}s ---")

    out_txt = path.with_name(path.stem + "_ocr.txt")
    out_panel = path.with_name(path.stem + "_panel.png")
    out_txt.write_text(result["text"], encoding="utf-8")
    ok, buf = cv2.imencode(".png", cv2.cvtColor(result["composite"], cv2.COLOR_RGB2BGR))
    if ok:
        buf.tofile(out_panel)
    print(f"saved: {out_txt.name}, {out_panel.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="run once on this image instead of launching the UI")
    parser.add_argument("--no-merge", action="store_true",
                        help="recognize each detected box separately")
    parser.add_argument("--beams", type=int, default=1, help="beam width (default greedy)")
    args = parser.parse_args()

    if args.image:
        run_cli(args.image, merge=not args.no_merge, beams=args.beams)
    else:
        build_interface().launch()
