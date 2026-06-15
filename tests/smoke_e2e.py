"""End-to-end smoke test: synthetic handwriting-style page -> PP-OCRv5
detection -> reading order -> TrOCR recognition.

The page has 6 lines in 2 paragraphs, rendered in a handwriting font and
rotated 3.5 degrees (a typical phone-photo skew). Asserts the recognized
ordinal words come out in reading order. Saves the debug overlay.

Run:  python tests/smoke_e2e.py
"""

import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LINES = [
    "My first line talks about the weather",
    "The second line is a bit longer and mentions tea",
    "Here the third line ends the opening paragraph",
    "A fourth line starts the next paragraph",
    "The fifth line keeps the story going",
    "Finally the sixth line wraps everything up",
]
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth"]


def render_page() -> np.ndarray:
    font = ImageFont.truetype(r"C:\Windows\Fonts\Inkfree.ttf", 46)
    img = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(img)
    y = 90
    for k, line in enumerate(LINES):
        if k == 3:
            y += 90  # paragraph break
        draw.text((100 + (k % 3) * 18, y), line, font=font, fill=(25, 25, 35))
        y += 86
    page = img.rotate(3.5, resample=Image.BICUBIC, expand=True,
                      fillcolor="white")  # photo-like skew
    return np.array(page)


def main() -> int:
    from ocr_engine import OcrEngine

    img = render_page()
    print(f"page: {img.shape[1]}x{img.shape[0]}")

    t0 = time.perf_counter()
    engine = OcrEngine()
    print(f"engine loaded in {time.perf_counter() - t0:.1f}s")

    result = engine.run(img)
    print(f"skew estimate: {result['skew_deg']:.2f} deg (expected ~-3.5)")
    print(f"timing: {result['seconds']}")
    print("---- recognized ----")
    print(result["text"])
    print("--------------------")

    out = Path(__file__).with_name("e2e_overlay.png")
    cv2.imwrite(str(out), cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR))
    print(f"overlay saved: {out}")

    text = result["text"].lower()
    positions = []
    for word in ORDINALS:
        m = re.search(word, text)
        if not m:
            print(f"FAIL: ordinal {word!r} missing from output")
            return 1
        positions.append(m.start())
    if positions != sorted(positions):
        print(f"FAIL: ordinals out of reading order: {positions}")
        return 1

    n_lines = len(result["lines"])
    if not 5 <= n_lines <= 8:
        print(f"FAIL: expected ~6 lines, got {n_lines}")
        return 1

    if "\n\n" not in result["text"]:
        print("WARN: paragraph gap not detected (cosmetic)")

    print(f"E2E SMOKE TEST PASSED ({n_lines} lines, ordinals in order)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
