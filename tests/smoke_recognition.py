"""Smoke test: load TrOCR from the repaired local dir and read one synthetic
handwriting-style line. Verifies (1) local dir loads after the rename,
(2) decoder_start_token_id resolves to 2, (3) generation produces sane text,
(4) KV-cache speedup of use_cache=True vs the shipped use_cache=False.

Run:  python tests/smoke_recognition.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ocr_engine import Recognizer, resolve_rec_source  # noqa: E402

SENTENCE = "The quick brown fox jumps over the lazy dog"


def render_line(text: str, font_path: str = r"C:\Windows\Fonts\Inkfree.ttf",
                size: int = 56) -> np.ndarray:
    font = ImageFont.truetype(font_path, size)
    pad = 24
    bbox = ImageDraw.Draw(Image.new("RGB", (8, 8))).textbbox((0, 0), text, font=font)
    img = Image.new("RGB", (bbox[2] - bbox[0] + 2 * pad, bbox[3] - bbox[1] + 2 * pad),
                    "white")
    ImageDraw.Draw(img).text((pad - bbox[0], pad - bbox[1]), text, font=font,
                             fill=(20, 20, 30))
    return np.array(img)


def main() -> int:
    source = resolve_rec_source()
    print(f"model source: {source}")
    if "local_trocr_model" not in source:
        print("FAIL: local model dir was not picked up")
        return 1

    t0 = time.perf_counter()
    rec = Recognizer(source)
    print(f"loaded in {time.perf_counter() - t0:.1f}s on {rec.device}")

    start_id = rec.model.generation_config.decoder_start_token_id
    print(f"decoder_start_token_id = {start_id}")
    if start_id != 2:
        print("FAIL: decoder_start_token_id should be 2 (</s>) for TrOCR")
        return 1

    line = render_line(SENTENCE)
    t0 = time.perf_counter()
    out = rec([line])[0]
    t_cached = time.perf_counter() - t0
    print(f"with KV cache:    {t_cached:5.1f}s -> {out!r}")

    # same generation but honoring the shipped use_cache=False, for comparison
    pixel_values = rec.processor(images=[line], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(rec.device, dtype=rec.model.dtype)
    t0 = time.perf_counter()
    with rec.torch.inference_mode():
        ids = rec.model.generate(pixel_values, max_new_tokens=96, use_cache=False)
    out_nc = rec.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    t_nocache = time.perf_counter() - t0
    print(f"without KV cache: {t_nocache:5.1f}s -> {out_nc!r}")

    ref = "".join(c for c in SENTENCE.lower() if c.isalnum())
    hyp = "".join(c for c in out.lower() if c.isalnum())
    matches = sum(a == b for a, b in zip(ref, hyp))
    rate = matches / max(len(ref), 1)
    print(f"aligned char match vs ground truth: {rate:.0%}")
    if rate < 0.7:
        print("FAIL: recognition far from ground truth")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
