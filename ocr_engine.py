"""Detection + recognition engine: PaddleOCR text detection -> reading order
(pipeline.py) -> TrOCR recognition.

Heavy dependencies (torch/transformers/paddleocr) are imported lazily inside
the components, so this module can be imported cheaply and the recognizer can
be used without paddle installed (and vice versa).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

import pipeline
# Detector lives in its own paddle-only module now (shared with the multilingual
# engine); re-exported here so existing imports `from ocr_engine import Detector`
# keep working.
from detector import Detector, DET_MODEL_NAME, MIN_DET_SCORE  # noqa: F401

MODEL_HUB_ID = "imperiusrex/Handwritten_model"  # re-save of microsoft/trocr-large-handwritten
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "local_trocr_model"

# crops more elongated than this get recognized in pieces (TrOCR squeezes
# everything to a 384x384 square)
ASPECT_CAP = 16.0
# a merged line crop whose height exceeds this multiple of the median member
# height probably swallowed two stacked lines -> fall back to per-box crops
MERGE_HEIGHT_GUARD = 1.8


def resolve_rec_source() -> str:
    """Prefer the local model dir when it is actually loadable."""
    if (LOCAL_MODEL_DIR / "model.safetensors").is_file() \
            and (LOCAL_MODEL_DIR / "config.json").is_file():
        return str(LOCAL_MODEL_DIR)
    return MODEL_HUB_ID


class Recognizer:
    """Batched TrOCR recognition."""

    def __init__(self, source: str | None = None, device: str | None = None,
                 fp16: bool | None = None):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.torch = torch
        self.source = source or resolve_rec_source()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        use_fp16 = fp16 if fp16 is not None else self.device.type == "cuda"
        dtype = torch.float16 if use_fp16 else torch.float32

        self.processor = TrOCRProcessor.from_pretrained(self.source)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.source, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, crops: list[np.ndarray], batch_size: int = 8,
                 num_beams: int = 1, max_new_tokens: int = 96) -> list[str]:
        texts: list[str] = []
        for start in range(0, len(crops), batch_size):
            # ascontiguousarray guards against negative-stride arrays (e.g. from
            # np.rot90/flips upstream), which torch.from_numpy refuses to wrap
            batch = [np.ascontiguousarray(c) for c in crops[start:start + batch_size]]
            pixel_values = self.processor(images=batch, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device, dtype=self.model.dtype)
            with self.torch.inference_mode():
                ids = self.model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=num_beams > 1,
                    # the shipped config says use_cache=false (a training
                    # leftover in microsoft's checkpoint); without the KV
                    # cache generation re-runs the whole decoder per token
                    use_cache=True,
                )
            texts.extend(self.processor.batch_decode(ids, skip_special_tokens=True))
        return [t.strip() for t in texts]


class OcrEngine:
    def __init__(self, rec_source: str | None = None, det_model: str = DET_MODEL_NAME,
                 device: str | None = None, fp16: bool | None = None):
        self.detector = Detector(det_model)
        self.recognizer = Recognizer(rec_source, device=device, fp16=fp16)

    def run(self, img_rgb: np.ndarray, merge_segments: bool = True,
            num_beams: int = 1, batch_size: int = 8) -> dict:
        """Full page OCR. Returns dict with text, lines, overlay, skew_deg, timing."""
        t0 = time.perf_counter()
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

        quads = self.detector(img_rgb)
        t_det = time.perf_counter()

        if len(quads) == 0:
            # fallback: treat the whole image as one line and say so
            text = self.recognizer([img_rgb], num_beams=num_beams)[0]
            return {
                "text": text,
                "lines": [],
                "overlay": img_rgb,
                "composite": pipeline.compose_transcript(img_rgb, [[]], [text]),
                "skew_deg": 0.0,
                "note": "no text boxes detected; whole-image fallback",
                "seconds": {"detect": t_det - t0, "recognize": time.perf_counter() - t_det},
            }

        lines, theta = pipeline.reading_order(quads)
        ordered = [pipeline.order_points(q) for q in quads]
        deskewed = [pipeline.rotate_points(q, -theta) for q in ordered]
        heights = [pipeline.quad_size(q)[1] for q in ordered]
        med_h = float(np.median(heights))

        # build crops: one per line-chunk when merging, else one per box
        crops: list[np.ndarray] = []
        owners: list[int] = []  # line index of each crop
        for li, line in enumerate(lines):
            chunks = pipeline.chunk_line(line, deskewed, aspect_cap=ASPECT_CAP) \
                if merge_segments else [[i] for i in line.members]
            for chunk in chunks:
                crop = None
                if merge_segments:
                    merged = pipeline.merge_quads([ordered[i] for i in chunk])
                    if pipeline.quad_size(merged)[1] <= MERGE_HEIGHT_GUARD * med_h:
                        crop = pipeline.perspective_crop(img_rgb, merged)
                if crop is not None:
                    crops.append(crop)
                    owners.append(li)
                else:  # per-box fallback (merge declined or degenerate)
                    for i in chunk:
                        w, h = pipeline.quad_size(ordered[i])
                        c = pipeline.perspective_crop(
                            img_rgb, ordered[i],
                            allow_rot90=h > 2.2 * med_h)
                        if c is not None:
                            crops.append(c)
                            owners.append(li)

        chunk_texts = self.recognizer(crops, batch_size=batch_size, num_beams=num_beams)
        line_texts = ["" for _ in lines]
        for text, li in zip(chunk_texts, owners):
            line_texts[li] = (line_texts[li] + " " + text).strip()

        text = pipeline.assemble_text(lines, line_texts)
        overlay = pipeline.annotate(img_rgb, quads, lines)
        line_boxes = [[ordered[i] for i in lines[k].members] for k in range(len(lines))]
        composite = pipeline.compose_transcript(img_rgb, line_boxes, line_texts)
        t_rec = time.perf_counter()

        return {
            "text": text,
            "lines": [
                {"text": line_texts[k], "boxes": [quads[i].tolist() for i in lines[k].members]}
                for k in range(len(lines))
            ],
            "overlay": overlay,
            "composite": composite,
            "skew_deg": theta,
            "seconds": {"detect": t_det - t0, "recognize": t_rec - t_det},
        }
