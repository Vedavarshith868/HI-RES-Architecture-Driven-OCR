"""PaddleOCR text-detection wrapper → clean (N,4,2) quads in image coordinates.

Paddle-only (no torch/transformers), so it is shared by *both* pipelines — the
handwriting engine (`ocr_engine.py`, TrOCR recognition) and the multilingual
engine (`multilingual/ml_engine.py`, PP-OCR recognition) — without either
dragging in the other's recognizer.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from core import pipeline

DET_MODEL_NAME = os.environ.get("OCR_DET_MODEL", "PP-OCRv5_server_det")
MIN_DET_SCORE = 0.30


class Detector:
    """PP-OCRv5 text detection wrapper that returns clean (N,4,2) quads in
    original-image coordinates."""

    def __init__(self, model_name: str = DET_MODEL_NAME,
                 enable_mkldnn: bool | None = None):
        from paddleocr import TextDetection  # lazy: paddle is optional
        self._cls = TextDetection
        self._model_name = model_name
        if enable_mkldnn is None:
            env = os.environ.get("OCR_DET_MKLDNN")
            # paddle's PIR + oneDNN executor is broken on Windows CPU
            # (ConvertPirAttribute2RuntimeAttribute NotImplementedError)
            enable_mkldnn = env == "1" if env is not None else sys.platform != "win32"
        self._det = TextDetection(model_name=model_name, enable_mkldnn=enable_mkldnn)

    def __call__(self, img_rgb: np.ndarray, min_score: float = MIN_DET_SCORE) -> np.ndarray:
        # PaddleOCR consumes cv2-style BGR; gradio/PIL hand us RGB.
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        try:
            results = self._det.predict(img_bgr, batch_size=1)
        except NotImplementedError as e:  # oneDNN executor bug -> retry without it
            if "Pir" not in str(e) and "onednn" not in str(e).lower():
                raise
            self._det = self._cls(model_name=self._model_name, enable_mkldnn=False)
            results = self._det.predict(img_bgr, batch_size=1)

        quads: list[np.ndarray] = []
        for res in results:
            polys = res.get("dt_polys", None) if hasattr(res, "get") else None
            if polys is None:
                continue
            scores = res.get("dt_scores", None)
            for k, poly in enumerate(polys):
                score = float(scores[k]) if scores is not None and k < len(scores) else 1.0
                if score < min_score:
                    continue
                arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
                if arr.shape[0] > 4:  # polygon mode -> reduce to min-area quad
                    arr = pipeline.merge_quads([arr])
                if arr.shape[0] != 4:
                    continue
                w, h = pipeline.quad_size(pipeline.order_points(arr))
                if w < 2 or h < 2:
                    continue
                quads.append(arr)
        return np.array(quads, dtype=np.float64) if quads else np.zeros((0, 4, 2))
