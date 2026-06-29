"""Multilingual full-page OCR: PP-OCRv5 detection -> reading-order
reconstruction (shared pipeline.py) -> PP-OCRv5 recognition.

Unlike the English pipeline (which recognizes with TrOCR), here *both* detection
and recognition are PaddleOCR PP-OCRv5 models, so it stays light and supports
every script PP-OCR ships. The contribution is the same deterministic
reading-order stage, inserted between PaddleOCR's detector and recognizer:

  * We run a strong detector (PP-OCRv5 *server* det) ourselves and recognize
    EVERY detected box. PaddleOCR's built-in pipeline silently drops boxes whose
    recognition confidence falls below an internal threshold, so words go
    missing from the output; recognizing each detected box keeps them.
  * Boxes are emitted in a geometry-derived reading order (deskew -> line
    cluster -> left-to-right within a line -> column split), instead of the
    detector's raw output order, so the transcript reads correctly.

Scope: left-to-right scripts (Latin, CJK, Indic). RTL (Arabic/Hebrew) is out of
scope — the ordering sorts left-to-right within each line.

CJK note: words inside a line are joined with no separator for CJK languages
(Chinese/Japanese/Korean have no inter-word spaces) and with a space otherwise,
so character-level CER is not inflated by spurious spaces.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# make `import pipeline` / `import ocr_engine` work when run from this subfolder
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import pipeline
from core.detector import Detector  # paddle-only detection wrapper (no TrOCR dependency)

# PP-OCRv5 server rec covers Chinese + English + Japanese (+ pinyin) in one
# model. Other scripts use their own PP-OCR language recognition model. These
# names are resolved at load time; PaddleRecognizer falls back gracefully if a
# given name is unavailable in the installed PaddleOCR version.
DEFAULT_REC_MODEL = "PP-OCRv5_server_rec"
REC_MODELS = {
    "ch": "PP-OCRv5_server_rec", "chinese": "PP-OCRv5_server_rec",
    "chinese_cht": "chinese_cht_PP-OCRv5_mobile_rec",
    "en": "PP-OCRv5_server_rec",
    "japan": "PP-OCRv5_server_rec", "ja": "PP-OCRv5_server_rec",
    "korean": "korean_PP-OCRv5_mobile_rec", "ko": "korean_PP-OCRv5_mobile_rec",
    "latin": "latin_PP-OCRv5_mobile_rec",
    "fr": "latin_PP-OCRv5_mobile_rec", "de": "latin_PP-OCRv5_mobile_rec",
    "es": "latin_PP-OCRv5_mobile_rec", "it": "latin_PP-OCRv5_mobile_rec",
    "pt": "latin_PP-OCRv5_mobile_rec",
}
# languages written without inter-word spaces -> join boxes with no separator
_NO_SPACE_LANGS = {"ch", "chinese", "chinese_cht", "japan", "ja", "korean", "ko"}


def rec_model_for(lang: str) -> str:
    return REC_MODELS.get(lang, DEFAULT_REC_MODEL)


class PaddleRecognizer:
    """Wraps a PaddleOCR PP-OCRv5 text-recognition model over RGB crops."""

    def __init__(self, model_name: str = DEFAULT_REC_MODEL,
                 enable_mkldnn: bool | None = None):
        from paddleocr import TextRecognition
        if enable_mkldnn is None:
            enable_mkldnn = sys.platform != "win32"  # PIR+oneDNN broken on win CPU
        self._cls = TextRecognition
        self._mkldnn = enable_mkldnn
        self.model_name = model_name
        self._rec = self._make(model_name, enable_mkldnn)

    def _make(self, model_name: str, mkldnn: bool):
        try:
            return self._cls(model_name=model_name, enable_mkldnn=mkldnn)
        except Exception as e:
            if model_name != DEFAULT_REC_MODEL:
                print(f"[ml] rec model {model_name!r} unavailable ({type(e).__name__}: "
                      f"{e}); falling back to {DEFAULT_REC_MODEL}")
                self.model_name = DEFAULT_REC_MODEL
                return self._cls(model_name=DEFAULT_REC_MODEL, enable_mkldnn=mkldnn)
            raise

    def __call__(self, crops: list[np.ndarray], batch_size: int = 16) -> list[str]:
        texts: list[str] = []
        for start in range(0, len(crops), batch_size):
            batch = [cv2.cvtColor(np.ascontiguousarray(c), cv2.COLOR_RGB2BGR)
                     for c in crops[start:start + batch_size]]
            try:
                results = self._rec.predict(batch, batch_size=len(batch))
            except NotImplementedError as e:  # PIR+oneDNN executor bug -> retry
                if "Pir" not in str(e) and "onednn" not in str(e).lower():
                    raise
                self._rec = self._cls(model_name=self.model_name, enable_mkldnn=False)
                results = self._rec.predict(batch, batch_size=len(batch))
            for r in results:
                txt = r.get("rec_text") if hasattr(r, "get") else None
                texts.append((txt or "").strip())
        return texts


class MultilingualOcrEngine:
    """PP-OCRv5 detection -> reading order -> PP-OCRv5 recognition."""

    def __init__(self, lang: str = "ch", det_model: str = "PP-OCRv5_server_det",
                 rec_model: str | None = None, crop_pad: float = 0.0):
        self.lang = lang
        self.word_sep = "" if lang in _NO_SPACE_LANGS else " "
        # PP-OCR rec models are trained on tight crops (the detector already
        # unclips each box), so default to zero extra padding + cubic warp to
        # match PaddleOCR's own get_rotate_crop_image; padding double-expands the
        # box and drags neighboring glyphs in, hurting recognition on dense pages.
        self.crop_pad = crop_pad
        self.detector = Detector(det_model)
        self.recognizer = PaddleRecognizer(rec_model or rec_model_for(lang))

    def run(self, img_rgb: np.ndarray, batch_size: int = 16,
            make_visuals: bool = True, column_split: bool = True) -> dict:
        """Full-page OCR. Returns text, per-line boxes/text, skew, timing, and
        (optionally) an overlay + side-by-side transcript composite."""
        t0 = time.perf_counter()
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

        quads = self.detector(img_rgb)
        t_det = time.perf_counter()
        if len(quads) == 0:
            return {"text": "", "lines": [], "skew_deg": 0.0,
                    "seconds": {"detect": t_det - t0, "recognize": 0.0},
                    "note": "no text boxes detected"}

        lines, theta = pipeline.reading_order(quads, column_split=column_split)
        ordered = [pipeline.order_points(q) for q in quads]
        med_h = float(np.median([pipeline.quad_size(q)[1] for q in ordered]))

        # one crop per detected box; recognize all, then assemble in reading order
        crops: list[np.ndarray] = []
        owner_line: list[int] = []
        owner_pos: list[int] = []
        for li, line in enumerate(lines):
            for pos, i in enumerate(line.members):
                h = pipeline.quad_size(ordered[i])[1]
                c = pipeline.perspective_crop(img_rgb, ordered[i],
                                              pad_frac=self.crop_pad,
                                              allow_rot90=h > 2.2 * med_h,
                                              interp=cv2.INTER_CUBIC)
                if c is not None:
                    crops.append(c)
                    owner_line.append(li)
                    owner_pos.append(pos)

        box_texts = self.recognizer(crops, batch_size=batch_size)

        line_words: list[list[tuple[int, str]]] = [[] for _ in lines]
        for txt, li, pos in zip(box_texts, owner_line, owner_pos):
            if txt:
                line_words[li].append((pos, txt))
        line_texts = [self.word_sep.join(t for _, t in sorted(ws))
                      for ws in line_words]

        # per-line tokens (text + box) in left-to-right order, so a downstream
        # consumer can reconstruct columns/tables from the geometry
        token_lines = [
            [{"text": txt, "box": ordered[lines[li].members[pos]].tolist()}
             for pos, txt in sorted(line_words[li])]
            for li in range(len(lines))
        ]

        text = pipeline.assemble_text(lines, line_texts)
        t_rec = time.perf_counter()

        out = {
            "text": text,
            "lines": [{"text": line_texts[k],
                       "boxes": [quads[i].tolist() for i in lines[k].members]}
                      for k in range(len(lines))],
            "token_lines": token_lines,
            "skew_deg": theta,
            "seconds": {"detect": t_det - t0, "recognize": t_rec - t_det},
            "n_boxes": int(len(quads)),
        }
        if make_visuals:
            out["overlay"] = pipeline.annotate(img_rgb, quads, lines)
            line_boxes = [[ordered[i] for i in lines[k].members]
                          for k in range(len(lines))]
            out["composite"] = pipeline.compose_transcript(
                img_rgb, line_boxes, line_texts)
        return out
