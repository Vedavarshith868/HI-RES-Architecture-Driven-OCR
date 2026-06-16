"""Generate multilingual_ocr.ipynb — a fully self-contained Colab notebook for the
multilingual pipeline (PP-OCRv5 detection → reading-order → PP-OCRv5 recognition),
evaluated against stock PaddleOCR on XFUND.

Embeds the source via %%writefile cells and auto-downloads XFUND, so the single
.ipynb runs top-to-bottom with no git clone and no manual data steps.

    python make_multilingual_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def writefile_cell(filename, path):
    body = (ROOT / path).read_text(encoding="utf-8")
    return code(f"%%writefile {filename}\n{body}")


cells = [
    md(
        "# Multilingual OCR — PP-OCR + HI-RES reading order (self-contained)\n"
        "\n"
        "A second, independent pipeline that brings the HI-RES reading-order stage to **multilingual\n"
        "document OCR**, where PaddleOCR PP-OCRv5 does *both* detection and recognition:\n"
        "\n"
        "```\n"
        "PP-OCRv5 server detection → reading-order reconstruction → PP-OCRv5 recognition (per language)\n"
        "```\n"
        "\n"
        "vs **stock `PaddleOCR(lang=…)`** on the same pages. The win is two-fold: we recognize **every**\n"
        "detected box (the stock pipeline drops low-confidence ones, so words vanish), and we emit them\n"
        "in **geometric reading order**. Left-to-right scripts (Latin + CJK).\n"
        "\n"
        "**Runtime → Change runtime type → T4 GPU** recommended (PP-OCRv5 server det is heavier on CPU).\n"
        "\n"
        "Self-contained: embeds the source, auto-downloads XFUND, runs top to bottom."
    ),
    md("## 1. Install dependencies (~2 min)"),
    code(
        "%pip install -q \"paddleocr>=3.0\" paddlepaddle datasets opencv-python-headless\n"
        "import paddleocr\n"
        "print('paddleocr', paddleocr.__version__)"
    ),
    md(
        "## 2. Project source (embedded)\n"
        "Geometry + detector + metrics, then the two multilingual modules. (No TrOCR here — recognition\n"
        "is PP-OCR, so this notebook is lighter than the handwriting one.)"
    ),
    writefile_cell("pipeline.py", "pipeline.py"),
    writefile_cell("ocr_engine.py", "ocr_engine.py"),
    writefile_cell("evaluate.py", "evaluate.py"),
    writefile_cell("ml_engine.py", "multilingual/ml_engine.py"),
    writefile_cell("ml_evaluate.py", "multilingual/ml_evaluate.py"),
    md(
        "## 3. Get XFUND data + pick models\n"
        "[XFUND](https://github.com/doc-analysis/XFUND) (CC-BY-4.0) is multilingual forms with\n"
        "per-segment text + boxes. Pick a language. We default to the **light PP-OCRv5 mobile** det+rec\n"
        "models: XFUND forms are box-dense, and on a **CPU runtime** the *server* detector is ~10× slower\n"
        "(a 40-page run can take an hour). If the GitHub release URL 404s, see the HF fallback at the bottom."
    ),
    code(
        "import urllib.request, zipfile\n"
        "from pathlib import Path\n"
        "\n"
        "XFUND_LANG = 'zh'      # zh | ja | es | fr | it | de | pt\n"
        "PPOCR_LANG = {'zh': 'ch', 'ja': 'japan', 'es': 'es', 'fr': 'fr',\n"
        "              'it': 'it', 'de': 'german', 'pt': 'pt'}[XFUND_LANG]\n"
        "# Light mobile models keep CPU runtimes usable (server det is ~10x slower on dense forms).\n"
        "DET_MODEL = 'PP-OCRv5_mobile_det'\n"
        "REC_MODEL = ('PP-OCRv5_mobile_rec' if XFUND_LANG in ('zh', 'ja')\n"
        "            else 'latin_PP-OCRv5_mobile_rec')\n"
        "\n"
        "DATA_DIR = f'xfund_{XFUND_LANG}'\n"
        "Path(DATA_DIR).mkdir(exist_ok=True)\n"
        "base = 'https://github.com/doc-analysis/XFUND/releases/download/v1.0'\n"
        "for fn in (f'{XFUND_LANG}.val.json', f'{XFUND_LANG}.val.zip'):\n"
        "    dst = Path(DATA_DIR, fn)\n"
        "    if not dst.exists():\n"
        "        print('downloading', fn, '...')\n"
        "        urllib.request.urlretrieve(f'{base}/{fn}', dst)\n"
        "with zipfile.ZipFile(Path(DATA_DIR, f'{XFUND_LANG}.val.zip')) as z:\n"
        "    z.extractall(DATA_DIR)\n"
        "print('XFUND', XFUND_LANG, 'ready in', DATA_DIR,\n"
        "      '| lang:', PPOCR_LANG, '| det:', DET_MODEL, '| rec:', REC_MODEL)"
    ),
    md(
        "## 4. Visual sanity check — one page\n"
        "Run the multilingual engine on a single page and look at the numbered boxes + reading-order\n"
        "transcript before trusting the metrics. The recognized text and the ground truth are printed\n"
        "below it."
    ),
    code(
        "import matplotlib.pyplot as plt\n"
        "import paddle\n"
        "import ml_evaluate as M, ml_engine as ML\n"
        "\n"
        "print('paddle device:', paddle.device.get_device(),\n"
        "      ' (cpu = slow; mobile models + small N help)')\n"
        "\n"
        "N_PAGES = 12          # XFUND forms are box-dense; raise once you know the per-page speed\n"
        "samples = M.load_xfund(DATA_DIR, PPOCR_LANG, split='val', n=N_PAGES)\n"
        "print(len(samples), 'pages loaded')\n"
        "\n"
        "engine = ML.MultilingualOcrEngine(lang=PPOCR_LANG, det_model=DET_MODEL, rec_model=REC_MODEL)\n"
        "s = samples[0]\n"
        "out = engine.run(s.image_rgb())\n"
        "\n"
        "comp = out['composite']\n"
        "ch, cw = comp.shape[:2]\n"
        "plt.figure(figsize=(16, max(6, 16 * ch / cw)))\n"
        "plt.imshow(comp); plt.axis('off')\n"
        "plt.title(s.name + ' — page + reading-order transcript'); plt.show()\n"
        "print('--- predicted (reading order) ---'); print(out['text'][:800])\n"
        "print('\\n--- ground truth ---'); print(s.gt[:800])"
    ),
    md(
        "## 5. Evaluate: HI-RES vs stock PaddleOCR\n"
        "Both scored on the same pages with the metrics from `evaluate.py`. **CER** is the headline;\n"
        "**WordAcc** (order-free) separates recognition from ordering. CJK is scored space-free (ignore\n"
        "WER there).\n"
        "\n"
        "`progress=True` prints one line per page (running CER + seconds) so you can watch it work — XFUND\n"
        "forms are dense, so expect a few seconds/page even with the mobile models on CPU. Lower `N_PAGES`\n"
        "in §4 if it's too slow."
    ),
    code(
        "import evaluate as E\n"
        "\n"
        "no_space = PPOCR_LANG in ML._NO_SPACE_LANGS\n"
        "norm = E.NormCfg()\n"
        "\n"
        "scores = [E.evaluate_system(f'hires-ml[{PPOCR_LANG}]', samples,\n"
        "                            M.hires_predict(engine, no_space), norm=norm, progress=True)]\n"
        "try:\n"
        "    scores.append(E.evaluate_system(f'paddle-stock[{PPOCR_LANG}]', samples,\n"
        "                                    M.builtin_predict(PPOCR_LANG, no_space),\n"
        "                                    norm=norm, progress=True))\n"
        "except Exception as e:\n"
        "    print('stock PaddleOCR skipped:', type(e).__name__, e)\n"
        "\n"
        "print('\\n' + E.format_table(scores))\n"
        "if no_space:\n"
        "    print('\\n(CJK scored space-free; read CER, ignore WER.)')\n"
        "E.save_csv(scores, 'ml_eval_results.csv')\n"
        "print('per-page breakdown -> ml_eval_results.csv')"
    ),
    md("## 6. Chart"),
    code(
        "import numpy as np, matplotlib.pyplot as plt\n"
        "names = [s.system for s in scores]; x = np.arange(len(names))\n"
        "fig, ax = plt.subplots(figsize=(1.9 * len(names) + 3, 4))\n"
        "ax.bar(x - 0.2, [s.cer for s in scores], 0.4, label='CER (lower better)')\n"
        "ax.bar(x + 0.2, [s.word_acc or 0 for s in scores], 0.4, label='WordAcc (higher better)')\n"
        "for i, s in enumerate(scores):\n"
        "    ax.text(i - 0.2, s.cer, f'{s.cer:.0%}', ha='center', va='bottom', fontsize=8)\n"
        "ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right')\n"
        "ax.legend(); ax.grid(axis='y', alpha=0.3)\n"
        "ax.set_title(f'XFUND {XFUND_LANG}: HI-RES vs stock PaddleOCR')\n"
        "plt.tight_layout(); plt.savefig('ml_comparison.png', dpi=130); plt.show()\n"
        "print('saved ml_comparison.png')"
    ),
    md(
        "## Notes\n"
        "- **What HI-RES adds** is *reading order* + *not dropping low-confidence boxes* — the recognizer\n"
        "  is unchanged PaddleOCR, so this is a detection-recall + ordering layer, not a recognition trick.\n"
        "- **Run more languages**: change `XFUND_LANG` at the top of §3 and re-run §3–§6.\n"
        "- **XFUND download fallback**: if the GitHub release 404s, load via Hugging Face instead, e.g.\n"
        "  `datasets.load_dataset('nielsr/XFUND', 'xfund.zh')`, write the images to `DATA_DIR`, and point\n"
        "  `load_xfund` at it — the loader also accepts any folder of `{lang}.val.json` + images.\n"
        "- Paste the printed table back and it goes into the repo's multilingual results section."
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = ROOT / "multilingual_ocr.ipynb"
target.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {target} ({target.stat().st_size:,} bytes, {len(cells)} cells)")
