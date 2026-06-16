"""Generate multilingual_ocr.ipynb — a fully self-contained Colab notebook for the
multilingual pipeline (PP-OCRv6 detection → reading-order → PP-OCRv6 recognition),
evaluated against stock PaddleOCR on XFUND.

Modern stack: PP-OCRv6 (June 2026, unified 50-language model) for BOTH detection
and recognition, on both sides, so the comparison isolates the reading-order
layer. Includes a skew-robustness sweep — the experiment that shows whether the
layer beats stock PaddleOCR's ordering on hard (rotated) layouts.

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
        "# Multilingual OCR — PP-OCRv6 + HI-RES reading order (self-contained)\n"
        "\n"
        "Brings the HI-RES deterministic **reading-order reconstruction** stage to multilingual document\n"
        "OCR, using **PP-OCRv6** (June 2026 — one unified model for 50 languages) for *both* detection and\n"
        "recognition, on both sides:\n"
        "\n"
        "```\n"
        "PP-OCRv6 detection → reading-order reconstruction → PP-OCRv6 recognition\n"
        "        vs   stock PaddleOCR running the same PP-OCRv6 models\n"
        "```\n"
        "\n"
        "Two experiments:\n"
        "1. **Clean forms (XFUND)** — controlled accuracy + speed at equal models (expect ~parity; the\n"
        "   layer's batched recognition is the speed edge).\n"
        "2. **Skew robustness** — rotate the pages and watch stock's reading order break while HI-RES\n"
        "   holds. *This* is the layer's real value and the basis for any PaddleOCR contribution.\n"
        "\n"
        "**Runtime → Change runtime type → T4 GPU** strongly recommended. Self-contained: embeds the\n"
        "source, auto-downloads XFUND, runs top to bottom."
    ),
    md("## 1. Install dependencies (~2 min)"),
    code(
        "%pip install -q -U paddleocr paddlepaddle datasets opencv-python-headless\n"
        "import paddleocr\n"
        "print('paddleocr', paddleocr.__version__, '(need a build with PP-OCRv6, i.e. mid-2026+)')"
    ),
    md(
        "## 2. Project source (embedded)\n"
        "Geometry + detector + metrics, then the two multilingual modules. (No TrOCR here — recognition\n"
        "is PP-OCR, so this notebook is light.)"
    ),
    writefile_cell("pipeline.py", "pipeline.py"),
    writefile_cell("ocr_engine.py", "ocr_engine.py"),
    writefile_cell("evaluate.py", "evaluate.py"),
    writefile_cell("ml_engine.py", "multilingual/ml_engine.py"),
    writefile_cell("ml_evaluate.py", "multilingual/ml_evaluate.py"),
    md(
        "## 3. Get XFUND data + PP-OCRv6 models\n"
        "[XFUND](https://github.com/doc-analysis/XFUND) (CC-BY-4.0) is multilingual forms with per-segment\n"
        "text + boxes. **PP-OCRv6 medium is one unified model for 50 languages**, so a single det+rec pair\n"
        "handles every language — no per-language rec model. `XFUND_LANG` only selects which dataset to\n"
        "download. If the GitHub release URL 404s, see the HF fallback at the bottom."
    ),
    code(
        "import urllib.request, zipfile\n"
        "from pathlib import Path\n"
        "\n"
        "XFUND_LANG = 'zh'      # zh | ja | es | fr | it | de | pt   (selects the dataset only)\n"
        "PPOCR_LANG = {'zh': 'ch', 'ja': 'japan', 'es': 'es', 'fr': 'fr',\n"
        "              'it': 'it', 'de': 'german', 'pt': 'pt'}[XFUND_LANG]\n"
        "\n"
        "# PP-OCRv6 medium: unified 50-language det+rec (one pair for ALL languages).\n"
        "# Lighter tiers if CPU is slow: 'PP-OCRv6_small_*' or 'PP-OCRv6_tiny_*'.\n"
        "DET_MODEL = 'PP-OCRv6_medium_det'\n"
        "REC_MODEL = 'PP-OCRv6_medium_rec'\n"
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
        "print('XFUND', XFUND_LANG, 'ready in', DATA_DIR, '| det:', DET_MODEL, '| rec:', REC_MODEL)"
    ),
    md(
        "## 4. Load data + visual sanity check\n"
        "Run the engine on one page and look at the numbered boxes + reading-order transcript before\n"
        "trusting metrics. `N_PAGES` is bigger here (40) for a meaningful sample; lower it if CPU is slow\n"
        "(the `paddle device` print tells you — `cpu` is much slower than a T4)."
    ),
    code(
        "import matplotlib.pyplot as plt\n"
        "import paddle\n"
        "import ml_evaluate as M, ml_engine as ML\n"
        "\n"
        "print('paddle device:', paddle.device.get_device(),\n"
        "      ' (cpu = slow; use a T4 GPU runtime, or a lighter PP-OCRv6 tier)')\n"
        "\n"
        "N_PAGES = 40          # bigger sample for a credible number; lower if CPU is slow\n"
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
        "## 5. Clean forms: accuracy **and speed** (controlled, same PP-OCRv6 models)\n"
        "Both run the **same** PP-OCRv6 det+rec, so the only differences are HI-RES's reading order,\n"
        "keep-every-box, and **batched recognition**. Two numbers matter, both in the table (`sec/img`) and\n"
        "summarized below it:\n"
        "- **CER** — expect ~parity on clean upright forms (naive order is already correct).\n"
        "- **Speed** — HI-RES collects every crop and runs the recognizer in batches, vs PaddleOCR's more\n"
        "  sequential per-page pass; the printed **× faster** ratio + pages/s is the speed claim. Both\n"
        "  predictors are **warmed up once** first so page-1 JIT isn't counted.\n"
        "\n"
        "CJK is scored space-free → WER/WordAcc show as `—`; read CER."
    ),
    code(
        "import evaluate as E\n"
        "\n"
        "no_space = PPOCR_LANG in ML._NO_SPACE_LANGS\n"
        "wm = not no_space          # CJK: no word boundaries -> skip WER/WordAcc\n"
        "norm = E.NormCfg()\n"
        "\n"
        "CONTROLLED = True          # stock uses the SAME PP-OCRv6 models as HI-RES (fair test)\n"
        "stock_det, stock_rec = (DET_MODEL, REC_MODEL) if CONTROLLED else (None, None)\n"
        "\n"
        "# build both predictors once (models load here) and REUSE them in §6\n"
        "hires = M.hires_predict(engine, no_space)\n"
        "try:\n"
        "    stock = M.builtin_predict(PPOCR_LANG, no_space, stock_det, stock_rec)\n"
        "except Exception as e:\n"
        "    stock = None; print('stock init failed:', type(e).__name__, e)\n"
        "\n"
        "# warm up once each so page-1 JIT/graph build is not counted in sec/img\n"
        "_ = hires(samples[0].image_rgb())\n"
        "if stock is not None:\n"
        "    _ = stock(samples[0].image_rgb())\n"
        "\n"
        "scores = [E.evaluate_system(f'hires-ml[{PPOCR_LANG}]', samples, hires,\n"
        "                            norm=norm, progress=True, word_metrics=wm)]\n"
        "if stock is not None:\n"
        "    scores.append(E.evaluate_system(f'paddle-stock[{PPOCR_LANG}]', samples, stock,\n"
        "                                    norm=norm, progress=True, word_metrics=wm))\n"
        "\n"
        "print('\\n' + E.format_table(scores))\n"
        "if no_space:\n"
        "    print('(CJK: no word boundaries -> WER/WordAcc are \\u2014; CER is the metric.)')\n"
        "if len(scores) == 2:\n"
        "    h, st = scores\n"
        "    ratio = st.sec_per_img / max(h.sec_per_img, 1e-9)\n"
        "    faster = 'faster' if ratio >= 1 else 'SLOWER'\n"
        "    print(f'\\nSPEED  hires {h.sec_per_img:.2f}s/page  vs  stock {st.sec_per_img:.2f}s/page'\n"
        "          f'  ->  {ratio:.1f}x {faster}'\n"
        "          f'   ({1 / h.sec_per_img:.2f} vs {1 / st.sec_per_img:.2f} pages/s)')\n"
        "E.save_csv(scores, 'ml_eval_results.csv')"
    ),
    md(
        "## 6. Skew-robustness stress test — the reading-order payoff\n"
        "Clean XFUND forms are upright, so naive top-to-bottom ordering is already right and §5 should be\n"
        "~parity. The layer earns its keep when layout is **hard**. Here we **rotate each page** by a fixed\n"
        "skew and re-score on the *same* ground truth (rotation doesn't change the words).\n"
        "\n"
        "- **Stock PaddleOCR** orders detected boxes by raw position → under skew, lines from different\n"
        "  rows interleave → wrong reading order → **CER rises**.\n"
        "- **HI-RES** estimates the page skew and clusters/orders lines in *deskewed* coordinates → it\n"
        "  should **stay flat**.\n"
        "\n"
        "The gap between the two curves is precisely what a reading-order contribution to PaddleOCR would\n"
        "close. If the curves *don't* diverge (stock already handles skew), there's no PR here — and that\n"
        "is exactly the thing to find out before proposing one."
    ),
    code(
        "import cv2, numpy as np, matplotlib.pyplot as plt\n"
        "\n"
        "def rotate_image(img, deg):\n"
        "    if not deg:\n"
        "        return img\n"
        "    h, w = img.shape[:2]\n"
        "    Mr = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)\n"
        "    return cv2.warpAffine(img, Mr, (w, h), borderValue=(255, 255, 255),\n"
        "                          flags=cv2.INTER_LINEAR)\n"
        "\n"
        "SKEW_ANGLES = [0, 5, 10]               # degrees; 0 = upright baseline\n"
        "SKEW_N = min(15, len(samples))         # pages per angle (N x angles x 2 systems runs)\n"
        "base = samples[:SKEW_N]\n"
        "\n"
        "res = {'hires': [], 'stock': []}\n"
        "for deg in SKEW_ANGLES:\n"
        "    sset = [E.Sample(name=f'{s.name}_r{deg}', gt=s.gt,\n"
        "                     image=rotate_image(s.image_rgb(), deg)) for s in base]\n"
        "    h = E.evaluate_system(f'hires@{deg}', sset, hires, norm=norm, word_metrics=wm)\n"
        "    res['hires'].append(h.cer)\n"
        "    line = f'skew {deg:2d}deg: hires CER {h.cer:.1%}'\n"
        "    if stock is not None:\n"
        "        st = E.evaluate_system(f'stock@{deg}', sset, stock, norm=norm, word_metrics=wm)\n"
        "        res['stock'].append(st.cer); line += f' | stock CER {st.cer:.1%}'\n"
        "    print(line, flush=True)\n"
        "\n"
        "plt.figure(figsize=(6.5, 4.5))\n"
        "plt.plot(SKEW_ANGLES, res['hires'], 'o-', lw=2, label='HI-RES (deskews + reorders)')\n"
        "if res['stock']:\n"
        "    plt.plot(SKEW_ANGLES, res['stock'], 's--', lw=2, label='stock PaddleOCR')\n"
        "plt.xlabel('page skew (degrees)'); plt.ylabel('CER  (lower is better)')\n"
        "plt.title(f'Reading-order robustness to skew — XFUND {XFUND_LANG}, PP-OCRv6')\n"
        "plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()\n"
        "plt.savefig('skew_robustness.png', dpi=130); plt.show()\n"
        "print('saved skew_robustness.png  ('\n"
        "      'diverging curves = the reading-order layer adds value; flat-together = it does not)')"
    ),
    md("## 7. Clean-forms chart (CER per system)"),
    code(
        "import numpy as np, matplotlib.pyplot as plt\n"
        "names = [s.system for s in scores]; x = np.arange(len(names))\n"
        "fig, ax = plt.subplots(figsize=(1.9 * len(names) + 3, 4))\n"
        "ax.bar(x, [s.cer for s in scores], 0.5, color=['#2563eb', '#9aa0a6'][:len(names)])\n"
        "for i, s in enumerate(scores):\n"
        "    ax.text(i, s.cer, f'{s.cer:.1%}\\n{s.sec_per_img:.1f}s', ha='center', va='bottom', fontsize=9)\n"
        "ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right')\n"
        "ax.set_ylabel('CER (lower better)'); ax.grid(axis='y', alpha=0.3)\n"
        "ax.set_title(f'XFUND {XFUND_LANG} (upright): HI-RES vs stock, same PP-OCRv6 models')\n"
        "plt.tight_layout(); plt.savefig('ml_comparison.png', dpi=130); plt.show()"
    ),
    md(
        "## Notes — reading this honestly\n"
        "- **§5 (clean forms)** measures recognition+crop parity and the batched-recognition speed edge.\n"
        "  HI-RES ≈ stock on CER here is the *expected, good* outcome — it means the layer is lossless on\n"
        "  easy layouts.\n"
        "- **Speed claim — stay rigorous.** The `× faster` is *end-to-end* at equal models. HI-RES batches\n"
        "  recognition (batch_size=16) across the whole page, while PaddleOCR's pipeline uses its own\n"
        "  default rec batching — so part of the gap may be a *config* difference, not architecture. Before\n"
        "  pitching speed as a PaddleOCR contribution, re-time stock with its rec batch raised; if the gap\n"
        "  survives, it's a genuine throughput win (batched whole-page recognition).\n"
        "- **§6 (skew)** is the decisive one. A widening CER gap (stock up, HI-RES flat) is reproducible\n"
        "  evidence that PaddleOCR's box ordering breaks under skew and a geometry-based reorder fixes it —\n"
        "  the seed of a focused PaddleOCR contribution (a reading-order post-process / pipeline option).\n"
        "  No gap ⇒ no PR, and that is worth knowing.\n"
        "- **Bigger / harder:** raise `N_PAGES`, add angles to `SKEW_ANGLES`, and try other `XFUND_LANG`\n"
        "  values to show it generalizes across scripts.\n"
        "- **XFUND download fallback**: if the GitHub release 404s, load via Hugging Face, e.g.\n"
        "  `datasets.load_dataset('nielsr/XFUND', 'xfund.zh')`, write the images to `DATA_DIR`, and point\n"
        "  `load_xfund` at it (it accepts any folder of `{lang}.val.json` + images).\n"
        "- Paste both tables + the skew figure back, and we decide the PR form from the data."
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
