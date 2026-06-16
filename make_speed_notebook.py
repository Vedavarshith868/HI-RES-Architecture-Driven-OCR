"""Generate speed_benchmark.ipynb — a fully self-contained Colab notebook that
times the HI-RES pipeline against the PP-OCRv5 server pipeline on the same pages.

Like make_colab_notebook.py, it embeds the source files via %%writefile cells, so
the single .ipynb runs top-to-bottom with no git clone and no manual data upload
(test pages are synthesized from real IAM handwriting lines, auto-downloaded).

    python make_speed_notebook.py
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
        "# HI-RES — Inference-speed benchmark (self-contained)\n"
        "\n"
        "Times the **HI-RES** pipeline (PP-OCRv5 detection → reading-order → TrOCR-large) against the\n"
        "**PP-OCRv5 server** built-in det+rec pipeline on the *same* pages, split per stage so the\n"
        "bottleneck is explicit. HI-RES and PP-OCRv5 server reach near-identical accuracy on GNHK\n"
        "(29.5% vs 28.3% CER), so the open question is throughput.\n"
        "\n"
        "**Runtime → Change runtime type → T4 GPU.** TrOCR is GPU-bound, and the PP-OCRv5 *server*\n"
        "recognizer is unstable on CPU — on a CPU runtime set the baseline tier to `mobile` (or skip it).\n"
        "\n"
        "Fully self-contained: embeds the source, synthesizes real-handwriting test pages from IAM\n"
        "(auto-downloaded — no manual upload), and runs top to bottom."
    ),
    md("## 1. Install dependencies (~2 min)"),
    code(
        "%pip install -q \"paddleocr>=3.0\" paddlepaddle \"transformers>=4.45\" safetensors datasets\n"
        "import torch, paddleocr, transformers\n"
        "print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(),\n"
        "      '| paddleocr', paddleocr.__version__, '| transformers', transformers.__version__)"
    ),
    md("## 2. Project source (embedded — geometry, engine, harness, benchmark)"),
    writefile_cell("pipeline.py", "pipeline.py"),
    writefile_cell("ocr_engine.py", "ocr_engine.py"),
    writefile_cell("evaluate.py", "evaluate.py"),
    writefile_cell("benchmark_speed.py", "benchmark_speed.py"),
    md(
        "## 3. Build test pages\n"
        "Pick where the test images come from with `SOURCE`:\n"
        "- `'synth'` (default): synthesizes **real-handwriting** multi-line pages by stacking IAM lines\n"
        "  with a small skew (auto-downloaded — no manual step).\n"
        "- `'gnhk'`: samples `N_PAGES` images from a **GNHK** folder you've already unzipped on Colab\n"
        "  (`GNHK_DIR`, default `/content/gnhk`). GNHK is behind a CC-BY click-through, so there's no\n"
        "  direct download — upload/unzip it first (see §12 of `colab_ocr_debug.ipynb`).\n"
        "- `'upload'`: pick image files from your machine.\n"
        "\n"
        "Speed is **independent of ground truth** — only the images are needed, no `.json`/`.txt`. The\n"
        "benchmark runs OCR but measures *time* and discards the text; per-image timings still land in\n"
        "`speed_benchmark.csv`."
    ),
    code(
        "import cv2, shutil\n"
        "from pathlib import Path\n"
        "import evaluate as E\n"
        "\n"
        "SOURCE = 'synth'        # 'synth' = IAM (auto) | 'gnhk' = sample a GNHK folder | 'upload' = pick files\n"
        "IMAGES_DIR = 'speed_images'\n"
        "N_PAGES = 12\n"
        "GNHK_DIR = '/content/gnhk'   # used when SOURCE='gnhk' (unzip GNHK here first)\n"
        "Path(IMAGES_DIR).mkdir(exist_ok=True)\n"
        "src = SOURCE.lower()\n"
        "\n"
        "if src == 'synth':\n"
        "    pages = E.build_iam_pages(n_pages=N_PAGES, lines_per_page=(5, 9),\n"
        "                              max_skew_deg=3.0, seed=0)\n"
        "    for s in pages:\n"
        "        cv2.imwrite(f'{IMAGES_DIR}/{s.name}.png',\n"
        "                    cv2.cvtColor(s.image_rgb(), cv2.COLOR_RGB2BGR))\n"
        "    print('wrote', len(pages), 'synthetic IAM pages to', IMAGES_DIR)\n"
        "elif src == 'gnhk':\n"
        "    root = Path(GNHK_DIR)\n"
        "    assert root.exists(), (f'{GNHK_DIR} not found — upload/unzip the GNHK zip there first '\n"
        "                           '(CC-BY click-through at goodnotes.com/gnhk).')\n"
        "    # prefer the test split if present, then take the first N image files\n"
        "    test_dirs = [p for p in root.rglob('*') if p.is_dir() and 'test' in p.name.lower()]\n"
        "    scan = test_dirs[0] if test_dirs else root\n"
        "    imgs = sorted(p for p in scan.rglob('*') if p.suffix.lower() in E.IMAGE_EXTS)[:N_PAGES]\n"
        "    assert imgs, f'no images found under {scan}'\n"
        "    for p in imgs:\n"
        "        shutil.copy(p, Path(IMAGES_DIR, p.name))\n"
        "    print(f'copied {len(imgs)} GNHK images from {scan} to {IMAGES_DIR}')\n"
        "else:  # 'upload'\n"
        "    from google.colab import files\n"
        "    up = files.upload()\n"
        "    for name, data in up.items():\n"
        "        Path(IMAGES_DIR, name).write_bytes(data)\n"
        "    print('saved', len(up), 'uploaded image(s) to', IMAGES_DIR)"
    ),
    md(
        "## 4. Run the benchmark\n"
        "Loads TrOCR (≈2.2 GB from the Hub, fp16 on GPU) and the PP-OCRv5 models, warms up, then times\n"
        "each page. HI-RES is split into **detect / reading-order / recognize**; the baseline is timed\n"
        "end-to-end. Both share the PP-OCRv5 detector, so the gap is the recognizer.\n"
        "\n"
        "If the server baseline crashes on a CPU runtime, set `BASELINE_TIER = 'mobile'` or\n"
        "`RUN_BASELINE = False`."
    ),
    code(
        "import benchmark_speed as B\n"
        "\n"
        "BEAMS = 1               # TrOCR beams (1 = greedy, fastest)\n"
        "WARMUP = 2              # warmup pages, excluded from timing\n"
        "BASELINE_TIER = 'server'   # 'mobile' if server crashes on CPU\n"
        "RUN_BASELINE = True\n"
        "\n"
        "paths = B.list_images(IMAGES_DIR)\n"
        "imgs = [B.read_rgb(p) for p in paths]\n"
        "print(f'{len(imgs)} pages from {IMAGES_DIR}')\n"
        "\n"
        "ht = B.HiResTimer(beams=BEAMS)\n"
        "print('TrOCR on', ht.device)\n"
        "hires = B.run_timer(ht, imgs, WARMUP, 'HI-RES')\n"
        "\n"
        "paddle = None\n"
        "if RUN_BASELINE:\n"
        "    try:\n"
        "        pt = B.PaddleServerTimer(tier=BASELINE_TIER)\n"
        "        paddle = B.run_timer(pt, imgs, WARMUP, f'PP-OCRv5 {BASELINE_TIER}')\n"
        "    except Exception as e:\n"
        "        print('baseline failed:', type(e).__name__, e)\n"
        "\n"
        "print('\\n' + B.report(hires, paddle, ht.device, BEAMS, WARMUP,\n"
        "                       f'PP-OCRv5 {BASELINE_TIER}'))"
    ),
    md("## 5. Chart: per-stage time per page"),
    code(
        "import statistics as st\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "det = st.fmean(r['detect'] for r in hires)\n"
        "order = st.fmean(r['order+crop'] for r in hires)\n"
        "rec = st.fmean(r['recognize'] for r in hires)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 4.5))\n"
        "ax.bar('HI-RES', det, label='detect (PP-OCRv5)')\n"
        "ax.bar('HI-RES', order, bottom=det, label='reading-order')\n"
        "ax.bar('HI-RES', rec, bottom=det + order, label='recognize (TrOCR-large)')\n"
        "if paddle:\n"
        "    p = st.fmean(r['total'] for r in paddle)\n"
        "    ax.bar(f'PP-OCRv5 {BASELINE_TIER}', p, color='#9aa0a6', label='total (det+rec)')\n"
        "ax.set_ylabel('seconds / page  (lower is faster)')\n"
        "ax.set_title('Inference time per page')\n"
        "ax.legend(); ax.grid(axis='y', alpha=0.3)\n"
        "plt.tight_layout(); plt.savefig('speed_comparison.png', dpi=130); plt.show()\n"
        "print('reading-order overhead: %.1f ms/page (the geometry stage is never the bottleneck)'\n"
        "      % (1000 * order))"
    ),
    md(
        "## What this shows\n"
        "Both systems share the PP-OCRv5 **detector**, so the timing difference is the **recognizer**:\n"
        "TrOCR-large is an autoregressive decoder (heavier, handwriting-grade) while PP-OCRv5 server rec\n"
        "is a CTC head (lighter). The **reading-order stage is geometry on a few hundred boxes —\n"
        "sub-millisecond per page** — so HI-RES's extra cost buys recognition quality, not ordering.\n"
        "\n"
        "Paste the printed report block back and it goes straight into the repo's *Inference speed* table."
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

target = ROOT / "speed_benchmark.ipynb"
target.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {target} ({target.stat().st_size:,} bytes, {len(cells)} cells)")
