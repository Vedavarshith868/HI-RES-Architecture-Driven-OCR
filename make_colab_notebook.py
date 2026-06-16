"""Generate colab_ocr_debug.ipynb from the current source files.

The notebook embeds pipeline.py / ocr_engine.py / test_geometry.py via
%%writefile cells so it is fully self-contained — upload the single .ipynb
to Google Colab and run top to bottom. Re-run this script after editing the
source files to keep the notebook in sync:

    python make_colab_notebook.py
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
        "# PolyOCR — Colab debug notebook\n"
        "\n"
        "Full pipeline: **PP-OCRv5 detection → coordinate-based reading order → TrOCR-large recognition**.\n"
        "\n"
        "**Before running:** `Runtime → Change runtime type → T4 GPU` (recognition is ~10× faster; the notebook also works on CPU).\n"
        "\n"
        "Run cells top to bottom. Section 5 lets you upload your own photos; sections 6–7 are the\n"
        "ordering diagnostics — they show why the *old* app's order changed from photo to photo and\n"
        "prove the new ordering is independent of detector output order.\n"
    ),
    md("## 1. Install dependencies (~2 min)"),
    code(
        "%pip install -q \"paddleocr>=3.0\" paddlepaddle \"transformers>=4.45\" safetensors\n"
        "import paddleocr, transformers, torch\n"
        "print('paddleocr', paddleocr.__version__, '| transformers', transformers.__version__,\n"
        "      '| torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
    ),
    md("## 2. Pipeline source (geometry / reading order — pure numpy+cv2)"),
    writefile_cell("pipeline.py", "pipeline.py"),
    md("## 3. Engine source (detection + recognition glue)"),
    writefile_cell("detector.py", "detector.py"),
    writefile_cell("ocr_engine.py", "ocr_engine.py"),
    md("## 4. Geometry unit tests (no models needed, ~1 s)\n20 tests incl. a fixture where naive y-sorting provably fails."),
    writefile_cell("test_geometry.py", "tests/test_geometry.py"),
    code("!python -m unittest test_geometry -v"),
    md(
        "## 5. Load models and run on your own photos\n"
        "Loads TrOCR from the Hub (`imperiusrex/Handwritten_model`, ~2.2 GB, fp16 on GPU) and\n"
        "PP-OCRv5 server detection.\n"
        "\n"
        "**Giving it photos — set `IMAGE_SOURCE` in the cell:**\n"
        "- `\"ask\"` (default): every run opens a *Choose Files* dialog and uses **only the files you\n"
        "  pick that run**, so testing a new image never silently reuses the previous one.\n"
        "- `\"folder\"`: open the 📁 sidebar, drag images into the **`inputs`** folder, then re-run;\n"
        "  it reads whatever is in that folder (handy for batches).\n"
        "\n"
        "**Output:** the recognized text is **printed right here inline** (no download needed), and\n"
        "the result image — the page with numbered boxes on the left and the matching numbered\n"
        "transcript beside it — is shown below each photo. Section 8 can zip everything if you want\n"
        "the files too."
    ),
    code(
        "import os\n"
        "os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'\n"
        "from ocr_engine import OcrEngine\n"
        "\n"
        "MERGE_SEGMENTS = True   # merge same-line fragments into one crop (recommended)\n"
        "NUM_BEAMS = 1           # try 4 for slightly better accuracy, slower\n"
        "\n"
        "engine = OcrEngine()\n"
        "print('TrOCR on', engine.recognizer.device, '| dtype', engine.recognizer.model.dtype)"
    ),
    code(
        "import cv2\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from pathlib import Path\n"
        "\n"
        "# Where images come from each run:\n"
        "#   'ask'    -> always open the upload dialog and use ONLY the files you pick now\n"
        "#               (so a previously-tested image is never silently reused)\n"
        "#   'folder' -> read images from INPUT_DIR; drag files into that sidebar folder\n"
        "IMAGE_SOURCE = 'ask'\n"
        "INPUT_DIR = '/content/inputs'\n"
        "EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}\n"
        "\n"
        "def load_images():\n"
        "    if IMAGE_SOURCE == 'folder':\n"
        "        d = Path(INPUT_DIR); d.mkdir(parents=True, exist_ok=True)\n"
        "        paths = sorted(p for p in d.rglob('*')\n"
        "                       if p.suffix.lower() in EXTS\n"
        "                       and '_panel' not in p.stem and '_overlay' not in p.stem)\n"
        "        if not paths:\n"
        "            print(f\"No images in {INPUT_DIR}. Open the sidebar (folder icon), \"\n"
        "                  f\"drag images into 'inputs', and re-run.\")\n"
        "        else:\n"
        "            print(f'Using {len(paths)} image(s) from {INPUT_DIR}:')\n"
        "            for p in paths: print('  ', p.name)\n"
        "        return {p.name: p.read_bytes() for p in paths}\n"
        "    from google.colab import files          # 'ask' (default)\n"
        "    print('Pick image(s) — only the files you select now are used:')\n"
        "    return files.upload()\n"
        "\n"
        "images = load_images()\n"
        "\n"
        "results = {}   # reset every run, so an earlier image never lingers\n"
        "for name, data in images.items():\n"
        "    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)\n"
        "    if img is None:\n"
        "        print(f'!! could not decode {name}'); continue\n"
        "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n"
        "    res = engine.run(img, merge_segments=MERGE_SEGMENTS, num_beams=NUM_BEAMS)\n"
        "    results[name] = (img, res)\n"
        "    # main visual: page with numbered boxes + matching transcript beside it\n"
        "    comp = res['composite']\n"
        "    ch, cw = comp.shape[:2]\n"
        "    plt.figure(figsize=(18, max(5, 18 * ch / cw)))\n"
        "    plt.imshow(comp); plt.axis('off')\n"
        "    plt.title(f\"{name} — page + transcript (numbers match)\"); plt.show()\n"
        "    print(f\"--- {name} | skew {res['skew_deg']:.1f}° | det {res['seconds']['detect']:.1f}s\"\n"
        "          f\" | rec {res['seconds']['recognize']:.1f}s ---\")\n"
        "    if res.get('note'): print('[', res['note'], ']')\n"
        "    print(res['text'])   # also printed inline as plain copy-pasteable text\n"
        "    print()"
    ),
    md(
        "## 6. Why the old app's order differed per photo\n"
        "The old `app.py` used the **detector's output order, reversed** (`cropped_images.reverse()`).\n"
        "That order is an undefined implementation detail of `cv2.findContours`, so it changes with\n"
        "image content — right on some photos, scrambled on others. Left: old behavior. Right: new\n"
        "coordinate-based order. (Uses the last photo you uploaded.)"
    ),
    code(
        "import pipeline\n"
        "\n"
        "name, (img, _res) = list(results.items())[-1]\n"
        "quads = engine.detector(img)\n"
        "print(f'{name}: {len(quads)} boxes detected')\n"
        "\n"
        "legacy_lines = [pipeline.Line(members=[i], top=0.0, bottom=0.0)\n"
        "                for i in reversed(range(len(quads)))]   # what legacy_app.py did\n"
        "new_lines, theta = pipeline.reading_order(quads)\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(22, 11))\n"
        "axes[0].imshow(pipeline.annotate(img, list(quads), legacy_lines))\n"
        "axes[0].set_title('OLD: detector order, reversed'); axes[0].axis('off')\n"
        "axes[1].imshow(pipeline.annotate(img, list(quads), new_lines))\n"
        "axes[1].set_title(f'NEW: coordinate-based (skew {theta:.1f}°)'); axes[1].axis('off')\n"
        "plt.show()"
    ),
    md(
        "## 7. Proof: new ordering is independent of detector output order\n"
        "We shuffle the detected boxes randomly 5 times and recompute the reading order. The\n"
        "resulting sequence of boxes (identified by centroid) must be identical every time —\n"
        "i.e. the failure mode you saw (order changing run to run / photo to photo) cannot occur."
    ),
    code(
        "import random\n"
        "\n"
        "def order_signature(quad_list):\n"
        "    lines, _ = pipeline.reading_order(quad_list)\n"
        "    return [tuple(np.round(np.asarray(quad_list[i]).mean(axis=0), 1))\n"
        "            for line in lines for i in line.members]\n"
        "\n"
        "base = order_signature(list(quads))\n"
        "for seed in range(5):\n"
        "    shuffled = list(quads)\n"
        "    random.Random(seed).shuffle(shuffled)\n"
        "    assert order_signature(shuffled) == base, f'order changed under shuffle (seed {seed})!'\n"
        "print('PASS: reading order identical under 5 random shuffles of detector output')"
    ),
    md("## 8. Download results (texts + overlays as a zip)"),
    code(
        "import zipfile\n"
        "from pathlib import Path\n"
        "\n"
        "out = Path('ocr_results.zip')\n"
        "with zipfile.ZipFile(out, 'w') as z:\n"
        "    for name, (img, res) in results.items():\n"
        "        stem = Path(name).stem\n"
        "        z.writestr(f'{stem}_ocr.txt', res['text'])\n"
        "        for tag, key in (('panel', 'composite'), ('overlay', 'overlay')):\n"
        "            ok, buf = cv2.imencode('.png', cv2.cvtColor(res[key], cv2.COLOR_RGB2BGR))\n"
        "            if ok: z.writestr(f'{stem}_{tag}.png', buf.tobytes())\n"
        "from google.colab import files\n"
        "files.download(str(out))"
    ),
    md(
        "## 9. Measure accuracy (CER / WER) and compare against other OCRs\n"
        "This is how you answer *\"does my pipeline beat anything?\"* — every system is scored on the\n"
        "**same** images, so the numbers are comparable.\n"
        "\n"
        "**Metrics:**\n"
        "- **CER / WER** (lower is better) — character / word error rate vs. ground truth\n"
        "  (corpus-aggregated, the standard way; can exceed 100% if a system hallucinates). The headline.\n"
        "- **WordAcc** (higher is better) — order-free word accuracy: fraction of ground-truth words\n"
        "  present in the prediction regardless of position. Separates recognition from reading order\n"
        "  cleanly (high WordAcc + high CER ⇒ ordering/segmentation issue; low WordAcc ⇒ recognition issue).\n"
        "\n"
        "**You need ground truth.** Put each image and its correct transcript together: for `foo.jpg`,\n"
        "a sibling `foo.txt` with one physical line per line, in reading order. Drag both into the\n"
        "`eval_data` folder (sidebar). Start with **5–10 of your own pages** — even that gives signal.\n"
        "For a standard benchmark, see the note at the bottom (IAM / GNHK)."
    ),
    writefile_cell("evaluate.py", "evaluate.py"),
    md("### 9a. Metric self-tests (proves the math; no data needed)"),
    code("!python -m unittest discover -s . -p 'test_metrics.py' -v"),
    md(
        "### 9b. Score this pipeline on your labeled data\n"
        "Set `EVAL_DIR`, drop image+`.txt` pairs in it, run."
    ),
    code(
        "import evaluate as E\n"
        "from pathlib import Path\n"
        "\n"
        "EVAL_DIR = '/content/eval_data'\n"
        "Path(EVAL_DIR).mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "# normalization applied to BOTH prediction and ground truth before scoring:\n"
        "NORM = E.NormCfg(lowercase=False, strip_punct=False)  # standard CER\n"
        "\n"
        "samples = E.load_pairs(EVAL_DIR)\n"
        "print(f'{len(samples)} labeled sample(s) in {EVAL_DIR}')\n"
        "assert samples, ('No image+.txt pairs found. Drag e.g. page1.jpg AND page1.txt '\n"
        "                 'into the eval_data folder, then re-run.')\n"
        "\n"
        "pt, _ = E.pipeline_predictors(engine)          # reuse the engine from section 5\n"
        "scores = [E.evaluate_system('this-pipeline', samples, pt, norm=NORM)]\n"
        "print(); print(E.format_table(scores))"
    ),
    md(
        "### 9c. Add baselines (Tesseract / EasyOCR / PaddleOCR) on the same data\n"
        "Installs are optional; skip any you don't want. Then they appear in the same table."
    ),
    code(
        "# Tesseract + EasyOCR (PaddleOCR is already installed from section 1)\n"
        "!apt-get -qq install -y tesseract-ocr >/dev/null && pip -q install pytesseract easyocr\n"
        "\n"
        "BASELINES = ['tesseract', 'easyocr', 'paddleocr']   # trim as you like\n"
        "for b in BASELINES:\n"
        "    try:\n"
        "        scores.append(E.evaluate_system(b, samples, E.baseline_predict(b), norm=NORM))\n"
        "        print(f'scored {b}')\n"
        "    except Exception as ex:\n"
        "        print(f'skip {b}: {type(ex).__name__}: {ex}')\n"
        "\n"
        "print(); print(E.format_table(scores))\n"
        "E.save_csv(scores, 'eval_results.csv')\n"
        "print('\\nper-image breakdown saved to eval_results.csv')"
    ),
    code(
        "# bar chart of CER/WER per system\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "names = [s.system for s in scores]\n"
        "x = np.arange(len(names))\n"
        "fig, ax = plt.subplots(figsize=(1.6 * len(names) + 3, 4))\n"
        "ax.bar(x - 0.2, [s.cer for s in scores], 0.4, label='CER')\n"
        "ax.bar(x + 0.2, [s.wer for s in scores], 0.4, label='WER')\n"
        "ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right')\n"
        "ax.set_ylabel('error rate'); ax.legend(); ax.set_title('Lower is better')\n"
        "ax.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()"
    ),
    md(
        "## 10. Score on a public dataset (IAM) — a citable number\n"
        "No registration or manual labeling: `evaluate.load_hf` streams **`Teklia/IAM-line`** (IAM\n"
        "handwriting test lines, image+text, not gated) straight from Hugging Face. IAM lines are\n"
        "single lines, so this scores the **recognizer alone** (no detection/ordering) — directly\n"
        "comparable to the published **TrOCR-large CER ≈ 2.9%**.\n"
        "\n"
        "Start with `N = 200` for a ~1-minute check; set `N = None` for the full 2,920-line test set\n"
        "(a few minutes on a T4)."
    ),
    code(
        "%pip install -q datasets\n"
        "import evaluate as E\n"
        "\n"
        "N = 200          # None = full IAM test (2,920 lines)\n"
        "NORM = E.NormCfg(lowercase=False, strip_punct=False)   # standard CER\n"
        "\n"
        "iam = E.load_hf('iam-lines', split='test', n=N)\n"
        "print(f'loaded {len(iam)} IAM test lines')\n"
        "print('example GT:', repr(iam[0].gt))\n"
        "\n"
        "# reuse the recognizer already loaded in section 5 (engine.recognizer)\n"
        "iam_scores = [E.evaluate_recognizer('trocr-recognizer', iam,\n"
        "                                    recognizer=engine.recognizer, norm=NORM)]\n"
        "print(); print(E.format_table(iam_scores))\n"
        "print('\\n(compare CER to the published TrOCR-large IAM number, ~2.9%)')"
    ),
    md(
        "### 10a. (optional) Baselines on the same IAM lines\n"
        "Same images, other engines — the only fair comparison. Skip if you only wanted your number."
    ),
    code(
        "# needs the installs from section 9c (tesseract/easyocr); paddleocr already present\n"
        "for b in ['tesseract', 'easyocr', 'paddleocr']:\n"
        "    try:\n"
        "        iam_scores.append(E.evaluate_system(b, iam, E.baseline_predict(b), norm=NORM))\n"
        "        print(f'scored {b}')\n"
        "    except Exception as ex:\n"
        "        print(f'skip {b}: {type(ex).__name__}: {ex}')\n"
        "print(); print(E.format_table(iam_scores))\n"
        "E.save_csv(iam_scores, 'iam_results.csv')"
    ),
    md(
        "## 11. Full-page evaluation (your pipeline's actual job)\n"
        "IAM lines (section 10) only score the recognizer. These score the **whole pipeline** —\n"
        "detection → reading order → recognition — on multi-line images, using\n"
        "`E.pipeline_predictors(engine)`. Compare **CER** (penalizes wrong reading order) against\n"
        "**WordAcc** (order-free): high WordAcc with high CER means recognition is fine but lines are\n"
        "out of order.\n"
        "\n"
        "These IAM-based pages (11a, 11b) are quick **detection + reading-order** checks. Your headline\n"
        "resume comparison vs other OCRs is **Section 12 (GNHK)** — real photos, unbiased.\n"
        "\n"
        "- **11a IAM_Sentences** — real multi-line IAM images, streamed from HF. One line of setup.\n"
        "- **11b Synthetic IAM pages** — stacks real IAM lines into skewed, indented pages; the skew\n"
        "  makes naive top-to-bottom ordering fail, so it genuinely tests reading order.\n"
        "\n"
        "Caveat baked in: TrOCR was trained on IAM, so IAM recognition CER is optimistic. IAM still\n"
        "validly tests detection+ordering (layout was never trained); GNHK (§12) gives the unbiased number."
    ),
    md("### 11a. IAM_Sentences — real multi-line images (HF)"),
    code(
        "import evaluate as E\n"
        "NORM = E.NormCfg(lowercase=False, strip_punct=False)\n"
        "\n"
        "PAGES_N = 30                       # cap for a quick run; raise for a fuller number\n"
        "sent = E.load_hf('iam-sentences', n=PAGES_N)\n"
        "print(f'loaded {len(sent)} multi-line IAM images')\n"
        "\n"
        "pt, _ = E.pipeline_predictors(engine)      # full detect -> order -> recognize\n"
        "page_scores = [E.evaluate_system('this-pipeline', sent, pt, norm=NORM)]\n"
        "print(); print(E.format_table(page_scores))\n"
        "print('\\nCER = with reading order; WordAcc = order-free recognition.')"
    ),
    md(
        "### 11b. Synthetic IAM pages — the reading-order stress test\n"
        "Real IAM lines stacked into skewed, indented pages. The skew makes naive top-to-bottom\n"
        "ordering fail, so a low CER here means reading order survived; WordAcc should stay high\n"
        "throughout (recognition is unaffected by the stacking)."
    ),
    code(
        "import matplotlib.pyplot as plt\n"
        "pages = E.build_iam_pages(n_pages=20, lines_per_page=(4, 8), max_skew_deg=3.0, seed=0)\n"
        "print(f'built {len(pages)} synthetic pages')\n"
        "plt.figure(figsize=(10, 6)); plt.imshow(pages[0].image_rgb())\n"
        "plt.axis('off'); plt.title(pages[0].name + '  (GT lines: ' + str(pages[0].gt.count(chr(10)) + 1) + ')')\n"
        "plt.show()\n"
        "\n"
        "pt, _ = E.pipeline_predictors(engine)\n"
        "syn_scores = [E.evaluate_system('this-pipeline', pages, pt, norm=NORM)]\n"
        "print(); print(E.format_table(syn_scores))\n"
        "print('\\nlow CER = reading order survives the skew | high WordAcc = recognition fine')"
    ),
    md(
        "## 12. GNHK head-to-head — the resume benchmark\n"
        "Real phone-photo English handwriting, **out-of-domain** for TrOCR, so this CER is unbiased —\n"
        "the honest number to put on a resume. Your pipeline runs against **Tesseract** and the full\n"
        "**PaddleOCR** on the *same* pages under one metric.\n"
        "\n"
        "**One-time download (manual — GNHK is behind a CC-BY-4.0 click-through, no direct link):**\n"
        "1. Open <https://goodnotes.com/gnhk>, scroll down, accept the terms, download `train_data.zip`\n"
        "   and/or `test_data.zip`.\n"
        "2. Upload the zip(s) to Colab (📁 sidebar → into `/content`), **or** drop them in Google Drive\n"
        "   and `from google.colab import drive; drive.mount('/content/drive')`.\n"
        "3. Run the cells below — they unzip, rebuild reading-order GT, sanity-check it, then score."
    ),
    code(
        "# install Tesseract (PaddleOCR already present from section 1)\n"
        "!apt-get -qq install -y tesseract-ocr >/dev/null && pip -q install pytesseract\n"
        "\n"
        "import glob, zipfile, pathlib, evaluate as E\n"
        "NORM = E.NormCfg(lowercase=False, strip_punct=False)\n"
        "GNHK_DIR = '/content/gnhk'\n"
        "\n"
        "# unzip whatever GNHK zip you uploaded (archive.zip, train_data.zip, ...)\n"
        "for z in glob.glob('/content/*.zip'):\n"
        "    with zipfile.ZipFile(z) as zf: zf.extractall(GNHK_DIR)\n"
        "    print('unzipped', z)\n"
        "\n"
        "# show what we actually got, so any layout problem is obvious\n"
        "root = pathlib.Path(GNHK_DIR)\n"
        "jsons = list(root.rglob('*.json'))\n"
        "imgs = [p for p in root.rglob('*') if p.suffix.lower() in E.IMAGE_EXTS]\n"
        "print(f'found {len(jsons)} json + {len(imgs)} image files under {GNHK_DIR}')\n"
        "if jsons: print('  e.g. json :', jsons[0].relative_to(root))\n"
        "if imgs:  print('  e.g. image:', imgs[0].relative_to(root))\n"
        "\n"
        "gnhk = E.load_gnhk(GNHK_DIR)\n"
        "print(f'{len(gnhk)} GNHK pages loaded')\n"
        "assert gnhk, ('0 pages. If counts above are >0, image+json do not share a folder/stem; "
        "else the zip had no GNHK .json files.')"
    ),
    md(
        "### 12a. Sanity-check the reconstructed ground truth\n"
        "GNHK GT is rebuilt from per-word `line_idx`; eyeball one page so you trust the reference text\n"
        "before scoring against it."
    ),
    code(
        "import matplotlib.pyplot as plt\n"
        "s = gnhk[0]\n"
        "plt.figure(figsize=(11, 8)); plt.imshow(s.image_rgb()); plt.axis('off'); plt.title(s.name); plt.show()\n"
        "print('--- reconstructed reading-order ground truth ---')\n"
        "print(s.gt)"
    ),
    md(
        "### 12b. Score: your pipeline vs Tesseract vs PaddleOCR\n"
        "Evaluates on the official **test** split (the standard set; scoring train+test would be hours).\n"
        "**CER/WER** are the headline comparison; **WordAcc** (order-free) tells you whether errors are\n"
        "recognition or ordering. Set `GNHK_N` to an int to cap for a quick pass; `None` runs the full split."
    ),
    code(
        "import pathlib, evaluate as E\n"
        "NORM = E.NormCfg(lowercase=False, strip_punct=False)\n"
        "\n"
        "# use the official TEST split if present under GNHK_DIR\n"
        "test_dirs = [p for p in pathlib.Path(GNHK_DIR).rglob('*')\n"
        "             if p.is_dir() and 'test' in p.name.lower()]\n"
        "EVAL_ROOT = str(test_dirs[0]) if test_dirs else GNHK_DIR\n"
        "eval_pages = E.load_gnhk(EVAL_ROOT)\n"
        "\n"
        "GNHK_N = None                       # None = all pages in EVAL_ROOT; set an int to cap\n"
        "subset = eval_pages[:GNHK_N] if GNHK_N else eval_pages\n"
        "print(f'evaluating on {EVAL_ROOT}: {len(subset)} pages '\n"
        "      f'(~{len(subset) * 14 / 60:.0f} min for the pipeline alone on a T4)')\n"
        "\n"
        "pt, _ = E.pipeline_predictors(engine)               # full detect -> order -> recognize\n"
        "gnhk_scores = [E.evaluate_system('this-pipeline', subset, pt, norm=NORM)]\n"
        "print('scored this-pipeline')\n"
        "for b in ['tesseract', 'paddleocr']:                # your chosen competitors\n"
        "    try:\n"
        "        gnhk_scores.append(E.evaluate_system(b, subset, E.baseline_predict(b), norm=NORM))\n"
        "        print('scored', b)\n"
        "    except Exception as ex:\n"
        "        import traceback\n"
        "        print('skip', b, ':', type(ex).__name__, ex); traceback.print_exc()\n"
        "\n"
        "print(); print(E.format_table(gnhk_scores))\n"
        "E.save_csv(gnhk_scores, 'gnhk_results.csv')\n"
        "print('\\nper-image breakdown -> gnhk_results.csv')"
    ),
    code(
        "# resume chart: CER/WER per system on GNHK\n"
        "import numpy as np, matplotlib.pyplot as plt\n"
        "names = [s.system for s in gnhk_scores]\n"
        "x = np.arange(len(names))\n"
        "fig, ax = plt.subplots(figsize=(1.7 * len(names) + 3, 4))\n"
        "ax.bar(x - 0.2, [s.cer for s in gnhk_scores], 0.4, label='CER')\n"
        "ax.bar(x + 0.2, [s.wer for s in gnhk_scores], 0.4, label='WER')\n"
        "for i, s in enumerate(gnhk_scores):\n"
        "    ax.text(i - 0.2, s.cer, f'{s.cer:.0%}', ha='center', va='bottom', fontsize=8)\n"
        "ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right')\n"
        "ax.set_ylabel('error rate (lower is better)'); ax.legend()\n"
        "ax.set_title('GNHK: full-page handwriting OCR comparison'); ax.grid(axis='y', alpha=0.3)\n"
        "plt.tight_layout(); plt.savefig('gnhk_comparison.png', dpi=130); plt.show()\n"
        "print('saved gnhk_comparison.png — drop this straight into your resume/README')"
    ),
    md(
        "### Any other dataset\n"
        "`E.load_hf('org/dataset', n=100, image_col=..., text_col=...)` works for any hub dataset with\n"
        "an image + text column. Score lines with `E.evaluate_recognizer`, pages with the\n"
        "`pipeline_predictors` + `evaluate_system` pattern above. Only compare systems on the **same** set."
    ),
    md(
        "## Debugging guide\n"
        "If something looks wrong, report back with the panel image + recognized text and note which case it is:\n"
        "\n"
        "1. **Boxes miss/split handwriting** → detection problem (we tune `unclip_ratio`/`box_thresh` or det model).\n"
        "2. **Boxes fine, numbers in wrong sequence** → ordering problem (send the panel; the geometry is testable, we add your case as a unit test).\n"
        "3. **Boxes and order fine, words wrong** → recognition problem (TrOCR domain gap → fine-tuning / beams). **Low WordAcc** confirms this; **high WordAcc with high CER** means ordering instead.\n"
        "4. **Page rotated 90°/180°** → known limitation, orientation classifier is on the roadmap.\n"
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

target = ROOT / "colab_ocr_debug.ipynb"
target.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {target} ({target.stat().st_size:,} bytes, {len(cells)} cells)")
