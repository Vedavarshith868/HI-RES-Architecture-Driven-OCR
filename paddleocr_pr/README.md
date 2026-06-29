# Upstream contribution to PaddleOCR — PR #18189

**[PaddlePaddle/PaddleOCR#18189](https://github.com/PaddlePaddle/PaddleOCR/pull/18189)**

While building HI-RES I found a reading-order bug in PaddleOCR (★84k): its
`sorted_boxes` routine groups detected boxes into lines with a **hardcoded 10 px
row threshold**. That assumes an upright page — under mild, real-world skew a
single text line spans far more than 10 px in y (a 1000 px-wide line at 3°
already spans ~52 px), so boxes from one line are split across several "rows"
and the reading order scrambles.

**The fix** ([`sorted_boxes_skew.py`](sorted_boxes_skew.py)) estimates the
dominant skew from the detected boxes, sorts in the deskewed frame, and uses a
row tolerance proportional to text height. On an upright page it reduces to the
original ordering — backward compatible — and only changes behaviour on skewed
input. Pure NumPy, single function, unit-tested.

**Evidence (isolated):** running PP-OCRv6 detection + recognition once and only
swapping the ordering, on the full **7-language XFUND** validation split, the
stock ordering inflates **CER by +10–20% at ~10° skew across every language**,
while the skew-robust ordering stays flat. Full benchmark, figure, and unit
test are in the PR.
