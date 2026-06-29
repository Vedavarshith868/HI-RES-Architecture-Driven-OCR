"""Heuristic table rendering from token geometry (no ML, pure Python).

Given the recognized tokens per reading-order line — each token carrying its
horizontal extent (x0, x1) — find the contiguous blocks of lines that form a
column-aligned table and render those as Markdown tables. Everything else
(headings, prose, notes) passes through as plain text, so a page that mixes a
table with ordinary text renders both correctly.

Approach (block-local, so prose elsewhere on the page cannot hide the table):

  1. mark each row "structured" if it has an internal gap wider than a column
     threshold (a table row),
  2. group consecutive structured rows into runs,
  3. for each run, find the column separators from that run only, and render it
     as a table when it has >= 2 columns and >= 2 rows.

The column threshold is keyed to text *height*, not width: an inter-column gap
is roughly a line-height or more, while an inter-word space is ~0.3x of it.
Width is a poor yardstick because a cell may hold a wide phrase.

token_lines: list (in reading order) of rows; each row is a list of
``(text, x0, x1)`` tuples. ``line_height`` is the median text height in px.
"""
from __future__ import annotations

from statistics import median

GAP_FACTOR_W = 1.5    # fallback gap threshold (x median token width) if no height
GAP_FACTOR_H = 0.7    # gap threshold as a fraction of median text height
MIN_ROWS = 2
MIN_COLS = 2


def _md_line(s: str) -> str:
    s = s.replace("|", "\\|")
    if s.lstrip().startswith("#"):
        s = s.replace("#", "\\#", 1)
    return s


def _md_text(s: str) -> str:
    """Plain text -> Markdown: keep line breaks, avoid accidental formatting."""
    return "  \n".join(_md_line(line) for line in s.split("\n"))


def _has_wide_gap(row, min_gap) -> bool:
    return any(b[1] - a[2] >= min_gap for a, b in zip(row, row[1:]))


def _column_separators(run, min_gap):
    """x-positions of whitespace channels that survive across the run's rows."""
    intervals = sorted((x0, x1) for r in run for (_, x0, x1) in r)
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for x0, x1 in intervals[1:]:
        if x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    return [(a[1] + b[0]) / 2.0 for a, b in zip(merged, merged[1:])
            if b[0] - a[1] >= min_gap]


def _col_of(x, seps):
    j = 0
    for s in seps:
        if x > s:
            j += 1
        else:
            break
    return j


def _row_text(row) -> str:
    return " ".join(t for (t, _, _) in row if t).strip()


def _grid_to_md(grid: list[list[str]]) -> str:
    ncol = max(len(r) for r in grid)
    grid = [r + [""] * (ncol - len(r)) for r in grid]
    head = "| " + " | ".join(_md_line(c).strip() for c in grid[0]) + " |"
    rule = "| " + " | ".join("---" for _ in range(ncol)) + " |"
    body = ["| " + " | ".join(_md_line(c).strip() for c in r) + " |"
            for r in grid[1:]]
    return "\n".join([head, rule, *body])


def _build_grid(run, seps):
    ncol = len(seps) + 1
    grid = []
    for r in run:
        cells = [""] * ncol
        for (t, x0, x1) in r:
            k = _col_of((x0 + x1) / 2, seps)
            cells[k] = (cells[k] + " " + t).strip()
        grid.append(cells)
    return grid


def _grid_md(grid) -> str:
    ncol = len(grid[0])
    keep = [c for c in range(ncol) if any(g[c] for g in grid)]
    return _grid_to_md([[g[c] for c in keep] for g in grid])


def layout_to_markdown(token_lines, plain_text, *, line_height=None,
                       min_rows=MIN_ROWS, min_cols=MIN_COLS):
    """Render token_lines as Markdown, turning aligned blocks into tables.

    Falls back to ``plain_text`` when nothing is confidently tabular.
    """
    rows = [sorted(r, key=lambda z: z[1]) for r in token_lines if r]
    if len(rows) < min_rows:
        return _md_text(plain_text)

    if line_height and line_height > 0:
        min_gap = max(GAP_FACTOR_H * line_height, 6.0)
    else:
        widths = [x1 - x0 for r in rows for (_, x0, x1) in r if x1 > x0]
        if not widths:
            return _md_text(plain_text)
        min_gap = max(GAP_FACTOR_W * median(widths), 1.0)

    structured = [_has_wide_gap(r, min_gap) for r in rows]

    segments: list[str] = []
    buf: list[str] = []
    found_table = False

    def flush_buf():
        if buf:
            segments.append("  \n".join(_md_line(x) for x in buf if x.strip()))
            buf.clear()

    i, n = 0, len(rows)
    while i < n:
        if structured[i]:
            j = i
            while j < n and structured[j]:
                j += 1
            run = rows[i:j]
            seps = _column_separators(run, min_gap) if len(run) >= min_rows else []
            if len(run) >= min_rows and len(seps) + 1 >= min_cols:
                grid = _build_grid(run, seps)
                # absorb up to 2 following short non-table lines that are a cell
                # which wrapped to a new physical line (their x lands inside the
                # table's column span) into the last row's matching column
                t_lo = min(x0 for r in run for (_, x0, _) in r)
                t_hi = max(x1 for r in run for (_, _, x1) in r)
                k, absorbed = j, 0
                while (k < n and not structured[k] and absorbed < 2
                       and 1 <= len(rows[k]) <= 2
                       and all(t_lo <= (x0 + x1) / 2 <= t_hi
                               for (_, x0, x1) in rows[k])):
                    for (t, x0, x1) in rows[k]:
                        c = min(_col_of((x0 + x1) / 2, seps), len(grid[-1]) - 1)
                        grid[-1][c] = (grid[-1][c] + " " + t).strip()
                    k += 1
                    absorbed += 1
                flush_buf()
                segments.append(_grid_md(grid))
                found_table = True
                i = k
            else:
                buf.extend(_row_text(r) for r in run)
                i = j
        else:
            buf.append(_row_text(rows[i]))
            i += 1
    flush_buf()

    if not found_table:
        return _md_text(plain_text)
    return "\n\n".join(s for s in segments if s.strip())


# --------------------------------------------------------------------------
if __name__ == "__main__":  # quick self-test (no deps)
    H = 24

    def tok(text, x0, w=60):
        return (text, float(x0), float(x0 + w))

    table = [
        [tok("Name", 0), tok("Age", 200), tok("City", 400)],
        [tok("Alice", 0), tok("30", 200), tok("Paris", 400)],
        [tok("Bob", 0), tok("25", 200), tok("Rome", 400)],
    ]
    print("=== TABLE (expect grid) ===")
    print(layout_to_markdown(table, "PLAIN_FALLBACK", line_height=H))

    prose = [
        [tok("the", 0), tok("quick", 70), tok("brown", 150), tok("fox", 250)],
        [tok("jumps", 0), tok("over", 90), tok("a", 170), tok("lazy", 210)],
        [tok("dog", 0), tok("near", 80), tok("the", 180), tok("river", 240)],
    ]
    print("\n=== PROSE (expect PLAIN_FALLBACK) ===")
    print(layout_to_markdown(prose, "PLAIN_FALLBACK", line_height=H))

    invoice = [
        [tok("INVOICE", 0, 200)],
        [tok("Mytreyi,", 0, 90), tok("Mandala", 95, 80), tok("Artist", 180, 70)],
        [tok("Aesthetic", 0, 100), tok("Indian", 105, 70), tok("Crafts", 180, 70)],
        [tok("Ph No: 9963467064", 0, 220), tok("Invoice No: 0146", 700, 200)],
        [tok("Date: 13th June 2026", 700, 200)],
        [tok("Billed to", 0, 100)],
        [tok("Taneira,", 0, 90), tok("jubilee", 95, 70), tok("hills", 170, 50)],
        [tok("Item Details", 0, 150), tok("Qty.", 320, 60),
         tok("Unit Price", 520, 130), tok("Total", 780, 90)],
        [tok("Workshop: Mandala Art", 0, 230), tok("16", 320, 40),
         tok("300/-", 520, 70), tok("4800/-", 780, 90)],
        [tok("Time: 4pm-6pm", 0, 160), tok("(No. of", 320, 90),
         tok("(Per head)", 520, 110)],
        [tok("participants)", 320, 110)],
        [tok("Thank", 0, 70), tok("you", 75, 50), tok("for", 130, 40),
         tok("hosting", 175, 90), tok("this", 270, 50), tok("workshop.", 325, 120)],
    ]
    print("\n=== INVOICE (expect prose lines + a grid for the table) ===")
    print(layout_to_markdown(invoice, "PLAIN_FALLBACK", line_height=H))
