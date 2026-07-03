"""Build paper_KCI.hwpx (single-column) and paper_KCI_2col.hwpx (journal 2-column)
from paper_draft_KCI.md using python-hwpx."""
import os, re, sys
sys.path.insert(0, "/data3/bhkim/workspace/train-ego-path-detection")
os.chdir("/data3/bhkim/workspace/train-ego-path-detection")
from hwpx import HwpxDocument

MD = open("paper_draft_KCI.md", encoding="utf-8").read()


def strip_inline(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    return s.strip()


def parse_blocks(md):
    """Markdown -> list of (kind, payload). kinds: h1,h2,h3,p,quote,bullet,table,hr"""
    blocks, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1; continue
        if s.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].strip()) <= set("|-: "):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [strip_inline(c) for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            rows.pop(1)  # separator row
            blocks.append(("table", rows)); continue
        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)$", s)
        if m:
            blocks.append(("img", m.group(1))); i += 1; continue
        if s == "---":
            blocks.append(("hr", None))
        elif s.startswith("### "):
            blocks.append(("h3", strip_inline(s[4:])))
        elif s.startswith("## "):
            blocks.append(("h2", strip_inline(s[3:])))
        elif s.startswith("# "):
            blocks.append(("h1", strip_inline(s[2:])))
        elif s.startswith("> "):
            blocks.append(("quote", strip_inline(s[2:])))
        elif s.startswith("- "):
            blocks.append(("bullet", strip_inline(s[2:])))
        else:
            blocks.append(("p", strip_inline(s)))
        i += 1
    return blocks


BLOCKS = parse_blocks(MD)
print(f"{len(BLOCKS)} blocks; tables={sum(1 for k,_ in BLOCKS if k=='table')}")


def fill_table(doc, tbl, rows):
    """Set cell texts of a python-hwpx table (introspect the cell API once)."""
    trs = tbl.rows if hasattr(tbl, "rows") else None
    for r, row in enumerate(rows):
        cells = trs[r].cells if trs is not None else None
        for c, text in enumerate(row):
            cell = cells[c]
            if hasattr(cell, "text"):
                try:
                    cell.text = text; continue
                except Exception:
                    pass
            if hasattr(cell, "set_text"):
                cell.set_text(text); continue
            raise RuntimeError(f"no cell text API: {dir(cell)[:20]}")


def build(two_col, out_path):
    doc = HwpxDocument.new()
    doc.set_page_setup(paper_size="A4")
    doc.set_page_margins(left=8500, right=8500, top=9000, bottom=9000)  # HWPUNIT

    body_size = 9.5 if two_col else 10.5
    st_title = doc.ensure_run_style(bold=True, size=16)
    st_subtitle = doc.ensure_run_style(bold=True, size=11)
    st_h2 = doc.ensure_run_style(bold=True, size=13)
    st_h3 = doc.ensure_run_style(bold=True, size=11.5)
    st_body = doc.ensure_run_style(size=body_size)
    st_small = doc.ensure_run_style(size=8.5, color="#555555")

    started_cols = False
    for kind, payload in BLOCKS:
        # journal layout: switch to 2 columns starting at "1. 서론"
        if two_col and not started_cols and kind == "h2" and payload.startswith("1. 서론"):
            p = doc.add_paragraph("")
            doc.set_columns(2, paragraph=p, same_gap=850)
            started_cols = True
        if kind == "h1":
            doc.add_paragraph(payload, char_pr_id_ref=st_title)
        elif kind == "h2":
            doc.add_paragraph(payload, char_pr_id_ref=st_h2)
        elif kind == "h3":
            doc.add_paragraph(payload, char_pr_id_ref=st_h3)
        elif kind == "quote":
            doc.add_paragraph("※ " + payload, char_pr_id_ref=st_small)
        elif kind == "bullet":
            doc.add_paragraph("• " + payload, char_pr_id_ref=st_body)
        elif kind == "hr":
            doc.add_paragraph("")
        elif kind == "p":
            # the bold English subtitle right under the title
            style = st_subtitle if payload.startswith("Implementation of") else st_body
            doc.add_paragraph(payload, char_pr_id_ref=style)
        elif kind == "img":
            if os.path.exists(payload):
                data = open(payload, "rb").read()
                fmt = os.path.splitext(payload)[1].lstrip(".").lower() or "png"
                w = 78.0 if two_col else 150.0   # fit the column / page width (mm)
                doc.add_picture(data, fmt, width_mm=w)
                doc.add_paragraph("")
            else:
                doc.add_paragraph(f"[그림 파일 없음: {payload}]", char_pr_id_ref=st_small)
        elif kind == "table":
            rows = payload
            tbl = doc.add_table(len(rows), len(rows[0]))
            fill_table(doc, tbl, rows)
            doc.add_paragraph("")
    doc.save_to_path(out_path)
    print("saved", out_path)
    # verify: reopen and extract text
    chk = HwpxDocument.open(out_path)
    txt = chk.export_text()
    print(f"  reopened OK, {len(txt)} chars, contains 초록: {'국문 초록' in txt}, 표4: {'표 4' in txt}")


build(False, "paper_KCI.hwpx")
build(True, "paper_KCI_2col.hwpx")
print("DONE")
