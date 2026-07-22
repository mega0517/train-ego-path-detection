"""Build paper_conf_IEIE.hwpx from paper_conf_IEIE.md using python-hwpx.

Adapted from make_hwpx.py for the IEIE conference summary paper: single column,
roman-numeral section headings (Ⅰ.~Ⅳ.), and the English subtitle line.
"""
import os, re, sys
sys.path.insert(0, "/data3/bhkim/workspace/train-ego-path-detection")
os.chdir("/data3/bhkim/workspace/train-ego-path-detection")
from hwpx import HwpxDocument

SRC = "paper_conf_IEIE.md"
OUT = "paper_conf_IEIE.hwpx"
MD = open(SRC, encoding="utf-8").read()


def strip_inline(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    return s.strip()


def parse_blocks(md):
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
            rows.pop(1)
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
print(f"{len(BLOCKS)} blocks; tables={sum(1 for k,_ in BLOCKS if k=='table')}, "
      f"imgs={sum(1 for k,_ in BLOCKS if k=='img')}")


def fill_table(doc, tbl, rows):
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


def build(out_path):
    doc = HwpxDocument.new()
    doc.set_page_setup(paper_size="A4")
    doc.set_page_margins(left=8500, right=8500, top=9000, bottom=9000)

    st_title = doc.ensure_run_style(bold=True, size=16)
    st_subtitle = doc.ensure_run_style(bold=True, size=11)
    st_h2 = doc.ensure_run_style(bold=True, size=13)
    st_body = doc.ensure_run_style(size=10.5)
    st_small = doc.ensure_run_style(size=8.5, color="#555555")
    # figure captions and the bold "그림 N." labels read as small centered notes
    st_cap = doc.ensure_run_style(size=9, color="#333333")

    for kind, payload in BLOCKS:
        if kind == "h1":
            doc.add_paragraph(payload, char_pr_id_ref=st_title)
        elif kind == "h2":
            doc.add_paragraph(payload, char_pr_id_ref=st_h2)
        elif kind == "h3":
            doc.add_paragraph(payload, char_pr_id_ref=st_h2)
        elif kind == "quote":
            doc.add_paragraph("※ " + payload, char_pr_id_ref=st_small)
        elif kind == "bullet":
            doc.add_paragraph("• " + payload, char_pr_id_ref=st_body)
        elif kind == "hr":
            doc.add_paragraph("")
        elif kind == "p":
            # bold English subtitle right under the title; figure-caption lines small
            if payload.startswith("Semi-automatic"):
                style = st_subtitle
            elif payload.startswith("그림 "):
                style = st_cap
            else:
                style = st_body
            doc.add_paragraph(payload, char_pr_id_ref=style)
        elif kind == "img":
            if os.path.exists(payload):
                data = open(payload, "rb").read()
                fmt = os.path.splitext(payload)[1].lstrip(".").lower() or "png"
                doc.add_picture(data, fmt, width_mm=140.0)
                doc.add_paragraph("")
            else:
                doc.add_paragraph(f"[그림 파일 없음: {payload}]", char_pr_id_ref=st_small)
        elif kind == "table":
            rows = payload
            tbl = doc.add_table(len(rows), len(rows[0]))
            fill_table(doc, tbl, rows)
            doc.add_paragraph("")

    import io as _io
    import zipfile as _zip
    raw = doc._to_bytes_raw(reset_dirty=False)
    hdr_xml = doc._root.headers[0].to_bytes()
    src = _zip.ZipFile(_io.BytesIO(raw))
    with _zip.ZipFile(out_path, "w", _zip.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            data = hdr_xml if info.filename == "Contents/header.xml" else src.read(info.filename)
            dst.writestr(info, data)
    print("saved", out_path)

    import re as _re
    z = _zip.ZipFile(out_path)
    hdr = z.read("Contents/header.xml").decode("utf-8")
    sec = z.read("Contents/section0.xml").decode("utf-8")
    used = set(_re.findall(r'charPrIDRef="(\d+)"', sec))
    defined = set(_re.findall(r"<hh:charPr\b[^>]*?\bid=\"(\d+)\"", hdr))
    missing = used - defined
    assert not missing, f"charPr missing in header: {missing}"
    chk = HwpxDocument.open(out_path)
    txt = chk.export_text()
    print(f"  reopened OK, {len(txt)} chars, charPr ok, "
          f"제목 있음: {'SAM 2' in txt}, 참고문헌 있음: {'참고문헌' in txt}, "
          f"부트스트래핑 수치: {'0.389' in txt}")


build(OUT)
print("DONE")
