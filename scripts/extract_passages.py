"""Step 3: turn a saved DailyMed label into inspected, section-aware passages.

Standard library only.

Usage (from the project root):
  python scripts/extract_passages.py <document_id>

Needs:
  data/manifest/documents.json   (written by fetch_label.py)
  data/manifest/curation.json    (written by hand: label_id, ingredient, product_title, ...)
Writes:
  data/passages/<label_id>.jsonl         the passages
  data/passages/review_<label_id>.md     a readable file to inspect against the real label
  and adds sections_found / sections_missing to the manifest entry.
"""
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest" / "documents.json"
CURATION = ROOT / "data" / "manifest" / "curation.json"
PASSAGES_DIR = ROOT / "data" / "passages"

H = "{urn:hl7-org:v3}"
NS = {"h": "urn:hl7-org:v3"}

EXTRACTOR_VERSION = "extract-v1"
MAX_WORDS = 300          # soft cap per passage
MIN_BODY_CHARS = 15      # sections with less own text are treated as header-only
REQUIRED_CURATION = ["label_id", "ingredient", "product_title", "labeler", "formulation", "route"]

# Target section types. A section gets its type from its LOINC code (and passes it down to
# its subsections) or, for renal impairment, from its title. One section = one type.
LOINC_TARGETS = {
    "34070-3": "contraindications",
    "43685-7": "warnings_precautions",
    "34073-7": "drug_interactions",
}
RENAL_TITLE = re.compile(r"renal impairment", re.IGNORECASE)
ALL_TYPES = ["contraindications", "warnings_precautions", "drug_interactions", "renal_impairment"]


# ---------- small helpers ----------
def tag(el):
    return el.tag.replace(H, "") if isinstance(el.tag, str) else ""


SPACED_TAGS = {"br", "paragraph", "item", "list", "table", "tr", "td", "th", "caption"}


def _text_of(el):
    parts = [el.text or ""]
    for c in el:
        spaced = tag(c) in SPACED_TAGS
        parts.append(" " if spaced else "")
        parts.append(_text_of(c))
        parts.append(" " if spaced else "")
        parts.append(c.tail or "")
    return "".join(parts)


def clean(el):
    """All text under an element with whitespace collapsed (formatting only, meaning unchanged).
    A space is inserted at line breaks and block boundaries so words are never glued together."""
    return " ".join(_text_of(el).split()) if el is not None else ""


def words(s):
    return len(s.split())


def title_of(sec):
    t = clean(sec.find("h:title", NS))
    if t:
        return t
    code = sec.find("h:code", NS)
    return (code.get("displayName") if code is not None else None) or "(untitled)"


def section_number(title):
    m = re.match(r"^(\d+(?:\.\d+)*)\b", title)
    if m:
        return m.group(1)
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:30] or "section"


# ---------- turning SPL text into blocks ----------
def row_text(tr):
    cells = [clean(c) for c in tr if tag(c) in ("th", "td")]
    return " | ".join(cells) if any(cells) else ""


def render_blocks(el, flags):
    blocks = []
    for child in el:
        t = tag(child)
        if t == "paragraph":
            s = clean(child)
            if s:
                blocks.append({"kind": "p", "text": s})
        elif t == "list":
            items = ["- " + clean(i) for i in child.findall("h:item", NS) if clean(i)]
            if items:
                blocks.append({"kind": "lines", "head": [], "lines": items, "foot": []})
        elif t == "table":
            flags.add("has_table")
            caption = clean(child.find("h:caption", NS))

            def rows_of(container):
                return [r for r in (row_text(tr) for tr in container.findall("h:tr", NS)) if r]

            thead = child.find("h:thead", NS)
            tfoot = child.find("h:tfoot", NS)
            head_rows = rows_of(thead) if thead is not None else []
            foot_rows = rows_of(tfoot) if tfoot is not None else []
            body_rows = rows_of(child)                       # rows directly under <table>
            for tb in child.findall("h:tbody", NS):
                body_rows += rows_of(tb)
            head = ([caption] if caption else []) + head_rows
            if head or body_rows or foot_rows:
                blocks.append({"kind": "lines", "head": head, "lines": body_rows, "foot": foot_rows})
        elif t == "renderMultiMedia":
            flags.add("has_image")
        else:
            blocks.extend(render_blocks(child, flags))
    if not blocks:
        s = clean(el)
        if s:
            blocks.append({"kind": "p", "text": s})
    return blocks


def split_block(block, flags):
    if block["kind"] == "p":
        text = block["text"]
        if words(text) <= MAX_WORDS:
            return [text]
        flags.add("split_long_paragraph")
        pieces, cur = [], []
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text):
            if cur and words(" ".join(cur + [s])) > MAX_WORDS:
                pieces.append(" ".join(cur))
                cur = []
            cur.append(s)
        if cur:
            pieces.append(" ".join(cur))
        return pieces
    head, lines, foot = block["head"], block["lines"], block["foot"]
    full = "\n".join(head + lines + foot)
    if words(full) <= MAX_WORDS:
        return [full]
    flags.add("split_table_or_list")      # header and footnote rows are repeated in every piece
    pieces, cur = [], []
    base = words(" ".join(head + foot))
    cur_words = base
    for line in lines:
        w = words(line)
        if cur and cur_words + w > MAX_WORDS:
            pieces.append("\n".join(head + cur + foot))
            cur, cur_words = [], base
        cur.append(line)
        cur_words += w
    if cur:
        pieces.append("\n".join(head + cur + foot))
    return pieces


def pack(pieces):
    out, cur = [], []
    for p in pieces:
        if cur and words("\n".join(cur + [p])) > MAX_WORDS:
            out.append("\n".join(cur))
            cur = []
        cur.append(p)
    if cur:
        out.append("\n".join(cur))
    return out


def text_flags(text):
    f = set()
    if "\ufffd" in text:
        f.add("replacement_char")
    if re.search(r"</?[A-Za-z][^>]*>", text) or re.search(r"&(amp|lt|gt|nbsp);", text):
        f.add("possible_markup_residue")
    return f


# ---------- finding the target sections ----------
def collect(sec, titles, inherited, out):
    code = sec.find("h:code", NS)
    loinc = code.get("code") if code is not None else None
    stype = inherited or LOINC_TARGETS.get(loinc)
    if stype is None and RENAL_TITLE.search(titles[-1]):
        stype = "renal_impairment"
    if stype:
        out.append({"sec": sec, "titles": list(titles), "loinc": loinc, "type": stype})
    for child in sec.findall("h:component/h:section", NS):
        collect(child, titles + [title_of(child)], stype, out)


# ---------- main ----------
def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 1
    doc_id = argv[1]

    if not MANIFEST.exists():
        print("No manifest found. Run fetch_label.py first.")
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if doc_id not in manifest:
        print(f"Unknown document_id. Known: {list(manifest)}")
        return 2
    entry = manifest[doc_id]

    if not CURATION.exists():
        print(f"Missing {CURATION.relative_to(ROOT)}. Create it (see the instructions) and run again.")
        return 2
    curation = json.loads(CURATION.read_text(encoding="utf-8"))
    cur = curation.get(doc_id)
    if not cur or any(not cur.get(k) for k in REQUIRED_CURATION):
        print(f"curation.json needs an entry for {doc_id} with all of: {REQUIRED_CURATION}")
        return 2
    label_id = cur["label_id"]

    raw_path = ROOT / Path(entry["raw_file"].replace("\\", "/"))
    raw = raw_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["source_hash"]:
        print("HASH MISMATCH: the saved XML no longer matches the manifest. Do not continue.")
        return 2
    root = ET.fromstring(raw)

    body = root.find("h:component/h:structuredBody", NS)
    if body is None:
        print("No structuredBody in this document.")
        return 2
    items = []
    for top in body.findall("h:component/h:section", NS):
        collect(top, [title_of(top)], None, items)

    passages, found, header_only, used_ids = [], {}, [], set()
    for item in items:
        sec, titles = item["sec"], item["titles"]
        text_el = sec.find("h:text", NS)
        own = clean(text_el)
        path = " > ".join(titles)
        if len(own) < MIN_BODY_CHARS:
            header_only.append({"section_path": path, "type": item["type"]})
            continue
        flags = set()
        if text_el is not None and any(True for _ in text_el.iter(H + "renderMultiMedia")):
            flags.add("has_image")
        pieces = []
        for block in render_blocks(text_el, flags):
            pieces.extend(split_block(block, flags))
        texts = pack(pieces)
        num = section_number(titles[-1])
        made = []
        for i, t in enumerate(texts, start=1):
            pid = f"{label_id}-{num}-p{i:02d}"
            if pid in used_ids:
                pid = pid + "-dup"
            used_ids.add(pid)
            p = {
                "passage_id": pid,
                "document_id": doc_id,
                "label_id": label_id,
                "section_type": item["type"],
                "section_title": titles[-1],
                "section_path": path,
                "loinc": item["loinc"],
                "text": t,
                "word_count": words(t),
                "flags": sorted(flags | text_flags(t)),
            }
            made.append(p)
        passages.extend(made)
        extracted_chars = sum(len(" ".join(p["text"].split())) for p in made)
        found.setdefault(item["type"], []).append({
            "section_path": path,
            "loinc": item["loinc"],
            "passages": len(made),
            "source_chars": len(own),
            "extracted_chars": extracted_chars,
        })

    missing = [t for t in ALL_TYPES if t not in found]

    # ---- write outputs ----
    PASSAGES_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = PASSAGES_DIR / f"{label_id}.jsonl"
    jsonl.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in passages) + "\n", encoding="utf-8")

    review = PASSAGES_DIR / f"review_{label_id}.md"
    lines = [f"# Extraction review: {label_id} - {cur['product_title']}",
             "",
             f"Source file: `{entry['raw_file']}`  (hash verified)",
             f"Set ID: {entry.get('set_id')}  |  SPL version: {entry.get('spl_version')}  |  effective date: {entry.get('effective_date')}",
             "",
             "## Target sections",
             ""]
    for t in ALL_TYPES:
        if t in found:
            for f in found[t]:
                lines.append(f"- **{t}**: FOUND at `{f['section_path']}` ({f['passages']} passage(s))")
        else:
            lines.append(f"- **{t}**: NOT_FOUND")
    if header_only:
        lines += ["", "Header-only sections (no own text, skipped): " + "; ".join(h["section_path"] for h in header_only)]
    lines += ["", "## Passages", ""]
    for p in passages:
        lines += [f"### {p['passage_id']} | {p['section_type']} | {p['word_count']} words | flags: {', '.join(p['flags']) or 'none'}",
                  f"Path: {p['section_path']}",
                  "",
                  "```text",
                  p["text"],
                  "```",
                  ""]
    review.write_text("\n".join(lines), encoding="utf-8")

    if "xml_title_text" not in entry:
        entry["xml_title_text"] = entry.get("product_title")
    for k in REQUIRED_CURATION:
        entry[k] = cur[k]
    entry["extraction"] = {
        "extractor_version": EXTRACTOR_VERSION,
        "max_passage_words": MAX_WORDS,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "n_passages": len(passages),
        "sections_found": found,
        "sections_missing": missing,
        "header_only_skipped": [h["section_path"] for h in header_only],
    }
    manifest[doc_id] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ---- console summary ----
    print(f"{len(passages)} passages written to {jsonl.relative_to(ROOT)}")
    print(f"Review file: {review.relative_to(ROOT)}\n")
    for t in ALL_TYPES:
        if t in found:
            for f in found[t]:
                ratio = f["extracted_chars"] / f["source_chars"] if f["source_chars"] else 0
                note = "" if 0.9 <= ratio <= 1.3 else "   <-- INSPECT (text length differs from source)"
                print(f"  {t:22s} FOUND  {f['section_path']}  [{f['passages']} passage(s), ratio {ratio:.2f}]{note}")
        else:
            print(f"  {t:22s} NOT_FOUND")
    flagged = [p for p in passages if p["flags"]]
    print(f"\nPassages with flags: {len(flagged)}")
    for p in flagged:
        print(f"  {p['passage_id']}: {', '.join(p['flags'])}")
    print("\nNext: open the review file and compare it with the real label.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))