"""Step 1: find and fetch ONE real DailyMed label, save it unchanged, print its section tree.

Standard library only (no pip installs needed).

Usage (run from the project root):
  python scripts/fetch_label.py search metformin
  python scripts/fetch_label.py fetch <SETID>
  python scripts/fetch_label.py tree data/raw/<file>.xml     # re-print tree from a saved file
"""
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
NS = {"h": "urn:hl7-org:v3"}
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest" / "documents.json"


def http_get(url: str) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "clinical-rag-academic-project/0.1"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _clean(el) -> str:
    """Collapse whitespace in all text under an element."""
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def search(drug_name: str, pagesize: int = 20) -> None:
    qs = urllib.parse.urlencode({"drug_name": drug_name, "pagesize": pagesize})
    data = json.loads(http_get(f"{BASE}/spls.json?{qs}"))
    for row in data.get("data", []):
        print(f'{row["setid"]}  v{row["spl_version"]}  {row["published_date"]}  {row["title"]}')
    total = data.get("metadata", {}).get("total_elements", "?")
    print(f"\n{total} labels match; showing up to {pagesize}.")
    print("Pick ONE explicit product label and run:  python scripts/fetch_label.py fetch <SETID>")


def parse_header(root) -> dict:
    def attr(path, name):
        node = root.find(path, NS)
        return node.get(name) if node is not None else None

    title_el = root.find("h:title", NS)
    return {
        "set_id": attr("h:setId", "root"),
        "spl_version": attr("h:versionNumber", "value"),
        "effective_date": attr("h:effectiveTime", "value"),  # raw string, not an approval date
        "xml_title_text": _clean(title_el) or None,
    }


def section_info(sec) -> dict:
    code = sec.find("h:code", NS)
    return {
        "loinc": code.get("code") if code is not None else None,
        "loinc_name": code.get("displayName") if code is not None else None,
        "title": _clean(sec.find("h:title", NS)),
        "own_chars": len(_clean(sec.find("h:text", NS))),  # this section's own text only
    }


def walk(sec, depth=0):
    yield depth, section_info(sec)
    for child in sec.findall("h:component/h:section", NS):
        yield from walk(child, depth + 1)


def print_tree(root) -> int:
    body = root.find("h:component/h:structuredBody", NS)
    if body is None:
        print("No structuredBody found in this document.")
        return 0
    count = 0
    for top in body.findall("h:component/h:section", NS):
        for depth, info in walk(top):
            count += 1
            label = info["title"] or info["loinc_name"] or "(untitled)"
            print(f'{"  " * depth}[{info["loinc"] or "-"}] {label}  ({info["own_chars"]} chars)')
    print(f"\n{count} sections total.")
    return count


def update_manifest(entry: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    docs = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    docs[entry["document_id"]] = entry
    MANIFEST.write_text(json.dumps(docs, indent=2), encoding="utf-8")


def fetch(setid: str) -> None:
    url = f"{BASE}/spls/{setid}.xml"
    raw = http_get(url)
    root = ET.fromstring(raw)
    head = parse_header(root)
    version = head["spl_version"] or "unknown"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{setid}_v{version}.xml"
    path.write_bytes(raw)  # original bytes, unmodified
    entry = {
        "document_id": f"{setid}_v{version}",
        **head,
        "source_url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_hash": hashlib.sha256(raw).hexdigest(),
        "raw_file": str(path.relative_to(ROOT)),
        # ingredient / formulation / route are filled during manual inspection, not guessed here
    }
    update_manifest(entry)
    print(f"Saved {path.relative_to(ROOT)}  ({len(raw)} bytes)")
    print(f"Header: {json.dumps(head)}")
    print(f"Hash: {entry['source_hash']}\n")
    print_tree(root)


def main(argv):
    try:
        if len(argv) >= 3 and argv[1] == "search":
            search(" ".join(argv[2:]))
        elif len(argv) == 3 and argv[1] == "fetch":
            fetch(argv[2])
        elif len(argv) == 3 and argv[1] == "tree":
            print_tree(ET.parse(argv[2]).getroot())
        else:
            print(__doc__)
            return 1
    except urllib.error.HTTPError as e:
        print(f"HTTP error {e.code} for {e.url}")
        return 2
    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}")
        return 2
    except ET.ParseError as e:
        print(f"Could not parse XML: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))