from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Capstone_Project_Dataset_FTTT_2026"


def section_ids() -> set[str]:
    ids: set[str] = set()
    for path in sorted((DATASET / "kb").glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        body = raw.split("---", 2)[2] if raw.startswith("---") and raw.count("---") >= 2 else raw
        found = re.findall(r"(?m)^#{1,4}\s+\[([^\]]+)\]", body)
        if found:
            ids.update(x.strip() for x in found)
        else:
            ids.add(path.stem.upper())
    return ids


def main() -> int:
    questions_path = DATASET / "practice" / "questions.csv"
    products_path = DATASET / "data" / "products.csv"
    if not questions_path.exists() or not products_path.exists():
        print("ERROR: Practice questions or products.csv is missing from the dataset folder.")
        return 2
    with products_path.open(encoding="utf-8-sig", newline="") as f:
        products = {row.get("sku", "").strip() for row in csv.DictReader(f)}
    known_sections = section_ids()
    with questions_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    issues: list[str] = []
    answerable = sum(row.get("answerable", "").strip().lower() == "true" for row in rows)
    rows_with_gold = 0
    for row in rows:
        qid = row.get("id", "<missing-id>")
        gold_parts = [x.strip() for x in row.get("gold_evidence", "").split(";") if x.strip()]
        neg_parts = [x.strip() for x in row.get("hard_negatives", "").split(";") if x.strip()]
        kb_gold = [x for x in gold_parts if not x.startswith("products.csv:")]
        if kb_gold:
            rows_with_gold += 1
        for ref in kb_gold + neg_parts:
            if ref not in known_sections:
                issues.append(f"{qid}: section reference not found: {ref}")
        for ref in (x for x in gold_parts if x.startswith("products.csv:")):
            sku = ref.split(":", 1)[1]
            if sku not in products:
                issues.append(f"{qid}: product SKU not found: {sku}")
        if row.get("answerable", "").strip().lower() == "false" and gold_parts:
            issues.append(f"{qid}: marked unanswerable but has gold_evidence: {gold_parts}")
    print(f"Practice questions checked: {len(rows)}")
    print(f"Marked answerable: {answerable}")
    print(f"Questions with KB gold evidence: {rows_with_gold}")
    print(f"Known KB section IDs: {len(known_sections)}")
    print(f"Known product SKUs: {len(products)}")
    if issues:
        print(f"FAIL: {len(issues)} evidence-reference issue(s)")
        for issue in issues:
            print(" -", issue)
        return 1
    print("PASS: All gold KB IDs, hard-negative IDs and product SKU references resolve.")
    print("NOTE: This checks reference integrity, not actual embedding recall or generated-answer accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
