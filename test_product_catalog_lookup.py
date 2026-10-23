"""Offline regression tests for VoiceOps exact-match product-catalog lookup.

Runs against the supplied synthetic products.csv and run.py helper. No API calls
are made and no project files are modified.
Run from the VoiceOps project root with: py test_product_catalog_lookup.py
"""
from __future__ import annotations

import csv
import unittest
from pathlib import Path

try:
    import run
except Exception as exc:  # pragma: no cover - diagnostics for user's environment
    raise SystemExit(
        "Could not import run.py. Run this test beside run.py and install the "
        f"project requirements. Import error: {type(exc).__name__}: {exc}"
    )

DATASET = Path(run.DATASET)
PRODUCTS_FILE = DATASET / "data" / "products.csv"


def read_products() -> list[dict[str, str]]:
    if not PRODUCTS_FILE.exists():
        raise SystemExit(
            f"Product catalog not found at {PRODUCTS_FILE}. Ensure the supplied "
            "dataset is extracted beside the project."
        )
    with PRODUCTS_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


PRODUCTS = read_products()


class ProductCatalogLookupTests(unittest.TestCase):
    def test_expected_synthetic_catalog_size_and_unique_skus(self):
        self.assertEqual(len(PRODUCTS), 70, "Expected 70 synthetic product rows")
        skus = [str(row.get("sku", "")).strip().upper() for row in PRODUCTS]
        self.assertTrue(all(skus), "Every product row must have a SKU")
        self.assertEqual(len(skus), len(set(skus)), "Product SKUs must be unique")

    def test_every_sku_resolves_to_its_exact_catalog_row(self):
        lookup = getattr(run, "_product_catalog_chunks_for_question", None)
        self.assertTrue(
            callable(lookup),
            "run.py is missing _product_catalog_chunks_for_question; check that "
            "the exact-match product-catalog improvement is installed.",
        )
        failures: list[str] = []
        for product in PRODUCTS:
            sku = str(product["sku"]).strip().upper()
            matches = lookup(f"Please look up SKU {sku}.")
            exact = [m for m in matches if str(m.get("section_id", "")).upper() == sku]
            if len(exact) != 1 or exact[0].get("source") != "products.csv":
                failures.append(f"{sku}: expected one exact products.csv match, got {len(exact)}")
                continue
            record_text = str(exact[0].get("text", ""))
            if f"SKU={sku}" not in record_text:
                failures.append(f"{sku}: returned record text does not include SKU field")
        self.assertEqual(failures, [], "SKU lookup failures:\n" + "\n".join(failures[:20]))

    def test_product_title_queries_include_each_expected_sku(self):
        lookup = getattr(run, "_product_catalog_chunks_for_question", None)
        self.assertTrue(callable(lookup), "Exact-match product lookup helper is missing")
        failures: list[str] = []
        for product in PRODUCTS:
            sku = str(product["sku"]).strip().upper()
            title = str(product.get("title", "")).strip()
            base_title = title.split(",", 1)[0].strip()
            matches = lookup(f"Tell me about the product {base_title}.")
            returned_skus = {str(m.get("section_id", "")).upper() for m in matches}
            if sku not in returned_skus:
                failures.append(f"{sku}: title query did not include {base_title!r}; returned {sorted(returned_skus)}")
        self.assertEqual(failures, [], "Title lookup failures:\n" + "\n".join(failures[:20]))

    def test_unknown_sku_does_not_match_a_different_product(self):
        lookup = getattr(run, "_product_catalog_chunks_for_question", None)
        self.assertTrue(callable(lookup), "Exact-match product lookup helper is missing")
        matches = lookup("Please look up SKU ZZQ-9999.")
        self.assertFalse(matches, f"Unknown SKU unexpectedly matched: {[m.get('section_id') for m in matches]}")

    def test_unrelated_question_does_not_add_product_catalog_evidence(self):
        lookup = getattr(run, "_product_catalog_chunks_for_question", None)
        self.assertTrue(callable(lookup), "Exact-match product lookup helper is missing")
        matches = lookup("What is the general process for getting a refund?")
        self.assertFalse(matches, f"Unrelated question unexpectedly matched catalog rows: {[m.get('section_id') for m in matches]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
