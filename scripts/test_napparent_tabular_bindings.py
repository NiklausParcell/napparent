#!/usr/bin/env python3
"""Smoke test: load CSV → PyArrow batches → napparent_tabular.transform_record_batches.

Run from repo root after building the extension, e.g.:

  maturin develop && python scripts/test_napparent_tabular_bindings.py

Or: pip install -e . then python scripts/test_napparent_tabular_bindings.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.csv as pacsv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "smoke.csv"
LEGACY_DATA = REPO_ROOT / "data" / "tabular" / "TN_Jerry4.csv"


def default_csv_path() -> Path:
    if DEFAULT_FIXTURE.is_file():
        return DEFAULT_FIXTURE
    return LEGACY_DATA


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test napparent_tabular Python bindings.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to input CSV (default: tests/fixtures/smoke.csv, else data/tabular/TN_Jerry4.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5_000,
        help="Max rows to read (0 = all rows; smaller is faster for a quick check).",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=2_500,
        help="Rows per RecordBatch (same schema across batches).",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print pipeline progress to stderr during transform (default: true).",
    )
    args = parser.parse_args()

    try:
        import napparent_tabular
    except ImportError:
        print(
            "Could not import napparent_tabular. Build the extension first, e.g.\n"
            "  maturin develop",
            file=sys.stderr,
        )
        return 1

    if napparent_tabular.transform_record_batches is None:
        print(
            "napparent_tabular.transform_record_batches is None (extension missing).",
            file=sys.stderr,
        )
        return 1

    csv_path: Path = args.csv if args.csv is not None else default_csv_path()
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    table = pacsv.read_csv(csv_path)
    if args.limit > 0 and table.num_rows > args.limit:
        table = table.slice(0, args.limit)
    if args.chunk_rows <= 0:
        batches = table.to_batches()
    else:
        n = table.num_rows
        batches = []
        step = args.chunk_rows
        for start in range(0, n, step):
            batches.extend(table.slice(start, min(step, n - start)).to_batches())

    print(f"Read {table.num_rows} rows from {csv_path} ({len(batches)} batches)")

    target = "std"
    cols_to_drop = [
        "npi",
        "billing_code",
        "npi_zip5",
        "census_zip",
        "census_city",
        "census_lat",
        "census_lng",
        "census_county_fips",
        "census_county_name",
        "median",
        target,
        "count",
        "family",
        "sub_family",
    ]

    main_depth = 8
    if args.verbose:
        print("Running transform (progress on stderr)...", file=sys.stderr)
    out = napparent_tabular.transform_record_batches(
        batches,
        target,
        cols_to_drop,
        main_depth,
        per_column=None,
        verbose=args.verbose,
    )

    print(f"Input rows: {table.num_rows}, batches: {len(batches)}")
    print(f"Output RecordBatch: {out.num_rows} rows × {out.num_columns} columns")
    print("Schema:")
    print(out.schema)
    print("\nFirst 3 column names:", [out.schema.field(i).name for i in range(min(3, out.num_columns))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
