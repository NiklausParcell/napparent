#!/usr/bin/env python3
"""Smoke test: load CSV → PyArrow batches → nnm_tabular.return_barn_record_batches.

Run from repo root after building the extension, e.g.:

  cd nnm-tabular-rust && uv run maturin develop && uv run python ../scripts/test_nnm_tabular_bindings.py

Or: pip install -e nnm-tabular-rust then python scripts/test_nnm_tabular_bindings.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test nnm_tabular Python bindings.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "tabular" / "TN_Jerry4.csv",
        help="Path to input CSV (default: rust-learn/data/tabular/TN_Jerry4.csv)",
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
    args = parser.parse_args()

    try:
        import nnm_tabular
    except ImportError:
        print(
            "Could not import nnm_tabular. Build the extension first, e.g.\n"
            "  cd nnm-tabular-rust && maturin develop",
            file=sys.stderr,
        )
        return 1

    if nnm_tabular.return_barn_record_batches is None:
        print("nnm_tabular.return_barn_record_batches is None (extension missing).", file=sys.stderr)
        return 1

    csv_path: Path = args.csv
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
    out = nnm_tabular.return_barn_record_batches(
        batches,
        target,
        cols_to_drop,
        main_depth,
        per_column=None,
    )

    print(f"Input rows: {table.num_rows}, batches: {len(batches)}")
    print(f"Output RecordBatch: {out.num_rows} rows × {out.num_columns} columns")
    print("Schema:")
    print(out.schema)
    print("\nFirst 3 column names:", [out.schema.field(i).name for i in range(min(3, out.num_columns))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
