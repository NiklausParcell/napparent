# napparent-tabular

**napparent** makes what a model is using apparent — starting with tabular effect features.

## Status

**0.1.0 — early release.** The API may change. The supported entry point is
`transform_record_batches`. Lower-level types (`PairAggregator`, `PreprocessStream`)
are exposed but unstable. Output feature columns use an `_effect` suffix (contrast
vs global mean outcome); outcomes are in `outcomes_effect`.

## Install

```bash
cargo add napparent-tabular
```

Optional Parquet support:

```bash
cargo add napparent-tabular --features parquet
```

## Usage

Build one or more Apache Arrow `RecordBatch` chunks (same schema), then run the
tabular transform:

```rust
use napparent_tabular::{BinDepth, transform_record_batches};
// construct batches: Vec<RecordBatch>
let depth = BinDepth::new(8);
let out = transform_record_batches(&batches, "target_col", &cols_to_drop, &depth)?;
```

Output includes original columns, `{column}_effect` features, `Actuals`, and
`outcomes_effect`.

## Python

Workspace bindings live in `napparent-tabular-py`. Future PyPI package:
`pip install napparent-tabular`.

```python
import napparent_tabular
out = napparent_tabular.transform_record_batches(batches, target, cols_to_drop, main_depth)
```

## Parity

Behavior is validated against `reference/nnm_tabular.py` in this repository.
Known differences:

- Global mean outcome uses cumulative sum/count across chunks.
- Effect columns use a scalar global mean (reference `_barn` used a per-group bug).
- Effect column order follows column index order, not hash iteration.
- Output suffix is `_effect` (reference uses `_barn`).

## Roadmap

- **napparent-multimodal** — image/sensor → embedding space (future crate)

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
