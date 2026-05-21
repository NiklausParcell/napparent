# napparent-tabular

**napparent** makes what a model is using apparent — starting with tabular effect features.

## Status

**0.4.0 — early release.** The API may change. The supported entry point is
`transform_record_batches` with a [`TransformConfig`](https://docs.rs/napparent-tabular).
Lower-level types (`PairAggregator`, `PreprocessStream`) are exposed but unstable.
Output feature columns use an `_effect` suffix; outcomes are in `outcomes_effect`.

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
use napparent_tabular::{BinDepth, TransformConfig, transform_record_batches};
// construct batches: Vec<RecordBatch>
let config = TransformConfig::new(BinDepth::new(8));
let out = transform_record_batches(&batches, "target_col", &cols_to_drop, &config)?;
```

Output includes original columns, `{column}_effect` features, `Actuals`, and
`outcomes_effect`.

## Arrow / ndarray bridge

Numeric columns use [ndarrow](https://docs.rs/ndarrow) for zero-copy views from
Apache Arrow `RecordBatch` data during preprocessing and aggregation training.
Float32 effect columns are exported back to Arrow without an extra buffer copy.
Binned label columns (`Utf8`) and KG HashMap state still allocate as before.

## Activations

KG pair edges and effect columns use pluggable activations (see `activation` module).

| Stage | Default | Formula |
|-------|---------|---------|
| KG pair | `LogFrequencyWeightedMean` | `(sum/count) * log10(count)` when count > 1 |
| Effect | `GlobalMeanContrast` | `combined - global_mean_outcome` |

Log-frequency weighting reduces bias from sparse / outlier pair cells in the HashMap KG.
Value-pair keys are stored in canonical `(min, max)` order so `(u,v)` and `(v,u)` share one bucket.
More activations (Bayesian, robust contrast, etc.) are planned.

```rust
use napparent_tabular::{ActivationConfig, BinDepth, TransformConfig};

let config = TransformConfig::new(BinDepth::new(8));
// defaults: LogFrequencyWeightedMean + GlobalMeanContrast
```

## Python

Workspace bindings live in `napparent-tabular-py`. Future PyPI package:
`pip install napparent-tabular`.

```python
import napparent_tabular
out = napparent_tabular.transform_record_batches(
    batches, target, cols_to_drop, main_depth,
    kg_activation="log_frequency_weighted_mean",
    effect_activation="global_mean_contrast",
)
```

## Parity

Behavior is validated against `reference/nnm_tabular.py` in this repository.
Known differences:

- Global mean outcome uses cumulative sum/count across chunks.
- Effect columns use a scalar global mean (reference `_barn` used a per-group bug).
- Effect column order follows column index order, not hash iteration.
- Output suffix is `_effect` (reference uses `_barn`).
- Value-pair keys are canonical unordered `(min, max)` (reference keeps directed `(u,v)` and `(v,u)` separately).

## Roadmap

- **napparent-multimodal** — image/sensor → embedding space (future crate)

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
