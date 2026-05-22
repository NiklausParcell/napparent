# napparent-tabular

**napparent** makes what a model is using apparent — starting with tabular effect features.

## Status

**0.6.0 — early release.** The API may change. The supported entry point is
`transform_record_batches` with a [`TransformConfig`](https://docs.rs/napparent-tabular).
For large datasets, prefer [`transform_record_batches_chunked`](https://docs.rs/napparent-tabular)
to avoid holding a second full copy at concat time.
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

### Large data (lower peak RAM)

```rust
use napparent_tabular::{BinDepth, TransformConfig, TransformLimits, transform_record_batches_chunked};

let config = TransformConfig::new(BinDepth::new(8)).with_limits(TransformLimits {
    max_rows: Some(1_000_000),
    max_active_columns: Some(100),
    ..TransformLimits::default()
});
let batches_out = transform_record_batches_chunked(&batches, "target_col", &cols_to_drop, &config)?;
// one output RecordBatch per input batch — no mega-concat
```

`transform_record_batches` still concatenates for convenience; use chunked output when
row count × column count is large.

## Arrow / ndarray bridge

Numeric columns use [ndarrow](https://docs.rs/ndarrow) for zero-copy views from
Apache Arrow `RecordBatch` data during preprocessing and aggregation training.
Float32 effect columns are exported back to Arrow without an extra buffer copy.
Binned label columns (`Utf8`) and KG HashMap state still allocate as before.

Long runs: enable progress with `TransformConfig::new(depth).with_verbose(true)`
(or Python `verbose=True`). On an interactive terminal this shows an in-place progress
bar per pass; when stderr is piped, it falls back to throttled line logs.
Ctrl+C cancels between batches (Python: `KeyboardInterrupt`).

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
    verbose=True,
)
# lower peak RAM: concat=False or transform_record_batches_chunked(...)
chunks = napparent_tabular.transform_record_batches(
    batches, target, cols_to_drop, main_depth, concat=False,
)
```

## Parity

Behavior is validated against `reference/nnm_tabular.py` in this repository.
Known differences:

- Global mean outcome uses cumulative sum/count across chunks.
- Effect columns use a scalar global mean (reference `_barn` used a per-group bug).
- Effect column order follows column index order, not hash iteration.
- Output suffix is `_effect` (reference uses `_barn`).
- Python reference keeps directed `(u,v)` keys separately; paper and crate use canonical $\kappa(u,v)$.

## Theory

Formal write-up of the tabular algorithm: [`paper/`](../paper/) (start with [`barn_effect_tight.tex`](../paper/barn_effect_tight.tex)).

## Roadmap

- **napparent-multimodal** — image/sensor → embedding space (future crate)

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
