# napparent-tabular

**napparent** makes what a model is using apparent — starting with tabular effect features.

## Status

**0.1.0 — early release.** The API may change. The supported entry point is
`transform_record_batches` with a [`TransformConfig`](https://docs.rs/napparent-tabular).
For large datasets, use [`transform_record_batches_chunked`](https://docs.rs/napparent-tabular)
and set [`TransformLimits`](https://docs.rs/napparent-tabular) — the default
`transform_record_batches` concatenates all output batches and can OOM when row count × column
count is large. Lower-level types (`PairAggregator`, `PreprocessStream`) are exposed but unstable.
Output feature columns use an `_effect` suffix; outcomes are in `outcomes_effect`.

## Install

```bash
cargo add napparent-tabular
```

Optional features:

```bash
cargo add napparent-tabular --features progress   # TTY progress bars + SIGINT helper
cargo add napparent-tabular --features parquet    # Parquet I/O
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

## Input conventions

Before binning and accumulation:

- Numerical `NaN` → `0`
- Categorical missing / `NaN` → `"empty"` (maps to rare token ε when infrequent)
- Outcome `NaN` → `0` (contributes zero to pair statistics)

These match the default rules in the [theory write-up](https://github.com/NiklausParcell/napparent/tree/main/paper).

## Arrow / ndarray bridge

Numeric columns use [ndarrow](https://docs.rs/ndarrow) (pre-1.0, pinned) for zero-copy views from
Apache Arrow `RecordBatch` data during preprocessing and aggregation training.
Float32 effect columns are exported back to Arrow without an extra buffer copy.
Binned label columns (`Utf8`) and KG HashMap state still allocate as before.

Long runs: enable progress with `TransformConfig::new(depth).with_verbose(true)`.
With the `progress` feature on a TTY, stderr shows an in-place bar per pass; without the feature
or when stderr is piped, verbose mode falls back to throttled line logs.
Cooperative cancellation uses `CancelToken` (Python bindings wire `KeyboardInterrupt` via a hook).

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

Python bindings live in the GitHub repo (`napparent-tabular-py`); **not on PyPI yet**.
Install from a clone with maturin:

```bash
git clone https://github.com/NiklausParcell/napparent
cd napparent
maturin develop
```

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

## Algorithm

Behavior follows the Barn Effect algorithm (canonical κ keys, fixed partner divisor
$m_c = p - 1$, per-chunk threshold $\theta_k$).

Formal write-up: [paper/](https://github.com/NiklausParcell/napparent/tree/main/paper)
(start with [barn_effect_tight.tex](https://github.com/NiklausParcell/napparent/blob/main/paper/barn_effect_tight.tex)).

## Roadmap

- **napparent-multimodal** — image/sensor → embedding space (future crate)

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
