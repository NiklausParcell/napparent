# Changelog

# 0.1.0

First [crates.io](https://crates.io/crates/napparent-tabular) release.

- `transform_record_batches` / `transform_record_batches_chunked` on Apache Arrow `RecordBatch` chunks
- `TransformConfig`: bin depth, pluggable activations, optional `TransformLimits`, verbose progress
- Default activations: `LogFrequencyWeightedMean` (KG pairs), `GlobalMeanContrast` (effect columns)
- Canonical value-pair keys `(min, max)`; output `{col}_effect`, `outcomes_effect`
- ndarrow zero-copy bridge on numeric Arrow columns; optional `parquet` feature
- Cooperative cancellation via `CancelToken`; optional `progress` feature for TTY bars and SIGINT helper
- Lower peak RAM: chunked output path, column-wise i32 interning, in-place effect combine

## Pre-crates.io development history

# 0.6.0

- `verbose=true` shows in-place indicatif progress bars on stderr (batch count + percent per pass)
- Non-TTY stderr (piped/CI) falls back to throttled line logs
- Cooperative Ctrl+C: pipeline checks cancellation between batches; Python raises `KeyboardInterrupt`
- Rust: `CancelToken`, optional `CancelToken::with_ctrlc_handler()` for standalone SIGINT
- `transform_record_batches_chunked` accepts optional `&CancelToken`

# 0.5.0

- Add `TransformLimits` on `TransformConfig` for fail-fast caps (`max_rows`, `max_active_columns`, `max_col_pairs`, `max_vals_map_keys`)
- Add `transform_record_batches_chunked` — returns `Vec<RecordBatch>` without a final concat (lower peak RAM)
- `transform_record_batches` now delegates to chunked path + `concat_same_schema` (same behavior as before)
- Aggregator: column-wise i32 interning replaces per-batch `Vec<Vec<String>>` matrix (pass 2/3 memory + CPU)
- `use_map` moves columns into output instead of cloning; in-place ndarray combine for effect columns
- Python: `transform_record_batches_chunked`, `concat=False` kwarg; smoke script `--no-concat`

# 0.4.1

- Add opt-in pipeline progress logging via `TransformConfig::verbose` (writes to stderr)
- Python: `transform_record_batches(..., verbose=True)`; smoke script `--verbose` (default on)

# 0.4.0

- Integrate [ndarrow](https://docs.rs/ndarrow) for zero-copy Arrow ↔ ndarray on numeric paths
- Bump `ndarray` to 0.17 (required by ndarrow)
- Add `split_batch_views`, `BatchChunk`, `BatchColumn`, `TargetColumn`, `OutcomesRef` for batch-scoped column views
- Pipeline uses `preprocess_batch` / `use_map_batch` with shared `ArrayRef` buffers instead of copying Float32 columns at ingest
- Effect columns export via `ColumnVec::F32Array` + `IntoArrow` (zero-copy `RecordBatch` build)
- `vals_map_updating` accepts `OutcomesRef` (view over null-free Float32 target when possible)
- `split_batch_xy` retained as owned-materialization wrapper for external callers

# 0.3.0

- KG value pairs use canonical unordered keys `(min, max)` in `vals_map` and `vals_map_avg`
- Fixes double-counting from separate `(u,v)` / `(v,u)` entries and redundant inverse handling in `finish_map`
- `vals_map_updating` uses a single row pass per column combo (more efficient)

# 0.2.0

- Add pluggable activations: `TransformConfig`, `ActivationConfig`, `KgPairActivation`, `EffectActivation`
- Default KG activation: `LogFrequencyWeightedMean` — `(sum/count) * log10(count)` for unbiased sparse-pair weighting
- Default effect activation: `GlobalMeanContrast` — subtract global mean outcome
- `transform_record_batches` takes `&TransformConfig` instead of `&BinDepth`
- Python: optional `kg_activation` and `effect_activation` keyword arguments (defaults unchanged)
