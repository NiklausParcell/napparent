# Changelog

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
- **Breaking:** diverges from Python reference directed `vals_map` keys (`unique_tup` / `unique_inv_tup`)

# 0.2.0

- Add pluggable activations: `TransformConfig`, `ActivationConfig`, `KgPairActivation`, `EffectActivation`
- Default KG activation: `LogFrequencyWeightedMean` — `(sum/count) * log10(count)` for unbiased sparse-pair weighting
- Default effect activation: `GlobalMeanContrast` — subtract global mean outcome
- **Breaking:** `transform_record_batches` now takes `&TransformConfig` instead of `&BinDepth`
  - Migration: `TransformConfig::new(BinDepth::new(n))` replaces passing `&BinDepth` directly
- Python: optional `kg_activation` and `effect_activation` keyword arguments (defaults unchanged)

# 0.1.0

- Initial publish as napparent-tabular
- API: `transform_record_batches`, `BinDepth`, `PairAggregator`
- Output columns: `{col}_effect`, `outcomes_effect`
- Optional `parquet` feature
