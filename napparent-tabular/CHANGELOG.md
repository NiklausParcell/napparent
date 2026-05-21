# Changelog

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
