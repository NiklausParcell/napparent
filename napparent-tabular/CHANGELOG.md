# Changelog

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
