# Theory: tabular Barn Effect

Formal mathematics for the tabular effect pipeline implemented in [`napparent-tabular`](../napparent-tabular/).

The paper uses the name **Barn Effect**; the Rust crate exports `{column}_effect` columns and `outcomes_effect` (see crate CHANGELOG).

## Reading order

| Read first | File | Why |
|------------|------|-----|
| Start here | [`barn_effect_tight.tex`](barn_effect_tight.tex) | Short exposition; closest to default crate activations |
| Full detail | [`barn_effect.tex`](barn_effect.tex) | Definitions, convergence, complexity |
| Lookup | [`barn_effect_symbols.tex`](barn_effect_symbols.tex) | Notation and symbol glossary |

## Mapping to `napparent-tabular`

| Paper concept | Crate |
|---------------|--------|
| Phase 1: binning / vocabulary | `PreprocessStream`, `BinDepth` |
| Pair graph Φ, `(sum/count) × log10(count)` | `PairAggregator`, `KgPairActivation::LogFrequencyWeightedMean` |
| Per-row effect = combined − global mean | `EffectActivation::GlobalMeanContrast`, `{col}_effect` |
| Canonical key $\kappa(u,v) = (\min,\max)$ on encoded labels | `PairAggregator::canonical_val_pair` |
| Per-chunk significance threshold $\theta_k = \lfloor \alpha n_k \rfloor$ | `vals_map_updating` (default $\alpha = 0.01$) |
| Fixed partner divisor $m_c = p - 1$ | `PairAggregator::m_divisor` |
| Three-pass chunked pipeline | `transform_record_batches` / `transform_record_batches_chunked` |

## Build PDF locally

Sources only are tracked in git. To compile (requires `pdflatex`):

```bash
cd paper
pdflatex barn_effect_tight.tex
pdflatex barn_effect_tight.tex   # second pass for references
```

Output `barn_effect_tight.pdf` is gitignored.
