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
| Pair graph Φ under KG-pair activation `φ` (defaults: `φ_log`, `φ_mean`) | `PairAggregator`, `KgPairActivation::{LogFrequencyWeightedMean, ConditionalMean}` |
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

## Submission

Reference for arXiv v1 submission. **Do not include this README in the upload bundle.**

### Files

| File | Role |
|------|------|
| `barn_effect.tex` | **Submit this.** Primary paper. |
| `barn_effect_tight.tex` | Do not submit; companion tight exposition for the repo only. |
| `barn_effect_symbols.tex` | Do not submit; symbol/notation glossary for the repo only. |

### Tarball hygiene

Before creating the arXiv upload tarball, verify no generated files are included:

```bash
cd paper
ls *.aux *.log *.out *.toc *.synctex.gz *.pdf 2>/dev/null  # should list nothing to include
```

The `.gitignore` already excludes these; double-check before tarballing if you compiled locally.

```bash
tar czf barn_effect_v1.tar.gz barn_effect.tex
```

### arXiv categories

- **Primary:** `cs.LG`
- **Cross-list:** `stat.ML`

### arXiv "Comments" field

```
13 pages (plus appendix). Algorithm specification with strong-law convergence proof
under fixed quantization. No empirical benchmarks (deferred to future versions).
Reference implementation: github.com/NiklausParcell/napparent (Rust crate
napparent-tabular with Python bindings napparent-tabular-py).
```

### Code availability footnote

`barn_effect.tex` §Implementation cites the Rust crate `napparent-tabular` and the GitHub repo `NiklausParcell/napparent`. **Verify the repo is public before tarballing** — a paper that points at a 404 is worse than one with no link. If the repo will not yet be public at v1, replace the footnote with `Code available on request from the author.`

### Endorsement

First-time submitters to `cs.LG` may need an endorsement. Check your arXiv account status at <https://arxiv.org/user> before attempting submission.

### License

The arXiv non-exclusive license is sufficient for v1. Consider CC-BY 4.0 only if a future venue requires it.
