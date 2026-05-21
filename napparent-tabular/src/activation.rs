//! Pluggable activations for KG pair edges and effect columns.
//!
//! **KG pair activation** maps accumulated `(sum, count)` stats in `vals_map` to scalar edge
//! weights in `vals_map_avg`. The default [`KgPairActivation::LogFrequencyWeightedMean`] applies
//! `(sum / count) * log10(count)` when count > 1, down-weighting sparse pair cells to reduce
//! outlier bias in the knowledge-graph structure.
//!
//! **Effect activation** maps per-row combined column signal to `{col}_effect` features. The
//! default [`EffectActivation::GlobalMeanContrast`] subtracts the global mean outcome.
//!
//! Future variants (Bayesian shrinkage, robust contrast, etc.) extend the enums without changing
//! the HashMap-based KG layout.

use crate::preprocess::BinDepth;

/// Raw accumulated stats for one directed pair key in `vals_map`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairStats {
    pub sum: f32,
    pub count: f32,
}

/// Context passed to effect activations per row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectContext {
    pub global_mean_outcome: f32,
}

/// How a KG edge weight is computed from accumulated pair counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KgPairActivation {
    /// `(sum / count) * log10(count)` when count > 1, else 0.
    #[default]
    LogFrequencyWeightedMean,
}

impl KgPairActivation {
    pub fn activate(&self, stats: PairStats) -> f32 {
        match self {
            KgPairActivation::LogFrequencyWeightedMean => log_frequency_weighted_mean(stats),
        }
    }
}

fn log_frequency_weighted_mean(stats: PairStats) -> f32 {
    if stats.count <= 1.0 {
        0.0
    } else {
        (stats.sum / stats.count) * stats.count.log10()
    }
}

/// How combined column signal becomes an effect feature value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EffectActivation {
    /// `combined - global_mean_outcome`
    #[default]
    GlobalMeanContrast,
}

impl EffectActivation {
    pub fn activate(&self, combined: f32, ctx: &EffectContext) -> f32 {
        match self {
            EffectActivation::GlobalMeanContrast => combined - ctx.global_mean_outcome,
        }
    }
}

/// Activation settings for both pipeline stages.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ActivationConfig {
    pub kg_pair: KgPairActivation,
    pub effect: EffectActivation,
}

/// Full configuration for [`crate::pipeline::transform_record_batches`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransformConfig {
    pub bin_depth: BinDepth,
    pub activation: ActivationConfig,
}

impl TransformConfig {
    pub fn new(bin_depth: BinDepth) -> Self {
        Self {
            bin_depth,
            activation: ActivationConfig::default(),
        }
    }

    pub fn with_activation(mut self, activation: ActivationConfig) -> Self {
        self.activation = activation;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_frequency_weighted_mean_cases() {
        let act = KgPairActivation::LogFrequencyWeightedMean;
        assert_eq!(act.activate(PairStats { sum: 10.0, count: 1.0 }), 0.0);
        assert_eq!(act.activate(PairStats { sum: 5.0, count: 0.5 }), 0.0);
        let v = act.activate(PairStats { sum: 50.0, count: 10.0 });
        assert!((v - 5.0_f32).abs() < 1e-5);
    }

    #[test]
    fn global_mean_contrast() {
        let act = EffectActivation::GlobalMeanContrast;
        let ctx = EffectContext {
            global_mean_outcome: 1.0,
        };
        assert!((act.activate(3.0, &ctx) - 2.0).abs() < 1e-6);
    }
}
