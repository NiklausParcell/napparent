//! napparent tabular preprocessing (Apache Arrow + ndarray).
//!
//! Python reference: `reference/nnm_tabular.py` in this workspace.
//!
//! ## Parity notes
//!
//! - **Global mean** (`avg_outcome`): Python `vals_map_updating` overwrites `self.avg_outcome` each
//!   chunk; this crate uses a **cumulative sum / count** so the final mean matches all rows.
//! - **`_effect` columns** (reference uses `_barn`): Diffs use `col_combined - avg_outcome` with a
//!   **scalar** global mean (fixes invalid `self.avg_outcome[g]` in the Python source).
//! - **Hash order**: Effect column order follows column index order, not `HashMap` iteration.

mod activation;
mod aggregator;
mod arrow_io;
mod cancel;
mod ndarrow_bridge;
mod pipeline;
mod preprocess;
mod progress;
mod sigfig;
mod table;

pub use activation::{
    ActivationConfig, EffectActivation, EffectContext, KgPairActivation, PairStats,
    TransformConfig, TransformLimits,
};
pub use cancel::{CancelToken, CtrlcGuard, INTERRUPT_MSG};
pub use aggregator::PairAggregator;
pub use arrow_io::{
    batch_from_map, concat_same_schema, split_batch_views, split_batch_xy, OutcomesRef,
};
pub use pipeline::{transform_record_batches, transform_record_batches_chunked};
pub use preprocess::{BinDepth, BinType, ColumnPreprocess, PreprocessStream, ValueKey};
pub use sigfig::round_to_significant_figures;
pub use table::{BatchChunk, BatchColumn, ChunkTable, ColGraph, ColumnVec, OutcomeSource, TargetColumn};
