//! napparent tabular preprocessing (Apache Arrow + ndarray).
//!
//! Formal algorithm: see `paper/` in the repository root.

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
pub use aggregator::PairAggregator;
pub use arrow_io::{
    batch_from_map, concat_same_schema, split_batch_views, split_batch_xy, OutcomesRef,
};
pub use cancel::{CancelToken, CtrlcGuard, INTERRUPT_MSG};
pub use pipeline::{transform_record_batches, transform_record_batches_chunked};
pub use preprocess::{BinDepth, BinType, ColumnPreprocess, PreprocessStream, ValueKey};
pub use sigfig::round_to_significant_figures;
pub use table::{
    BatchChunk, BatchColumn, ChunkTable, ColGraph, ColumnVec, OutcomeSource, TargetColumn,
};
