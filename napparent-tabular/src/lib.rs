//! Tabular effect features on Apache Arrow batches (Barn Effect algorithm).
//!
//! Discretize columns, accumulate pairwise co-occurrence statistics, and produce per-row
//! `{column}_effect` attributions. See [`transform_record_batches`](crate::pipeline::transform_record_batches)
//! and [`transform_record_batches_chunked`](crate::pipeline::transform_record_batches_chunked).
//!
//! - API docs: <https://docs.rs/napparent-tabular>
//! - Theory: <https://github.com/NiklausParcell/napparent/tree/main/paper>

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
