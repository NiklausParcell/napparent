//! NNM tabular preprocessing + Psychic Barnacle (Apache Arrow + ndarray).
//!
//! Python reference: `reference/nnm_tabular.py` in this workspace.
//!
//! ## Parity notes
//!
//! - **Global mean** (`avg_outcome`): Python `vals_map_updating` overwrites `self.avg_outcome` each
//!   chunk; this crate uses a **cumulative sum / count** so the final mean matches all rows.
//! - **`_barn` columns**: Diffs use `col_combined - avg_outcome` with a **scalar** global mean
//!   (fixes invalid `self.avg_outcome[g]` in the Python source).
//! - **Hash order**: Barn column order follows column index order, not `HashMap` iteration.

mod arrow_io;
mod pipeline;
mod preprocess;
mod psychic;
mod sigfig;
mod table;

pub use arrow_io::{batch_from_map, concat_same_schema, split_batch_xy};
pub use pipeline::return_barn_record_batches;
pub use preprocess::{BarnacleDepth, BinType, ColumnPreprocess, PreprocessStream, ValueKey};
pub use psychic::PsychicBarnacle;
pub use sigfig::round_to_significant_figures;
pub use table::{ChunkTable, ColGraph, ColumnVec, OutcomeSource};
