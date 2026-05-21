//! Columnar chunk representation (Arrow-derived).

use arrow::array::ArrayRef;
use ndarray::Array1;

/// Feature matrix columns (no target column), in schema order.
#[derive(Clone, Debug)]
pub struct ChunkTable {
    pub names: Vec<String>,
    pub cols: Vec<ColumnVec>,
}

#[derive(Clone, Debug)]
pub enum ColumnVec {
    F32(Vec<f32>),
    /// Owned ndarray column for zero-copy Arrow export via ndarrow.
    F32Array(Array1<f32>),
    Utf8(Vec<String>),
}

impl ChunkTable {
    pub fn nrows(&self) -> usize {
        self.cols.first().map(|c| c.len()).unwrap_or(0)
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.nrows();
        for (i, c) in self.cols.iter().enumerate() {
            if c.len() != n {
                return Err(format!(
                    "column length mismatch: column {} len {} vs {}",
                    i,
                    c.len(),
                    n
                ));
            }
        }
        Ok(())
    }
}

impl ColumnVec {
    pub fn len(&self) -> usize {
        match self {
            ColumnVec::F32(v) => v.len(),
            ColumnVec::F32Array(a) => a.len(),
            ColumnVec::Utf8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Feature columns from one batch — Arrow buffers shared via `ArrayRef` where possible.
#[derive(Clone, Debug)]
pub struct BatchChunk {
    pub names: Vec<String>,
    pub cols: Vec<BatchColumn>,
}

/// Column storage for one batch pass.
#[derive(Clone, Debug)]
pub enum BatchColumn {
    F32(ArrayRef),
    F64(ArrayRef),
    Utf8(ArrayRef),
    Owned(ColumnVec),
}

impl BatchChunk {
    pub fn nrows(&self) -> usize {
        self.cols.first().map(|c| c.len()).unwrap_or(0)
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.nrows();
        for (i, c) in self.cols.iter().enumerate() {
            if c.len() != n {
                return Err(format!(
                    "column length mismatch: column {} len {} vs {}",
                    i,
                    c.len(),
                    n
                ));
            }
        }
        Ok(())
    }
}

impl BatchColumn {
    pub fn len(&self) -> usize {
        match self {
            BatchColumn::F32(a)
            | BatchColumn::F64(a)
            | BatchColumn::Utf8(a) => a.len(),
            BatchColumn::Owned(c) => c.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            BatchColumn::F32(_)
                | BatchColumn::F64(_)
                | BatchColumn::Owned(ColumnVec::F32(_))
                | BatchColumn::Owned(ColumnVec::F32Array(_))
        )
    }

    pub fn array_ref(&self) -> Option<&ArrayRef> {
        match self {
            BatchColumn::F32(a) | BatchColumn::F64(a) | BatchColumn::Utf8(a) => Some(a),
            BatchColumn::Owned(_) => None,
        }
    }
}

/// Target column — shared Float32 buffer or owned fallback.
#[derive(Clone, Debug)]
pub enum TargetColumn {
    F32(ArrayRef),
    Owned(Vec<f32>),
}

impl TargetColumn {
    pub fn len(&self) -> usize {
        match self {
            TargetColumn::F32(a) => a.len(),
            TargetColumn::Owned(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Maps column index in `ChunkTable` -> original name; which indices are dropped from preprocessing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ColGraph {
    pub names: Vec<String>,
    pub dropped: std::collections::HashSet<usize>,
}

impl ColGraph {
    /// Indices of columns that participate in binning / pair interactions.
    pub fn active_indices(&self) -> Vec<usize> {
        (0..self.names.len())
            .filter(|i| !self.dropped.contains(i))
            .collect()
    }
}

/// Outcome vector aligned with table rows.
#[derive(Clone, Debug)]
pub enum OutcomeSource {
    FromTarget(Vec<f32>),
    External(Vec<f32>),
}

impl OutcomeSource {
    pub fn as_slice(&self) -> &[f32] {
        match self {
            OutcomeSource::FromTarget(v) | OutcomeSource::External(v) => v.as_slice(),
        }
    }
}
