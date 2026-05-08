//! Columnar chunk representation (Arrow-derived).

/// Feature matrix columns (no target column), in schema order.
#[derive(Clone, Debug)]
pub struct ChunkTable {
    pub names: Vec<String>,
    pub cols: Vec<ColumnVec>,
}

#[derive(Clone, Debug)]
pub enum ColumnVec {
    F32(Vec<f32>),
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
            ColumnVec::Utf8(v) => v.len(),
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
