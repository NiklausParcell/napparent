//! Load [`ChunkTable`] / [`ColGraph`] from Arrow [`RecordBatch`].

use crate::ndarrow_bridge::{array1_f32_to_arrow, f32_view, f64_view};
use crate::table::{BatchChunk, BatchColumn, ChunkTable, ColGraph, ColumnVec, TargetColumn};
use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Float32Array, Int16Array, Int32Array, Int64Array,
    Int8Array, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use std::collections::HashSet;
use std::sync::Arc;

pub(crate) fn col_to_f32(col: &ArrayRef) -> Result<Vec<f32>, String> {
    match col.data_type() {
        DataType::Float32 => Ok(col
            .as_primitive::<arrow::datatypes::Float32Type>()
            .values()
            .iter()
            .copied()
            .collect()),
        DataType::Float64 => {
            let a = col.as_primitive::<arrow::datatypes::Float64Type>();
            Ok(a.values().iter().map(|x| *x as f32).collect())
        }
        DataType::Int8 => {
            let a: &Int8Array = col.as_primitive();
            Ok(a.values().iter().map(|x| *x as f32).collect())
        }
        DataType::Int16 => {
            let a: &Int16Array = col.as_primitive();
            Ok(a.values().iter().map(|x| *x as f32).collect())
        }
        DataType::Int32 => {
            let a: &Int32Array = col.as_primitive();
            Ok(a.values().iter().map(|x| *x as f32).collect())
        }
        DataType::Int64 => {
            let a: &Int64Array = col.as_primitive();
            Ok(a.values().iter().map(|x| *x as f32).collect())
        }
        DataType::UInt8 => {
            let a: &UInt8Array = col.as_primitive();
            Ok(a.values().iter().map(|x| *x as f32).collect())
        }
        DataType::UInt16 => {
            let a: &UInt16Array = col.as_primitive();
            Ok(a.values().iter().map(|x| *x as f32).collect())
        }
        DataType::UInt32 => {
            let a: &UInt32Array = col.as_primitive();
            Ok(a.values().iter().map(|x| *x as f32).collect())
        }
        DataType::UInt64 => {
            let a: &UInt64Array = col.as_primitive();
            Ok(a.values().iter().map(|x| *x as f32).collect())
        }
        DataType::Boolean => {
            let a: &BooleanArray = col.as_boolean();
            Ok((0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        0.0
                    } else if a.value(i) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect())
        }
        other => Err(format!("unsupported numeric arrow type: {other}")),
    }
}

fn col_to_utf8(col: &ArrayRef) -> Result<Vec<String>, String> {
    match col.data_type() {
        DataType::Utf8 => {
            let a = col.as_string::<i32>();
            Ok((0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        "empty".to_string()
                    } else {
                        a.value(i).to_string()
                    }
                })
                .collect())
        }
        DataType::LargeUtf8 => {
            let a = col.as_string::<i64>();
            Ok((0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        "empty".to_string()
                    } else {
                        a.value(i).to_string()
                    }
                })
                .collect())
        }
        DataType::Boolean => {
            let a: &BooleanArray = col.as_boolean();
            Ok((0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        "empty".to_string()
                    } else if a.value(i) {
                        "true".to_string()
                    } else {
                        "false".to_string()
                    }
                })
                .collect())
        }
        DataType::Float32
        | DataType::Float64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64 => {
            let v = col_to_f32(col)?;
            Ok(v.into_iter().map(|x| x.to_string()).collect())
        }
        other => Err(format!("unsupported string-like arrow type: {other}")),
    }
}

fn feature_column_from_arrow(col: ArrayRef, dt: &DataType) -> Result<BatchColumn, String> {
    match dt {
        DataType::Float32 => match f32_view(&col) {
            Ok(_) => Ok(BatchColumn::F32(col)),
            Err(_) => Ok(BatchColumn::Owned(ColumnVec::F32(col_to_f32(&col)?))),
        },
        DataType::Float64 => match f64_view(&col) {
            Ok(_) => Ok(BatchColumn::F64(col)),
            Err(_) => Ok(BatchColumn::Owned(ColumnVec::F32(col_to_f32(&col)?))),
        },
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Boolean => Ok(BatchColumn::Owned(ColumnVec::F32(col_to_f32(&col)?))),
        DataType::Utf8 | DataType::LargeUtf8 => Ok(BatchColumn::Utf8(col)),
        DataType::Dictionary(_, _) => {
            let a = arrow::compute::cast(&col, &DataType::Utf8)
                .map_err(|e| format!("dictionary decode: {e}"))?;
            Ok(BatchColumn::Owned(ColumnVec::Utf8(col_to_utf8(&a)?)))
        }
        _ => Ok(BatchColumn::Owned(ColumnVec::Utf8(col_to_utf8(&col)?))),
    }
}

fn target_from_arrow(col: ArrayRef) -> Result<TargetColumn, String> {
    if col.data_type() == &DataType::Float32 && f32_view(&col).is_ok() {
        return Ok(TargetColumn::F32(col));
    }
    let y = col_to_f32(&col).or_else(|_| {
        col_to_utf8(&col).map(|v| {
            v.into_iter()
                .map(|s| s.parse::<f32>().unwrap_or(0.0))
                .collect()
        })
    })?;
    Ok(TargetColumn::Owned(y))
}

/// Zero-copy batch split: feature columns share Arrow buffers via `ArrayRef`.
pub fn split_batch_views(
    batch: &RecordBatch,
    target: &str,
    cols_to_drop: &[String],
) -> Result<(BatchChunk, TargetColumn, ColGraph), String> {
    let schema = batch.schema();
    let fields: Vec<_> = schema.fields().iter().cloned().collect();
    let mut target_idx: Option<usize> = None;
    for (i, f) in fields.iter().enumerate() {
        if f.name() == target {
            target_idx = Some(i);
            break;
        }
    }
    let ti = target_idx.ok_or_else(|| format!("target column `{target}` not in batch schema"))?;

    let drop_set: HashSet<&str> = cols_to_drop.iter().map(String::as_str).collect();
    if !drop_set.contains(target) {
        return Err(format!(
            "`cols_to_drop` must include target `{target}` (see Python `load_numpy`)"
        ));
    }

    let n = batch.num_rows();
    let y_col = batch.column(ti);
    let target_col = target_from_arrow(y_col.clone())?;

    let mut names = Vec::new();
    let mut cols = Vec::new();
    let mut dropped = std::collections::HashSet::new();

    for (i, field) in fields.iter().enumerate() {
        if i == ti {
            continue;
        }
        let name = field.name().clone();
        let col = batch.column(i);
        let logical = feature_column_from_arrow(col.clone(), field.data_type())?;
        let idx = names.len();
        if drop_set.contains(name.as_str()) {
            dropped.insert(idx);
        }
        names.push(name);
        cols.push(logical);
    }

    let chunk = BatchChunk { names, cols };
    chunk.validate()?;
    if target_col.len() != n {
        return Err("Y length mismatch".into());
    }
    let col_graph = ColGraph {
        names: chunk.names.clone(),
        dropped,
    };
    Ok((chunk, target_col, col_graph))
}

pub(crate) fn batch_column_to_owned(col: &BatchColumn) -> Result<ColumnVec, String> {
    match col {
        BatchColumn::F32(a) => Ok(ColumnVec::F32(col_to_f32(a)?)),
        BatchColumn::F64(a) => Ok(ColumnVec::F32(col_to_f32(a)?)),
        BatchColumn::Utf8(a) => Ok(ColumnVec::Utf8(col_to_utf8(a)?)),
        BatchColumn::Owned(c) => Ok(c.clone()),
    }
}

pub fn batch_chunk_to_table(chunk: &BatchChunk) -> Result<ChunkTable, String> {
    chunk.validate()?;
    let cols = chunk
        .cols
        .iter()
        .map(batch_column_to_owned)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ChunkTable {
        names: chunk.names.clone(),
        cols,
    })
}

pub fn target_to_vec(target: &TargetColumn) -> Vec<f32> {
    match target {
        TargetColumn::F32(a) => f32_view(a)
            .map(|v| v.iter().copied().collect())
            .unwrap_or_else(|_| col_to_f32(a).unwrap_or_default()),
        TargetColumn::Owned(v) => v.clone(),
    }
}

pub fn target_as_outcomes(target: &TargetColumn) -> OutcomesRef<'_> {
    match target {
        TargetColumn::F32(a) => {
            OutcomesRef::View(f32_view(a).expect("F32 target validated at split"))
        }
        TargetColumn::Owned(v) => OutcomesRef::Slice(v),
    }
}

/// Borrowed outcomes for zero-copy aggregation when target is null-free Float32.
pub enum OutcomesRef<'a> {
    View(ndarray::ArrayView1<'a, f32>),
    Slice(&'a [f32]),
}

impl OutcomesRef<'_> {
    pub fn len(&self) -> usize {
        match self {
            OutcomesRef::View(v) => v.len(),
            OutcomesRef::Slice(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, i: usize) -> f32 {
        match self {
            OutcomesRef::View(v) => v[i],
            OutcomesRef::Slice(s) => s[i],
        }
    }

    pub fn sum(&self) -> f32 {
        match self {
            OutcomesRef::View(v) => v.sum(),
            OutcomesRef::Slice(s) => s.iter().sum(),
        }
    }

    pub fn to_nan0_vec(&self) -> Vec<f32> {
        match self {
            OutcomesRef::View(v) => v
                .iter()
                .map(|&x| if x.is_nan() { 0.0 } else { x })
                .collect(),
            OutcomesRef::Slice(s) => s
                .iter()
                .map(|&x| if x.is_nan() { 0.0 } else { x })
                .collect(),
        }
    }
}

/// Return the UTF-8 value at row `i`, matching [`col_to_utf8`] null/empty semantics.
pub fn utf8_value_at(col: &ArrayRef, i: usize) -> String {
    match col.data_type() {
        DataType::Utf8 => {
            let a = col.as_string::<i32>();
            if a.is_null(i) {
                "empty".to_string()
            } else {
                a.value(i).to_string()
            }
        }
        DataType::LargeUtf8 => {
            let a = col.as_string::<i64>();
            if a.is_null(i) {
                "empty".to_string()
            } else {
                a.value(i).to_string()
            }
        }
        _ => "empty".to_string(),
    }
}

/// Extract feature table `X`, target `Y`, and column metadata (materializes owned columns).
///
/// Prefer [`split_batch_views`] internally for zero-copy numeric ingest.
pub fn split_batch_xy(
    batch: &RecordBatch,
    target: &str,
    cols_to_drop: &[String],
) -> Result<(ChunkTable, Vec<f32>, ColGraph), String> {
    let (chunk, target_col, col_graph) = split_batch_views(batch, target, cols_to_drop)?;
    let table = batch_chunk_to_table(&chunk)?;
    let y = target_to_vec(&target_col);
    Ok((table, y, col_graph))
}

/// Build a [`RecordBatch`] from string-name → column map (mixed float / utf8).
pub fn batch_from_map(
    schema: arrow::datatypes::SchemaRef,
    mut columns_by_name: std::collections::HashMap<String, ColumnVec>,
) -> Result<RecordBatch, String> {
    use arrow::array::StringBuilder;

    let n = schema
        .fields()
        .first()
        .and_then(|f| columns_by_name.get(f.name()).map(|c| c.len()))
        .unwrap_or(0);

    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let name = field.name();
        let col = columns_by_name
            .remove(name)
            .ok_or_else(|| format!("missing column `{name}` building RecordBatch"))?;
        let arr: ArrayRef = match col {
            ColumnVec::F32(v) => {
                if v.len() != n {
                    return Err(format!("column {name} length {}", v.len()));
                }
                Arc::new(Float32Array::from(v))
            }
            ColumnVec::F32Array(a) => {
                if a.len() != n {
                    return Err(format!("column {name} length {}", a.len()));
                }
                array1_f32_to_arrow(a)?
            }
            ColumnVec::Utf8(v) => {
                if v.len() != n {
                    return Err(format!("column {name} length {}", v.len()));
                }
                let mut b = StringBuilder::new();
                for s in v {
                    b.append_value(s);
                }
                Arc::new(b.finish())
            }
        };
        arrays.push(arr);
    }
    RecordBatch::try_new(schema, arrays).map_err(|e| e.to_string())
}

pub fn concat_same_schema(batches: &[RecordBatch]) -> Result<RecordBatch, String> {
    if batches.is_empty() {
        return Err("empty batch list".into());
    }
    let schema = batches[0].schema();
    arrow::compute::concat_batches(&schema, batches).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn sample_batch() -> RecordBatch {
        let id = Arc::new(StringArray::from(vec!["a", "b"]));
        let feat = Arc::new(Float32Array::from(vec![1.0_f32, 20.0]));
        let target = Arc::new(Float32Array::from(vec![0.5_f32, 1.5]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("feat", DataType::Float32, false),
            Field::new("target", DataType::Float32, false),
        ]));
        RecordBatch::try_new(schema, vec![id, feat, target]).unwrap()
    }

    #[test]
    fn split_batch_views_f32_zero_copy() {
        let batch = sample_batch();
        let (chunk, target, _cg) = split_batch_views(&batch, "target", &["target".into()]).unwrap();
        assert!(matches!(target, TargetColumn::F32(_)));
        assert!(matches!(chunk.cols[1], BatchColumn::F32(_)));
    }

    #[test]
    fn split_batch_xy_matches_views_materialized() {
        let batch = sample_batch();
        let (table, y, cg) = split_batch_xy(&batch, "target", &["target".into()]).unwrap();
        let (chunk, target, cg2) = split_batch_views(&batch, "target", &["target".into()]).unwrap();
        assert_eq!(cg, cg2);
        assert_eq!(y, target_to_vec(&target));
        assert_eq!(table.names, chunk.names);
    }
}
