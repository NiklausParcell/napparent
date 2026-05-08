//! Load [`ChunkTable`] / [`ColGraph`] from Arrow [`RecordBatch`].

use crate::table::{ChunkTable, ColGraph, ColumnVec};
use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Float32Array, Int16Array, Int32Array, Int64Array,
    Int8Array, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use std::collections::HashSet;
use std::sync::Arc;

fn col_to_f32(col: &ArrayRef) -> Result<Vec<f32>, String> {
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
        DataType::Float32 | DataType::Float64 | DataType::Int8 | DataType::Int16
        | DataType::Int32 | DataType::Int64 => {
            let v = col_to_f32(col)?;
            Ok(v.into_iter().map(|x| x.to_string()).collect())
        }
        other => Err(format!("unsupported string-like arrow type: {other}")),
    }
}

/// Extract feature table `X`, target `Y`, and column metadata.
///
/// `cols_to_drop` must include `target` (matching `load_numpy` in Python).
pub fn split_batch_xy(
    batch: &RecordBatch,
    target: &str,
    cols_to_drop: &[String],
) -> Result<(ChunkTable, Vec<f32>, ColGraph), String> {
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
    let y = col_to_f32(y_col).or_else(|_| {
        col_to_utf8(y_col).map(|v| {
            v.into_iter()
                .map(|s| s.parse::<f32>().unwrap_or(0.0))
                .collect()
        })
    })?;

    let mut names = Vec::new();
    let mut cols = Vec::new();
    let mut dropped = std::collections::HashSet::new();

    for (i, field) in fields.iter().enumerate() {
        if i == ti {
            continue;
        }
        let name = field.name().clone();
        let col = batch.column(i);
        let logical = match field.data_type() {
            DataType::Float32
            | DataType::Float64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Boolean => ColumnVec::F32(col_to_f32(col)?),
            DataType::Utf8 | DataType::LargeUtf8 => ColumnVec::Utf8(col_to_utf8(col)?),
            DataType::Dictionary(_, _) => {
                let a = arrow::compute::cast(col, &DataType::Utf8)
                    .map_err(|e| format!("dictionary decode: {e}"))?;
                ColumnVec::Utf8(col_to_utf8(&a)?)
            }
            _ => ColumnVec::Utf8(col_to_utf8(col)?),
        };
        let idx = names.len();
        if drop_set.contains(name.as_str()) {
            dropped.insert(idx);
        }
        names.push(name);
        cols.push(logical);
    }

    let table = ChunkTable { names, cols };
    table.validate()?;
    if y.len() != n {
        return Err("Y length mismatch".into());
    }
    let col_graph = ColGraph {
        names: table.names.clone(),
        dropped,
    };
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
        .and_then(|f| {
            columns_by_name
                .get(f.name())
                .map(|c| c.len())
        })
        .unwrap_or(0);

    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        let name = field.name();
        let col = columns_by_name.remove(name).ok_or_else(|| {
            format!("missing column `{name}` building RecordBatch")
        })?;
        let arr: ArrayRef = match col {
            ColumnVec::F32(v) => {
                if v.len() != n {
                    return Err(format!("column {name} length {}", v.len()));
                }
                Arc::new(Float32Array::from(v))
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
