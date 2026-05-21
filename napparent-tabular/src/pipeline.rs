//! End-to-end chunked tabular transform driver.

use crate::activation::TransformConfig;
use crate::aggregator::PairAggregator;
use crate::arrow_io::{batch_from_map, concat_same_schema, split_batch_xy};
use crate::preprocess::PreprocessStream;
use crate::table::ColumnVec;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::Arc;

fn nan0(xs: &[f32]) -> Vec<f32> {
    xs.iter()
        .map(|&x| if x.is_nan() { 0.0 } else { x })
        .collect()
}

/// Chunked pipeline: preprocess passes, pair aggregation, then apply per batch.
///
/// # Example
///
/// ```
/// use arrow::array::{Float32Array, StringArray};
/// use arrow::datatypes::{DataType, Field, Schema};
/// use arrow::record_batch::RecordBatch;
/// use napparent_tabular::{BinDepth, TransformConfig, transform_record_batches};
/// use std::sync::Arc;
///
/// let id = Arc::new(StringArray::from(vec!["a", "b"]));
/// let feat = Arc::new(Float32Array::from(vec![1.0_f32, 20.0]));
/// let target = Arc::new(Float32Array::from(vec![0.5_f32, 1.5]));
/// let schema = Arc::new(Schema::new(vec![
///     Field::new("id", DataType::Utf8, false),
///     Field::new("feat", DataType::Float32, false),
///     Field::new("target", DataType::Float32, false),
/// ]));
/// let batch = RecordBatch::try_new(schema, vec![id, feat, target]).unwrap();
/// let config = TransformConfig::new(BinDepth::new(4));
/// let out = transform_record_batches(&[batch], "target", &["target".into()], &config).unwrap();
/// assert_eq!(out.num_rows(), 2);
/// ```
pub fn transform_record_batches(
    batches: &[RecordBatch],
    target: &str,
    cols_to_drop: &[String],
    config: &TransformConfig,
) -> Result<RecordBatch, String> {
    if batches.is_empty() {
        return Err("no record batches".into());
    }

    let (_, _, cg0) = split_batch_xy(&batches[0], target, cols_to_drop)?;
    let mut pst = PreprocessStream::new(cg0);
    for b in batches {
        let (table, _y, cg) = split_batch_xy(b, target, cols_to_drop)?;
        if cg != pst.col_graph {
            return Err("inconsistent schema across chunks".into());
        }
        pst.preprocess(&table)?;
    }
    pst.finish_map(&config.bin_depth)?;

    let column_order: Vec<String> = pst.col_graph.names.clone();

    let mut agg = PairAggregator::with_activation(config.activation.clone());
    let mut first = true;
    for b in batches {
        let (table, y, col_graph) = split_batch_xy(b, target, cols_to_drop)?;
        let x_proc = pst.use_map(&table)?;
        let outcomes = nan0(&y);

        if first {
            agg.initialize_inputs(&col_graph, target, &column_order)?;
            agg.make_col_combos();
            first = false;
        }
        agg.vals_map_updating(&x_proc, &outcomes)?;
    }
    agg.finish_map();

    let mut out_batches: Vec<RecordBatch> = Vec::new();
    for b in batches {
        let (table, y, _cg) = split_batch_xy(b, target, cols_to_drop)?;
        let x_proc = pst.use_map(&table)?;
        let outcomes = nan0(&y);
        let nnm = agg.use_map(&x_proc, &y, &outcomes)?;
        let schema = Arc::new(build_output_schema(&nnm, &column_order)?);
        let batch = batch_from_map(schema, nnm)?;
        out_batches.push(batch);
    }

    concat_same_schema(&out_batches)
}

fn build_output_schema(
    nnm: &HashMap<String, ColumnVec>,
    column_order: &[String],
) -> Result<Schema, String> {
    let mut fields: Vec<Field> = Vec::new();
    for name in column_order {
        let col = nnm
            .get(name)
            .ok_or_else(|| format!("missing column {name} in output"))?;
        let dt = match col {
            ColumnVec::F32(_) => DataType::Float32,
            ColumnVec::Utf8(_) => DataType::Utf8,
        };
        fields.push(Field::new(name, dt, false));
    }
    for name in column_order {
        let effect = format!("{name}_effect");
        if nnm.contains_key(&effect) {
            let col = nnm.get(&effect).unwrap();
            let dt = match col {
                ColumnVec::F32(_) => DataType::Float32,
                ColumnVec::Utf8(_) => DataType::Utf8,
            };
            fields.push(Field::new(&effect, dt, false));
        }
    }
    if nnm.contains_key("Actuals") {
        fields.push(Field::new("Actuals", DataType::Float32, false));
    }
    if nnm.contains_key("outcomes_effect") {
        fields.push(Field::new("outcomes_effect", DataType::Float32, false));
    }
    Ok(Schema::new(fields))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::BinDepth;
    use arrow::array::{Float32Array, StringArray};
    use std::sync::Arc;

    fn batch_small() -> RecordBatch {
        let id = Arc::new(StringArray::from(vec!["a", "b"]));
        let x = Arc::new(Float32Array::from(vec![1.0_f32, 20.0]));
        let y = Arc::new(Float32Array::from(vec![0.5_f32, 1.5]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("feat", DataType::Float32, false),
            Field::new("target", DataType::Float32, false),
        ]));
        RecordBatch::try_new(schema, vec![id, x, y]).unwrap()
    }

    #[test]
    fn pipeline_runs() {
        let b = batch_small();
        let config = TransformConfig::new(BinDepth::new(4));
        let r = transform_record_batches(
            &[b.clone(), b],
            "target",
            &["target".into()],
            &config,
        );
        assert!(r.is_ok());
        let out = r.unwrap();
        assert_eq!(out.num_rows(), 4);
    }
}
