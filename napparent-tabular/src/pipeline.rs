//! End-to-end chunked tabular transform driver.

use crate::activation::TransformConfig;
use crate::aggregator::PairAggregator;
use crate::arrow_io::{
    batch_from_map, concat_same_schema, split_batch_views, target_as_outcomes, target_to_vec,
    OutcomesRef,
};
use crate::cancel::{CancelToken, INTERRUPT_MSG};
use crate::preprocess::PreprocessStream;
use crate::progress::{ProgressReporter, ProgressTimer};
use crate::table::ColumnVec;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::sync::Arc;

fn check_limit(value: usize, max: Option<usize>, label: &str) -> Result<(), String> {
    if let Some(max) = max {
        if value > max {
            return Err(format!("limit exceeded: {label} ({value} > {max})"));
        }
    }
    Ok(())
}

fn outcomes_for_effect(outcomes: &OutcomesRef<'_>) -> Vec<f32> {
    outcomes.to_nan0_vec()
}

fn check_cancel(cancel: Option<&CancelToken>) -> Result<(), String> {
    if let Some(c) = cancel {
        c.check()?;
    }
    Ok(())
}

/// Chunked pipeline: preprocess, pair aggregation, apply — returns one output batch per input batch.
pub fn transform_record_batches_chunked(
    batches: &[RecordBatch],
    target: &str,
    cols_to_drop: &[String],
    config: &TransformConfig,
    cancel: Option<&CancelToken>,
) -> Result<Vec<RecordBatch>, String> {
    let mut reporter = ProgressReporter::from_verbose(config.verbose);
    match transform_record_batches_chunked_inner(
        batches,
        target,
        cols_to_drop,
        config,
        cancel,
        &mut reporter,
    ) {
        Ok(out) => Ok(out),
        Err(e) => {
            if e.contains(INTERRUPT_MSG) {
                reporter.abandon();
            }
            Err(e)
        }
    }
}

fn transform_record_batches_chunked_inner(
    batches: &[RecordBatch],
    target: &str,
    cols_to_drop: &[String],
    config: &TransformConfig,
    cancel: Option<&CancelToken>,
    reporter: &mut ProgressReporter,
) -> Result<Vec<RecordBatch>, String> {
    if batches.is_empty() {
        return Err("no record batches".into());
    }

    let limits = &config.limits;
    let total_timer = ProgressTimer::start();
    let n_batches = batches.len();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

    check_limit(total_rows, limits.max_rows, "max_rows")?;

    reporter.log(&format!(
        "starting transform: {n_batches} batches, {total_rows} total rows"
    ));

    let (_, _, cg0) = split_batch_views(&batches[0], target, cols_to_drop)?;
    let mut pst = PreprocessStream::new(cg0);

    reporter.pass_start(1, 3, "preprocessing", n_batches);
    let pass1 = ProgressTimer::start();
    for (i, b) in batches.iter().enumerate() {
        check_cancel(cancel)?;
        let (chunk, _target, cg) = split_batch_views(b, target, cols_to_drop)?;
        if cg != pst.col_graph {
            return Err("inconsistent schema across chunks".into());
        }
        reporter.batch_tick(
            i,
            n_batches,
            b.num_rows(),
            &format!(
                "preprocess batch {}/{} ({} rows)",
                i + 1,
                n_batches,
                b.num_rows()
            ),
        );
        pst.preprocess_batch(&chunk)?;
    }
    pst.finish_map(&config.bin_depth)?;
    check_limit(pst.cols.len(), limits.max_active_columns, "max_active_columns")?;
    reporter.pass_finish(&format!(
        "pass 1/3 done in {:.1}s: bins finished, {} active feature columns",
        pass1.elapsed_secs(),
        pst.cols.len()
    ));

    let column_order: Vec<String> = pst.col_graph.names.clone();

    let mut agg = PairAggregator::with_activation(config.activation.clone());
    let mut first = true;

    reporter.pass_start(2, 3, "KG", n_batches);
    let pass2 = ProgressTimer::start();
    for (i, b) in batches.iter().enumerate() {
        check_cancel(cancel)?;
        let (chunk, target_col, col_graph) = split_batch_views(b, target, cols_to_drop)?;
        let x_proc = pst.use_map_batch(&chunk)?;
        let outcomes = target_as_outcomes(&target_col);

        if first {
            agg.initialize_inputs(&col_graph, target, &column_order)?;
            agg.make_col_combos();
            check_limit(agg.col_array.len(), limits.max_col_pairs, "max_col_pairs")?;
            first = false;
        }
        reporter.batch_tick(
            i,
            n_batches,
            b.num_rows(),
            &format!("KG update batch {}/{}", i + 1, n_batches),
        );
        agg.vals_map_updating(&x_proc, &outcomes)?;
        check_limit(agg.vals_map_len(), limits.max_vals_map_keys, "max_vals_map_keys")?;
    }
    agg.finish_map();
    reporter.pass_finish(&format!(
        "pass 2/3 done in {:.1}s: KG finished, {} pair keys, global mean outcome {:.6}",
        pass2.elapsed_secs(),
        agg.vals_map_avg.len(),
        agg.avg_outcome
    ));

    reporter.pass_start(3, 3, "transform", n_batches);
    let pass3 = ProgressTimer::start();
    let mut out_batches: Vec<RecordBatch> = Vec::with_capacity(n_batches);
    for (i, b) in batches.iter().enumerate() {
        check_cancel(cancel)?;
        let (chunk, target_col, _cg) = split_batch_views(b, target, cols_to_drop)?;
        let x_proc = pst.use_map_batch(&chunk)?;
        let outcomes = target_as_outcomes(&target_col);
        let outcomes_vec = outcomes_for_effect(&outcomes);
        let y = target_to_vec(&target_col);
        reporter.batch_tick(
            i,
            n_batches,
            b.num_rows(),
            &format!("transform batch {}/{}", i + 1, n_batches),
        );
        let nnm = agg.use_map(x_proc, y, outcomes_vec)?;
        let schema = Arc::new(build_output_schema(&nnm, &column_order)?);
        let batch = batch_from_map(schema, nnm)?;
        out_batches.push(batch);
    }
    reporter.pass_finish(&format!(
        "pass 3/3 done in {:.1}s",
        pass3.elapsed_secs()
    ));

    let total_out_rows: usize = out_batches.iter().map(|b| b.num_rows()).sum();
    let n_cols = out_batches.first().map(|b| b.num_columns()).unwrap_or(0);
    reporter.finish(&format!(
        "complete in {:.1}s → {} rows × {} columns ({} output batches)",
        total_timer.elapsed_secs(),
        total_out_rows,
        n_cols,
        out_batches.len()
    ));
    Ok(out_batches)
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
    let chunks = transform_record_batches_chunked(batches, target, cols_to_drop, config, None)?;
    concat_same_schema(&chunks)
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
            ColumnVec::F32(_) | ColumnVec::F32Array(_) => DataType::Float32,
            ColumnVec::Utf8(_) => DataType::Utf8,
        };
        fields.push(Field::new(name, dt, false));
    }
    for name in column_order {
        let effect = format!("{name}_effect");
        if nnm.contains_key(&effect) {
            let col = nnm.get(&effect).unwrap();
            let dt = match col {
                ColumnVec::F32(_) | ColumnVec::F32Array(_) => DataType::Float32,
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
    use crate::activation::TransformLimits;
    use crate::cancel::CancelToken;
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

    fn run_config() -> TransformConfig {
        TransformConfig::new(BinDepth::new(4))
    }

    #[test]
    fn pipeline_runs() {
        let b = batch_small();
        let r = transform_record_batches(
            &[b.clone(), b],
            "target",
            &["target".into()],
            &run_config(),
        );
        assert!(r.is_ok());
        let out = r.unwrap();
        assert_eq!(out.num_rows(), 4);
    }

    #[test]
    fn chunked_matches_concat_row_count() {
        let b = batch_small();
        let batches = [b.clone(), b];
        let config = run_config();
        let concat = transform_record_batches(&batches, "target", &["target".into()], &config)
            .unwrap();
        let chunked = transform_record_batches_chunked(
            &batches,
            "target",
            &["target".into()],
            &config,
            None,
        )
        .unwrap();
        let chunked_rows: usize = chunked.iter().map(|b| b.num_rows()).sum();
        assert_eq!(concat.num_rows(), chunked_rows);
        assert_eq!(concat.num_columns(), chunked[0].num_columns());
        assert_eq!(concat.schema(), chunked[0].schema());
    }

    #[test]
    fn max_rows_limit_rejects() {
        let b = batch_small();
        let config = TransformConfig::new(BinDepth::new(4)).with_limits(TransformLimits {
            max_rows: Some(1),
            ..TransformLimits::default()
        });
        let err = transform_record_batches_chunked(
            &[b],
            "target",
            &["target".into()],
            &config,
            None,
        )
        .unwrap_err();
        assert!(err.contains("max_rows"));
    }

    #[test]
    fn cancel_token_stops_between_batches() {
        let b = batch_small();
        let token = CancelToken::none();
        token.request_stop();
        let config = run_config();
        let err = transform_record_batches_chunked(
            &[b.clone(), b],
            "target",
            &["target".into()],
            &config,
            Some(&token),
        )
        .unwrap_err();
        assert!(err.contains("interrupted"));
    }
}
