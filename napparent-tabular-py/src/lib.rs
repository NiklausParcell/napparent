//! PyO3 bindings: PyArrow `RecordBatch` in/out via `arrow-pyarrow`.

use arrow::record_batch::RecordBatch;
use arrow_pyarrow::{FromPyArrow, IntoPyArrow};
use napparent_tabular::{
    split_batch_xy, transform_record_batches, ActivationConfig, BinDepth, EffectActivation,
    KgPairActivation, TransformConfig,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn depth_from_args(main: usize, per_column: Option<Vec<(usize, usize)>>) -> BinDepth {
    let mut d = BinDepth::new(main);
    if let Some(p) = per_column {
        for (col, depth) in p {
            d.per_column.insert(col, depth);
        }
    }
    d
}

fn parse_kg_activation(name: &str) -> Result<KgPairActivation, String> {
    match name {
        "log_frequency_weighted_mean" => Ok(KgPairActivation::LogFrequencyWeightedMean),
        other => Err(format!(
            "unknown kg_activation {other:?}; supported: \"log_frequency_weighted_mean\""
        )),
    }
}

fn parse_effect_activation(name: &str) -> Result<EffectActivation, String> {
    match name {
        "global_mean_contrast" => Ok(EffectActivation::GlobalMeanContrast),
        other => Err(format!(
            "unknown effect_activation {other:?}; supported: \"global_mean_contrast\""
        )),
    }
}

fn config_from_args(
    main_depth: usize,
    per_column: Option<Vec<(usize, usize)>>,
    kg_activation: &str,
    effect_activation: &str,
) -> Result<TransformConfig, String> {
    let bin_depth = depth_from_args(main_depth, per_column);
    let activation = ActivationConfig {
        kg_pair: parse_kg_activation(kg_activation)?,
        effect: parse_effect_activation(effect_activation)?,
    };
    Ok(TransformConfig {
        bin_depth,
        activation,
    })
}

/// Run full tabular transform on one or more `pyarrow.RecordBatch` (same schema).
#[pyfunction]
#[pyo3(name = "transform_record_batches")]
#[pyo3(signature = (batches, target, cols_to_drop, main_depth, per_column=None, kg_activation="log_frequency_weighted_mean", effect_activation="global_mean_contrast"))]
fn transform_record_batches_py<'py>(
    py: Python<'py>,
    batches: Vec<Bound<'py, PyAny>>,
    target: String,
    cols_to_drop: Vec<String>,
    main_depth: usize,
    per_column: Option<Vec<(usize, usize)>>,
    kg_activation: &str,
    effect_activation: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let mut rs_batches = Vec::with_capacity(batches.len());
    for b in &batches {
        let rb = RecordBatch::from_pyarrow_bound(b)?;
        rs_batches.push(rb);
    }
    let config = config_from_args(main_depth, per_column, kg_activation, effect_activation)
        .map_err(PyValueError::new_err)?;
    let out = transform_record_batches(&rs_batches, &target, &cols_to_drop, &config)
        .map_err(PyValueError::new_err)?;
    out.into_pyarrow(py)
}

/// Debug helper: split X/Y like Python `load_numpy` (returns Y as float32 array + col metadata dict).
#[pyfunction]
#[pyo3(name = "split_batch_xy")]
fn split_batch_xy_py(
    batch: Bound<'_, PyAny>,
    target: String,
    cols_to_drop: Vec<String>,
) -> PyResult<(Bound<'_, PyAny>, Bound<'_, PyAny>)> {
    let py = batch.py();
    let rb = RecordBatch::from_pyarrow_bound(&batch)?;
    let (_table, y, cg) =
        split_batch_xy(&rb, &target, &cols_to_drop).map_err(PyValueError::new_err)?;
    let y_list = pyo3::types::PyList::new(py, y.iter().copied())?;
    let meta = pyo3::types::PyDict::new(py);
    meta.set_item("names", cg.names)?;
    meta.set_item("dropped", cg.dropped.iter().copied().collect::<Vec<_>>())?;
    Ok((y_list.into_any(), meta.into_any()))
}

#[pymodule]
fn _napparent_tabular(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transform_record_batches_py, m)?)?;
    m.add_function(wrap_pyfunction!(split_batch_xy_py, m)?)?;
    Ok(())
}
