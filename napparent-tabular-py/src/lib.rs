//! PyO3 bindings: PyArrow `RecordBatch` in/out via `arrow-pyarrow`.

use arrow::record_batch::RecordBatch;
use arrow_pyarrow::{FromPyArrow, IntoPyArrow};
use napparent_tabular::{
    split_batch_xy, transform_record_batches, transform_record_batches_chunked,
    ActivationConfig, BinDepth, EffectActivation, KgPairActivation, TransformConfig,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

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
    verbose: bool,
) -> Result<TransformConfig, String> {
    let bin_depth = depth_from_args(main_depth, per_column);
    let activation = ActivationConfig {
        kg_pair: parse_kg_activation(kg_activation)?,
        effect: parse_effect_activation(effect_activation)?,
    };
    Ok(TransformConfig {
        bin_depth,
        activation,
        verbose,
        limits: Default::default(),
    })
}

fn batches_from_py<'py>(
    batches: &[Bound<'py, PyAny>],
) -> PyResult<Vec<RecordBatch>> {
    let mut rs_batches = Vec::with_capacity(batches.len());
    for b in batches {
        rs_batches.push(RecordBatch::from_pyarrow_bound(b)?);
    }
    Ok(rs_batches)
}

/// Run full tabular transform; returns one concatenated `pyarrow.RecordBatch` by default.
#[pyfunction]
#[pyo3(name = "transform_record_batches")]
#[pyo3(signature = (batches, target, cols_to_drop, main_depth, per_column=None, kg_activation="log_frequency_weighted_mean", effect_activation="global_mean_contrast", verbose=false, concat=true))]
fn transform_record_batches_py<'py>(
    py: Python<'py>,
    batches: Vec<Bound<'py, PyAny>>,
    target: String,
    cols_to_drop: Vec<String>,
    main_depth: usize,
    per_column: Option<Vec<(usize, usize)>>,
    kg_activation: &str,
    effect_activation: &str,
    verbose: bool,
    concat: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let rs_batches = batches_from_py(&batches)?;
    let config = config_from_args(
        main_depth,
        per_column,
        kg_activation,
        effect_activation,
        verbose,
    )
    .map_err(PyValueError::new_err)?;

    if concat {
        let out = transform_record_batches(&rs_batches, &target, &cols_to_drop, &config)
            .map_err(PyValueError::new_err)?;
        out.into_pyarrow(py)
    } else {
        let chunks = transform_record_batches_chunked(&rs_batches, &target, &cols_to_drop, &config)
            .map_err(PyValueError::new_err)?;
        let py_batches = PyList::empty(py);
        for batch in chunks {
            py_batches.append(batch.into_pyarrow(py)?)?;
        }
        Ok(py_batches.into_any())
    }
}

/// Run full tabular transform; returns `list[pyarrow.RecordBatch]` (no concat).
#[pyfunction]
#[pyo3(name = "transform_record_batches_chunked")]
#[pyo3(signature = (batches, target, cols_to_drop, main_depth, per_column=None, kg_activation="log_frequency_weighted_mean", effect_activation="global_mean_contrast", verbose=false))]
fn transform_record_batches_chunked_py<'py>(
    py: Python<'py>,
    batches: Vec<Bound<'py, PyAny>>,
    target: String,
    cols_to_drop: Vec<String>,
    main_depth: usize,
    per_column: Option<Vec<(usize, usize)>>,
    kg_activation: &str,
    effect_activation: &str,
    verbose: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let rs_batches = batches_from_py(&batches)?;
    let config = config_from_args(
        main_depth,
        per_column,
        kg_activation,
        effect_activation,
        verbose,
    )
    .map_err(PyValueError::new_err)?;
    let chunks = transform_record_batches_chunked(&rs_batches, &target, &cols_to_drop, &config)
        .map_err(PyValueError::new_err)?;
    let py_batches = PyList::empty(py);
    for batch in chunks {
        py_batches.append(batch.into_pyarrow(py)?)?;
    }
    Ok(py_batches.into_any())
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
    let y_list = PyList::new(py, y.iter().copied())?;
    let meta = pyo3::types::PyDict::new(py);
    meta.set_item("names", cg.names)?;
    meta.set_item("dropped", cg.dropped.iter().copied().collect::<Vec<_>>())?;
    Ok((y_list.into_any(), meta.into_any()))
}

#[pymodule]
fn _napparent_tabular(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transform_record_batches_py, m)?)?;
    m.add_function(wrap_pyfunction!(transform_record_batches_chunked_py, m)?)?;
    m.add_function(wrap_pyfunction!(split_batch_xy_py, m)?)?;
    Ok(())
}
