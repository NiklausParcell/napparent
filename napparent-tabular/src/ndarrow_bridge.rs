//! Zero-copy Arrow ↔ ndarray bridge via [ndarrow](https://docs.rs/ndarrow).

use arrow::array::{Array, ArrayRef, AsArray, Float32Array};
use arrow::datatypes::DataType;
use ndarrow::{AsNdarray, IntoArrow};
use ndarray::{Array1, ArrayView1};
use std::sync::Arc;

pub fn map_ndarrow_err(e: ndarrow::error::NdarrowError) -> String {
    e.to_string()
}

/// Zero-copy `Float32` column view when the array has no nulls.
pub fn f32_view(col: &ArrayRef) -> Result<ArrayView1<'_, f32>, String> {
    if col.data_type() != &DataType::Float32 {
        return Err(format!("expected Float32, got {}", col.data_type()));
    }
    let arr = col.as_primitive::<arrow::datatypes::Float32Type>();
    arr.as_ndarray().map_err(map_ndarrow_err)
}

/// Zero-copy `Float64` column view when the array has no nulls.
pub fn f64_view(col: &ArrayRef) -> Result<ArrayView1<'_, f64>, String> {
    if col.data_type() != &DataType::Float64 {
        return Err(format!("expected Float64, got {}", col.data_type()));
    }
    let arr = col.as_primitive::<arrow::datatypes::Float64Type>();
    arr.as_ndarray().map_err(map_ndarrow_err)
}

/// Transfer an owned `Array1<f32>` into an Arrow `Float32Array` without copying data.
pub fn array1_f32_to_arrow(arr: Array1<f32>) -> Result<ArrayRef, String> {
    let arrow: Float32Array = arr.into_arrow().map_err(map_ndarrow_err)?;
    Ok(Arc::new(arrow))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float32Array;
    use std::sync::Arc;

    #[test]
    fn f32_view_roundtrip() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let arrow: ArrayRef = Arc::new(Float32Array::from(data.clone()));
        let view = f32_view(&arrow).unwrap();
        assert_eq!(view.as_slice(), Some(data.as_slice()));

        let owned = Array1::from(view.to_owned());
        let back = array1_f32_to_arrow(owned).unwrap();
        let view2 = f32_view(&back).unwrap();
        assert_eq!(view2.as_slice(), Some(data.as_slice()));
    }
}
