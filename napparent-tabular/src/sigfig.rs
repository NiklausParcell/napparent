//! Rounding to *n* significant figures (numpy-style).

use ndarray::{Array1, ArrayView1};

/// Match `preprocess_stream.round_to_significant_figures` / NumPy `// 1` on log scale.
pub fn round_to_significant_figures(arr: ArrayView1<f32>, n: u32) -> Array1<f32> {
    let n = n.max(1);
    let mut out = Array1::zeros(arr.len());
    for (i, &x) in arr.iter().enumerate() {
        let x = x as f64;
        if x == 0.0 || !x.is_finite() {
            out[i] = x as f32;
            continue;
        }
        let ax = x.abs();
        let scale = (ax.log10().floor() as i32) - (n as i32 - 1);
        let pow10 = 10_f64.powi(scale);
        let rounded = (x / pow10).round() * pow10;
        out[i] = rounded as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn zeros_unchanged() {
        let a = array![0.0_f32, 0.0];
        let b = round_to_significant_figures(a.view(), 4);
        assert_eq!(b, a);
    }

    #[test]
    fn matches_numpy_style_mantissa() {
        let a = array![123.456_f32, -0.00789];
        let b = round_to_significant_figures(a.view(), 4);
        assert!((b[0] - 123.5).abs() < 1e-3);
        assert!((b[1] - (-0.00789)).abs() < 1e-5 || (b[1] - (-0.007889)).abs() < 1e-5);
    }
}
