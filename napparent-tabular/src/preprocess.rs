//! Streaming preprocessing (`preprocess_stream` in Python).

use crate::sigfig::round_to_significant_figures;
use crate::table::{ChunkTable, ColGraph, ColumnVec};
use ndarray::Array1;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinType {
    Numerical,
    Categorical,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ValueKey {
    Num(u32),
    Str(String),
}

impl ValueKey {
    pub fn from_f32(x: f32) -> Self {
        ValueKey::Num(f32_to_bits(x))
    }

    fn as_num_bits(&self) -> Option<u32> {
        match self {
            ValueKey::Num(b) => Some(*b),
            _ => None,
        }
    }
}

fn f32_to_bits(x: f32) -> u32 {
    if x == 0.0 {
        0.0f32.to_bits()
    } else {
        x.to_bits()
    }
}

fn bits_to_f32(b: u32) -> f32 {
    f32::from_bits(b)
}

#[derive(Clone, Debug)]
pub struct NumericalBin {
    pub bottom: f32,
    pub top: f32,
    pub label: String,
}

#[derive(Clone, Debug)]
pub struct ColumnPreprocess {
    pub bin_type: BinType,
    pub min_v: f32,
    pub max_v: f32,
    pub values: HashMap<ValueKey, u64>,
    pub bins: HashMap<ValueKey, u64>,
    pub bin_labels: Vec<String>,
    pub numerical_bins: Vec<NumericalBin>,
}

impl ColumnPreprocess {
    fn new_numerical(min_v: f32, max_v: f32) -> Self {
        Self {
            bin_type: BinType::Numerical,
            min_v,
            max_v,
            values: HashMap::new(),
            bins: HashMap::new(),
            bin_labels: Vec::new(),
            numerical_bins: Vec::new(),
        }
    }

    fn new_categorical() -> Self {
        Self {
            bin_type: BinType::Categorical,
            min_v: 0.0,
            max_v: 0.0,
            values: HashMap::new(),
            bins: HashMap::new(),
            bin_labels: Vec::new(),
            numerical_bins: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BinDepth {
    pub main: usize,
    pub per_column: HashMap<usize, usize>,
}

impl BinDepth {
    pub fn new(main: usize) -> Self {
        Self {
            main,
            per_column: HashMap::new(),
        }
    }

    fn depth_for(&self, col: usize) -> usize {
        self.per_column.get(&col).copied().unwrap_or(self.main)
    }
}

#[derive(Clone, Debug)]
pub struct PreprocessStream {
    pub num_chunks: u64,
    pub col_graph: ColGraph,
    pub cols: Vec<usize>,
    pub preprocess_map: HashMap<usize, ColumnPreprocess>,
    finished: bool,
}

impl PreprocessStream {
    pub fn new(col_graph: ColGraph) -> Self {
        let cols = col_graph.active_indices();
        Self {
            num_chunks: 0,
            col_graph,
            cols,
            preprocess_map: HashMap::new(),
            finished: false,
        }
    }

    pub fn preprocess(&mut self, chunk: &ChunkTable) -> Result<(), String> {
        if self.finished {
            return Err("preprocess called after finish_map".into());
        }
        chunk.validate()?;
        if self.num_chunks == 0 {
            self.initialize_preprocess_map(chunk)?;
        } else {
            self.update_preprocess_map(chunk)?;
        }
        self.num_chunks += 1;
        Ok(())
    }

    fn column_is_numeric(col: &ColumnVec) -> bool {
        matches!(col, ColumnVec::F32(_))
    }

    fn initialize_preprocess_map(&mut self, chunk: &ChunkTable) -> Result<(), String> {
        self.preprocess_map.clear();
        for &col in &self.cols {
            let c = chunk
                .cols
                .get(col)
                .ok_or_else(|| format!("missing column index {col}"))?;
            if Self::column_is_numeric(c) {
                let arr = Self::col_as_f32(c)?;
                let arr = round_to_significant_figures(&Array1::from(arr), 4);
                let arr = nan_to_zero_f32(arr.to_vec());
                let min_v = arr.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_v = arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut cp = ColumnPreprocess::new_numerical(min_v, max_v);
                merge_uniques_numerical(&mut cp.values, &arr);
                self.preprocess_map.insert(col, cp);
            } else {
                let mut s = Self::col_as_utf8(c)?;
                for t in &mut s {
                    if t == "nan" || t.eq_ignore_ascii_case("nan") {
                        *t = "empty".to_string();
                    }
                }
                let mut cp = ColumnPreprocess::new_categorical();
                merge_uniques_categorical(&mut cp.values, &s);
                self.preprocess_map.insert(col, cp);
            }
        }
        Ok(())
    }

    fn update_preprocess_map(&mut self, chunk: &ChunkTable) -> Result<(), String> {
        for &col in &self.cols {
            let c = chunk
                .cols
                .get(col)
                .ok_or_else(|| format!("missing column index {col}"))?;
            let cp = self
                .preprocess_map
                .get_mut(&col)
                .ok_or_else(|| format!("preprocess_map missing col {col}"))?;
            match cp.bin_type {
                BinType::Numerical => {
                    let arr = Self::col_as_f32(c)?;
                    let arr = round_to_significant_figures(&Array1::from(arr), 4);
                    let arr = nan_to_zero_f32(arr.to_vec());
                    let min_v = arr.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max_v = arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    if min_v < cp.min_v {
                        cp.min_v = min_v;
                    }
                    if max_v > cp.max_v {
                        cp.max_v = max_v;
                    }
                    merge_uniques_numerical(&mut cp.values, &arr);
                }
                BinType::Categorical => {
                    let mut s = Self::col_as_utf8(c)?;
                    for t in &mut s {
                        if t == "nan" || t.eq_ignore_ascii_case("nan") {
                            *t = "empty".to_string();
                        }
                    }
                    merge_uniques_categorical(&mut cp.values, &s);
                }
            }
        }
        Ok(())
    }

    fn col_as_f32(c: &ColumnVec) -> Result<Vec<f32>, String> {
        match c {
            ColumnVec::F32(v) => Ok(v.clone()),
            ColumnVec::Utf8(_) => Err("expected numeric column".into()),
        }
    }

    fn col_as_utf8(c: &ColumnVec) -> Result<Vec<String>, String> {
        match c {
            ColumnVec::Utf8(v) => Ok(v.clone()),
            ColumnVec::F32(v) => Ok(v.iter().map(|x| x.to_string()).collect()),
        }
    }

    pub fn finish_map(&mut self, depth: &BinDepth) -> Result<(), String> {
        for (&col, cp) in self.preprocess_map.iter_mut() {
            let bin_depth = depth.depth_for(col);
            cp.values.remove(&ValueKey::Str("nan".into()));

            let items: Vec<(ValueKey, u64)> =
                cp.values.iter().map(|(k, v)| (k.clone(), *v)).collect();
            let selected = return_accounted_bins(&items, bin_depth);
            let mut bins_map: HashMap<ValueKey, u64> = HashMap::new();
            for (k, c) in &selected {
                bins_map.insert(k.clone(), *c);
            }

            let mut bins_sorted: Vec<ValueKey> = bins_map.keys().cloned().collect();
            bins_sorted.sort_by(key_cmp_for_sort);

            match cp.bin_type {
                BinType::Numerical => {
                    let mut nums: Vec<f32> = bins_sorted
                        .iter()
                        .filter_map(|k| k.as_num_bits().map(bits_to_f32))
                        .collect();
                    if nums.is_empty() {
                        cp.bins = bins_map;
                        cp.bin_labels = Vec::new();
                        cp.numerical_bins = Vec::new();
                        continue;
                    }
                    nums.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                    let mut bins_sorted_f = nums;

                    if cp.min_v < bins_sorted_f[0] {
                        bins_sorted_f.insert(0, cp.min_v);
                    }
                    if cp.max_v > *bins_sorted_f.last().unwrap() {
                        let last = bins_sorted_f.len() - 1;
                        bins_sorted_f[last] = cp.max_v;
                    }
                    bins_sorted_f
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                    let mut nb: Vec<NumericalBin> = Vec::new();
                    for k in 0..bins_sorted_f.len().saturating_sub(1) {
                        let bottom = bins_sorted_f[k];
                        let top = bins_sorted_f[k + 1];
                        let label = format_bin_label_bottom_top(bottom, top, bottom);
                        nb.push(NumericalBin { bottom, top, label });
                    }
                    cp.bin_labels = nb.iter().map(|x| x.label.clone()).collect();
                    cp.numerical_bins = nb;
                }
                BinType::Categorical => {
                    cp.bin_labels = bins_sorted
                        .iter()
                        .filter_map(|k| match k {
                            ValueKey::Str(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect();
                    cp.numerical_bins = Vec::new();
                }
            }

            cp.bins = bins_map;
        }
        self.finished = true;
        Ok(())
    }

    pub fn use_map(&self, chunk: &ChunkTable) -> Result<HashMap<String, ColumnVec>, String> {
        if !self.finished {
            return Err("finish_map must be called before use_map".into());
        }
        chunk.validate()?;
        let mut out = HashMap::new();
        for col_idx in 0..chunk.names.len() {
            let name = chunk.names[col_idx].clone();
            let col = chunk.cols.get(col_idx).unwrap();
            if let Some(cp) = self.preprocess_map.get(&col_idx) {
                match cp.bin_type {
                    BinType::Numerical => {
                        let mut arr = Self::col_as_f32(col)?;
                        for x in &mut arr {
                            if x.is_nan() {
                                *x = 0.0;
                            }
                        }
                        let labels = vectorized_map_numerical_bins(cp, &arr);
                        out.insert(name, ColumnVec::Utf8(labels));
                    }
                    BinType::Categorical => {
                        let mut s = Self::col_as_utf8(col)?;
                        for t in &mut s {
                            if t == "nan" || t.eq_ignore_ascii_case("nan") {
                                *t = "empty".to_string();
                            }
                        }
                        let labels: Vec<String> = s
                            .into_iter()
                            .map(|val| map_categorical_bins(cp, &val))
                            .collect();
                        out.insert(name, ColumnVec::Utf8(labels));
                    }
                }
            } else {
                out.insert(name, col.clone());
            }
        }
        Ok(out)
    }
}

fn key_cmp_for_sort(a: &ValueKey, b: &ValueKey) -> std::cmp::Ordering {
    match (a, b) {
        (ValueKey::Num(x), ValueKey::Num(y)) => bits_to_f32(*x)
            .partial_cmp(&bits_to_f32(*y))
            .unwrap_or(std::cmp::Ordering::Equal),
        (ValueKey::Str(x), ValueKey::Str(y)) => x.cmp(y),
        (ValueKey::Num(_), ValueKey::Str(_)) => std::cmp::Ordering::Less,
        (ValueKey::Str(_), ValueKey::Num(_)) => std::cmp::Ordering::Greater,
    }
}

fn merge_uniques_numerical(acc: &mut HashMap<ValueKey, u64>, arr: &[f32]) {
    let mut local: HashMap<ValueKey, u64> = HashMap::new();
    for &x in arr {
        let k = ValueKey::from_f32(x);
        *local.entry(k).or_insert(0) += 1;
    }
    for (k, c) in local {
        *acc.entry(k).or_insert(0) += c;
    }
}

fn merge_uniques_categorical(acc: &mut HashMap<ValueKey, u64>, arr: &[String]) {
    for t in arr {
        let k = ValueKey::Str(t.clone());
        *acc.entry(k).or_insert(0) += 1;
    }
}

fn nan_to_zero_f32(mut v: Vec<f32>) -> Vec<f32> {
    for x in &mut v {
        if x.is_nan() {
            *x = 0.0;
        }
    }
    v
}

fn return_accounted_bins(items: &[(ValueKey, u64)], bin_depth: usize) -> Vec<(ValueKey, u64)> {
    if items.is_empty() || bin_depth == 0 {
        return Vec::new();
    }
    let mut sorted: Vec<(ValueKey, u64)> = items.to_vec();
    sorted.sort_by_key(|b| std::cmp::Reverse(b.1));

    let mut selected: Vec<(ValueKey, u64)> = sorted.iter().take(bin_depth).cloned().collect();

    if bin_depth > sorted.len() {
        return selected;
    }

    let last_count = sorted[bin_depth - 1].1;
    let mut iterater = bin_depth;
    while iterater < sorted.len() && sorted[iterater].1 == last_count {
        selected = sorted.iter().take(iterater).cloned().collect();
        iterater += 1;
    }

    selected
}

fn format_bin_label_bottom_top(_bottom: f32, top: f32, bin_val: f32) -> String {
    let bin_1 = if bin_val.abs() < 10000.0 && bin_val.abs() > 0.1 {
        format!("{:.4}", bin_val)
    } else {
        format!("{:.4e}", bin_val)
    };
    let bin_2 = if top.abs() < 10000.0 && top.abs() > 0.1 {
        format!("{:.4}", top)
    } else {
        format!("{:.4e}", top)
    };
    format!("{bin_1} - {bin_2}")
}

fn vectorized_map_numerical_bins(cp: &ColumnPreprocess, arr: &[f32]) -> Vec<String> {
    if cp.numerical_bins.is_empty() {
        return vec!["other".to_string(); arr.len()];
    }
    let labels: Vec<&str> = cp.numerical_bins.iter().map(|b| b.label.as_str()).collect();

    let mut out = Vec::with_capacity(arr.len());
    for &x in arr {
        let mut idx: Option<usize> = None;
        for (i, nb) in cp.numerical_bins.iter().enumerate() {
            if x >= nb.bottom && x < nb.top {
                idx = Some(i);
                break;
            }
        }
        let label = idx
            .map(|i| labels[i].to_string())
            .unwrap_or_else(|| "other".to_string());
        out.push(label);
    }
    out
}

fn map_categorical_bins(cp: &ColumnPreprocess, val: &str) -> String {
    let k = ValueKey::Str(val.to_string());
    if cp.values.contains_key(&k) {
        val.to_string()
    } else {
        "other".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accounted_bins_ties_keep_extending() {
        let items = vec![
            (ValueKey::Str("a".into()), 10),
            (ValueKey::Str("b".into()), 8),
            (ValueKey::Str("c".into()), 8),
            (ValueKey::Str("d".into()), 8),
            (ValueKey::Str("e".into()), 3),
        ];
        let r = return_accounted_bins(&items, 2);
        assert!(r.len() >= 2);
    }
}
