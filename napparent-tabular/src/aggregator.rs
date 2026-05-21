//! Column-pair aggregation for effect features.

use crate::activation::{ActivationConfig, EffectContext, PairStats};
use crate::arrow_io::OutcomesRef;
use crate::table::{ColGraph, ColumnVec};
use ndarray::{Array1, Array2};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

#[inline]
fn canonical_val_pair(a: i32, b: i32) -> (i32, i32) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[derive(Clone, Debug)]
pub struct PairAggregator {
    pub target: String,
    pub num_chunks: u64,
    avg_outcome_sum: f32,
    pub avg_count: u64,
    k: i32,
    col_map_int: HashMap<i32, String>,
    col_map_str: HashMap<String, i32>,
    val_map_int: HashMap<i32, String>,
    pub val_map_str: HashMap<String, i32>,
    col_graph_names: Vec<String>,
    cols_dropped: HashSet<usize>,
    pub cols: Vec<usize>,
    col_names_ordered: Vec<String>,
    vals_map: HashMap<(i32, i32), [f32; 2]>,
    pub vals_map_avg: HashMap<(i32, i32), f32>,
    pub avg_outcome: f32,
    pub col_array: Vec<usize>,
    tup_combos: HashMap<usize, (usize, usize)>,
    col_to_tup: HashMap<usize, Vec<(usize, usize)>>,
    m_divisor: f32,
    combos_initialized: bool,
    activation: ActivationConfig,
}

impl PairAggregator {
    pub fn new() -> Self {
        Self {
            target: String::new(),
            num_chunks: 0,
            avg_outcome_sum: 0.0,
            avg_count: 0,
            k: 0,
            col_map_int: HashMap::new(),
            col_map_str: HashMap::new(),
            val_map_int: HashMap::new(),
            val_map_str: HashMap::new(),
            col_graph_names: Vec::new(),
            cols_dropped: HashSet::new(),
            cols: Vec::new(),
            col_names_ordered: Vec::new(),
            vals_map: HashMap::new(),
            vals_map_avg: HashMap::new(),
            avg_outcome: 0.0,
            col_array: Vec::new(),
            tup_combos: HashMap::new(),
            col_to_tup: HashMap::new(),
            m_divisor: 1.0,
            combos_initialized: false,
            activation: ActivationConfig::default(),
        }
    }

    pub fn with_activation(activation: ActivationConfig) -> Self {
        Self {
            activation,
            ..Self::new()
        }
    }

    pub fn vals_map_len(&self) -> usize {
        self.vals_map.len()
    }

    pub fn initialize_inputs(
        &mut self,
        col_info: &ColGraph,
        target: &str,
        column_order: &[String],
    ) -> Result<(), String> {
        self.target = target.to_string();
        self.num_chunks = 0;
        self.avg_outcome_sum = 0.0;
        self.avg_count = 0;
        self.k = 0;
        self.col_map_int.clear();
        self.col_map_str.clear();
        self.val_map_int.clear();
        self.val_map_str.clear();
        self.col_graph_names = col_info.names.clone();
        self.cols_dropped = col_info.dropped.clone();
        self.cols = col_info.active_indices();
        self.col_names_ordered = column_order.to_vec();
        self.combos_initialized = false;
        self.col_array.clear();
        self.tup_combos.clear();
        self.col_to_tup.clear();

        for name in column_order {
            let s = name.clone();
            if !self.col_map_str.contains_key(&s) {
                let id = self.k;
                self.col_map_int.insert(id, s.clone());
                self.col_map_str.insert(s, id);
                self.k += 1;
            }
        }
        Ok(())
    }

    pub fn make_col_combos(&mut self) {
        if self.combos_initialized {
            return;
        }
        self.vals_map.clear();
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        let idxs = &self.cols;
        for i in 0..idxs.len() {
            for j in (i + 1)..idxs.len() {
                pairs.push((idxs[i], idxs[j]));
            }
        }
        for (k, &(a, b)) in pairs.iter().enumerate() {
            self.col_array.push(k);
            self.tup_combos.insert(k, (a, b));
        }
        let mut col_to_tup: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (&_c, &(a, b)) in self.tup_combos.iter() {
            let srt = if a < b { (a, b) } else { (b, a) };
            col_to_tup.entry(a).or_default().push(srt);
            col_to_tup.entry(b).or_default().push(srt);
        }
        self.col_to_tup = col_to_tup
            .into_iter()
            .map(|(c, mut v)| {
                v.sort();
                v.dedup();
                (c, v)
            })
            .collect();

        self.m_divisor = (self.col_to_tup.len() as f32 - 1.0).max(1.0);
        self.combos_initialized = true;
    }

    fn ensure_sentinel_values(&mut self) {
        if self.num_chunks == 0 {
            if !self.val_map_str.contains_key("no data") {
                let id = self.k;
                self.val_map_int.insert(id, "no data".into());
                self.val_map_str.insert("no data".into(), id);
                self.k += 1;
            }
            if !self.val_map_str.contains_key("None") {
                let id = self.k;
                self.val_map_int.insert(id, "None".into());
                self.val_map_str.insert("None".into(), id);
                self.k += 1;
            }
        }
    }

    fn column_vec_to_labels(col: &ColumnVec) -> Result<Vec<String>, String> {
        match col {
            ColumnVec::Utf8(v) => Ok(v
                .iter()
                .map(|s| {
                    let t = s.as_str();
                    if t.is_empty() {
                        "no data".to_string()
                    } else {
                        t.to_string()
                    }
                })
                .collect()),
            ColumnVec::F32(v) => Ok(v
                .iter()
                .map(|&x| {
                    if x.is_finite() {
                        x.to_string()
                    } else {
                        "0".to_string()
                    }
                })
                .collect()),
            ColumnVec::F32Array(v) => Ok(v
                .iter()
                .map(|&x| {
                    if x.is_finite() {
                        x.to_string()
                    } else {
                        "0".to_string()
                    }
                })
                .collect()),
        }
    }

    fn register_column_values(&mut self, labels: &[String]) -> Result<(), String> {
        self.ensure_sentinel_values();
        let mut uniq: Vec<&String> = labels.iter().collect();
        uniq.sort();
        uniq.dedup();
        for val in uniq {
            if let Entry::Vacant(e) = self.val_map_str.entry(val.clone()) {
                let id = self.k;
                self.val_map_int.insert(id, val.clone());
                e.insert(id);
                self.k += 1;
            }
        }
        Ok(())
    }

    fn column_to_ids(&self, labels: &[String]) -> Result<Array1<i32>, String> {
        let mut v = Vec::with_capacity(labels.len());
        for val in labels {
            let id = self
                .val_map_str
                .get(val)
                .copied()
                .ok_or_else(|| format!("unknown val {val}"))?;
            v.push(id);
        }
        Ok(Array1::from(v))
    }

    fn x_processed_to_mapped(
        &mut self,
        x: &HashMap<String, ColumnVec>,
    ) -> Result<HashMap<usize, Array1<i32>>, String> {
        let n = x
            .get(&self.col_graph_names[0])
            .map(|c| c.len())
            .ok_or_else(|| "missing first col".to_string())?;
        let col_indices: Vec<usize> = self.cols.clone();
        let mut out: HashMap<usize, Array1<i32>> = HashMap::new();
        for col_idx in col_indices {
            let name = self.col_graph_names[col_idx].clone();
            let col = x
                .get(&name)
                .ok_or_else(|| format!("missing col {name}"))?;
            let labels = Self::column_vec_to_labels(col)?;
            if labels.len() != n {
                return Err("column length mismatch in x_processed_to_mapped".into());
            }
            self.register_column_values(&labels)?;
            let ids = self.column_to_ids(&labels)?;
            out.insert(col_idx, ids);
        }
        Ok(out)
    }

    pub fn vals_map_updating(
        &mut self,
        x_processed: &HashMap<String, ColumnVec>,
        outcomes: &OutcomesRef<'_>,
    ) -> Result<(), String> {
        let x_mapped = self.x_processed_to_mapped(x_processed)?;

        self.avg_outcome_sum += outcomes.sum();
        self.avg_count += outcomes.len() as u64;

        let n = outcomes.len();
        let one_percent = ((n as f32) * 0.01).floor() as usize;

        for &c in &self.col_array {
            let &(c1, c2) = self.tup_combos.get(&c).unwrap();
            let a = x_mapped.get(&c1).unwrap();
            let b = x_mapped.get(&c2).unwrap();

            let mut local: HashMap<(i32, i32), PairStats> = HashMap::new();
            for i in 0..n {
                let key = canonical_val_pair(a[i], b[i]);
                let out_i = {
                    let x = outcomes.get(i);
                    if x.is_nan() { 0.0 } else { x }
                };
                let entry = local.entry(key).or_insert(PairStats {
                    sum: 0.0,
                    count: 0.0,
                });
                entry.sum += out_i;
                entry.count += 1.0;
            }

            for (key, stats) in local {
                if stats.count as usize > one_percent {
                    let entry = self.vals_map.entry(key).or_insert([0.0, 0.0]);
                    entry[0] += stats.sum;
                    entry[1] += stats.count;
                } else {
                    self.vals_map.entry(key).or_insert([0.0, 0.0]);
                }
            }
        }

        self.num_chunks += 1;
        Ok(())
    }

    pub fn finish_map(&mut self) {
        self.vals_map_avg.clear();
        for (&key, &arr) in &self.vals_map {
            let weight = self.activation.kg_pair.activate(PairStats {
                sum: arr[0],
                count: arr[1],
            });
            self.vals_map_avg.insert(key, weight);
        }

        if self.avg_count > 0 {
            self.avg_outcome = self.avg_outcome_sum / self.avg_count as f32;
        } else {
            self.avg_outcome = 0.0;
        }
    }

    fn make_cvto_inner(
        &mut self,
        x_processed: &HashMap<String, ColumnVec>,
    ) -> Result<Array2<f32>, String> {
        let x_mapped = self.x_processed_to_mapped(x_processed)?;
        let n = x_mapped
            .get(&self.cols[0])
            .map(|a| a.len())
            .ok_or_else(|| "no active columns".to_string())?;
        let m = self.col_array.len();
        let mut col_vals = Array2::<f32>::zeros((n, m));

        for (mi, &combo_id) in self.col_array.iter().enumerate() {
            let &(c1, c2) = self.tup_combos.get(&combo_id).unwrap();
            let a = x_mapped.get(&c1).unwrap();
            let b = x_mapped.get(&c2).unwrap();
            for i in 0..n {
                let tup = canonical_val_pair(a[i], b[i]);
                let v = *self.vals_map_avg.get(&tup).unwrap_or(&0.0);
                col_vals[[i, mi]] = v;
            }
        }
        Ok(col_vals)
    }

    pub fn use_map(
        &mut self,
        mut x_processed: HashMap<String, ColumnVec>,
        y: Vec<f32>,
        outcomes: Vec<f32>,
    ) -> Result<HashMap<String, ColumnVec>, String> {
        let col_vals_outcomes = self.make_cvto_inner(&x_processed)?;
        let n = col_vals_outcomes.nrows();

        let mut col_combined: HashMap<usize, Array1<f32>> = HashMap::new();
        for &combo_id in &self.col_array {
            let &(c1, c2) = self.tup_combos.get(&combo_id).unwrap();
            let col_view = col_vals_outcomes.column(combo_id);
            for &col_idx in &[c1, c2] {
                col_combined
                    .entry(col_idx)
                    .or_insert_with(|| Array1::zeros(n))
                    .scaled_add(1.0, &col_view);
            }
        }
        let m = self.m_divisor.max(1.0);
        for v in col_combined.values_mut() {
            *v /= m;
        }

        let mut nnm: HashMap<String, ColumnVec> = HashMap::new();
        for name in &self.col_graph_names {
            let colvec = x_processed.remove(name).unwrap_or_else(|| {
                ColumnVec::Utf8(vec!["no data".into(); n])
            });
            nnm.insert(name.clone(), colvec);
        }

        let ctx = EffectContext {
            global_mean_outcome: self.avg_outcome,
        };
        for c_idx in 0..self.col_graph_names.len() {
            if let Some(arr) = col_combined.get(&c_idx) {
                let col_name = &self.col_graph_names[c_idx];
                let diffs = Array1::from(
                    arr.iter()
                        .map(|x| self.activation.effect.activate(*x, &ctx))
                        .collect::<Vec<f32>>(),
                );
                nnm.insert(format!("{col_name}_effect"), ColumnVec::F32Array(diffs));
            }
        }

        nnm.insert("Actuals".into(), ColumnVec::F32Array(Array1::from(y)));
        nnm.insert(
            "outcomes_effect".into(),
            ColumnVec::F32Array(Array1::from(outcomes)),
        );
        Ok(nnm)
    }
}

impl Default for PairAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn two_col_aggregator() -> PairAggregator {
        let cg = ColGraph {
            names: vec!["a".into(), "b".into()],
            dropped: HashSet::new(),
        };
        let mut agg = PairAggregator::new();
        agg.initialize_inputs(&cg, "target", &["a".into(), "b".into()])
            .unwrap();
        agg.make_col_combos();
        agg
    }

    fn oriented_pair_data() -> HashMap<String, ColumnVec> {
        let mut x = HashMap::new();
        x.insert("a".into(), ColumnVec::Utf8(vec!["5".into(), "7".into()]));
        x.insert("b".into(), ColumnVec::Utf8(vec!["7".into(), "5".into()]));
        x
    }

    fn outcomes_slice(v: &[f32]) -> crate::arrow_io::OutcomesRef<'_> {
        crate::arrow_io::OutcomesRef::Slice(v)
    }

    #[test]
    fn canonical_val_pair_commutes() {
        assert_eq!(canonical_val_pair(5, 7), canonical_val_pair(7, 5));
        assert_eq!(canonical_val_pair(5, 7), (5, 7));
    }

    #[test]
    fn vals_map_merges_orientations() {
        let mut agg = two_col_aggregator();
        let x = oriented_pair_data();
        agg.vals_map_updating(&x, &outcomes_slice(&[1.0, 2.0])).unwrap();

        let id5 = agg.val_map_str["5"];
        let id7 = agg.val_map_str["7"];
        let key = canonical_val_pair(id5, id7);

        assert_eq!(agg.vals_map.len(), 1);
        let arr = agg.vals_map[&key];
        assert!((arr[0] - 3.0).abs() < 1e-5);
        assert!((arr[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn finish_map_no_duplicate_keys() {
        let mut agg = two_col_aggregator();
        let x = oriented_pair_data();
        agg.vals_map_updating(&x, &outcomes_slice(&[1.0, 2.0])).unwrap();
        agg.finish_map();
        assert_eq!(agg.vals_map_avg.len(), agg.vals_map.len());
    }

    #[test]
    fn lookup_symmetric() {
        let mut agg = two_col_aggregator();
        let x = oriented_pair_data();
        agg.vals_map_updating(&x, &outcomes_slice(&[1.0, 2.0])).unwrap();
        agg.finish_map();

        let col_vals = agg.make_cvto_inner(&x).unwrap();
        assert_eq!(col_vals.nrows(), 2);
        assert!((col_vals[[0, 0]] - col_vals[[1, 0]]).abs() < 1e-5);
    }
}
