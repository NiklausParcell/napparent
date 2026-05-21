//! Column-pair aggregation for effect features.

use crate::table::{ColGraph, ColumnVec};
use ndarray::{Array1, Array2, Axis};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

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
        }
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

    fn help_x_col_slice(col: &ColumnVec) -> Vec<String> {
        match col {
            ColumnVec::F32(v) => v
                .iter()
                .map(|&x| {
                    if x.is_finite() {
                        x.to_string()
                    } else {
                        "0".to_string()
                    }
                })
                .collect(),
            ColumnVec::Utf8(v) => v
                .iter()
                .map(|s| {
                    let t = s.as_str();
                    if t.is_empty() {
                        "no data".to_string()
                    } else {
                        t.to_string()
                    }
                })
                .collect(),
        }
    }

    fn convert_ff_to_string_matrix(
        &self,
        x: &HashMap<String, ColumnVec>,
    ) -> Result<Vec<Vec<String>>, String> {
        let n = x
            .get(&self.col_graph_names[0])
            .map(|c| c.len())
            .ok_or_else(|| "missing first col".to_string())?;
        let m = self.col_graph_names.len();
        let mut mat = vec![vec![String::new(); m]; n];
        for (j, name) in self.col_graph_names.iter().enumerate() {
            let col = x.get(name).ok_or_else(|| format!("missing col {name}"))?;
            let mapped = Self::help_x_col_slice(col);
            if mapped.len() != n {
                return Err("column length mismatch in convert_ff".into());
            }
            for i in 0..n {
                mat[i][j] = mapped[i].clone();
            }
        }
        Ok(mat)
    }

    fn val_checking(&mut self, mat: &[Vec<String>]) -> Result<(), String> {
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
        for &col in &self.cols {
            let mut uniq: Vec<String> = mat.iter().map(|row| row[col].clone()).collect();
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
        }
        Ok(())
    }

    fn x_to_vec_mapped(&self, mat: &[Vec<String>]) -> Result<HashMap<usize, Array1<i32>>, String> {
        let n = mat.len();
        let mut out: HashMap<usize, Array1<i32>> = HashMap::new();
        for &col in &self.cols {
            let mut v = Vec::with_capacity(n);
            for row in mat.iter().take(n) {
                let id = self
                    .val_map_str
                    .get(&row[col])
                    .copied()
                    .ok_or_else(|| format!("unknown val {}", row[col]))?;
                v.push(id);
            }
            out.insert(col, Array1::from(v));
        }
        Ok(out)
    }

    pub fn vals_map_updating(
        &mut self,
        x_processed: &HashMap<String, ColumnVec>,
        outcomes: &[f32],
    ) -> Result<(), String> {
        let mat = self.convert_ff_to_string_matrix(x_processed)?;
        self.val_checking(&mat)?;
        let x_mapped = self.x_to_vec_mapped(&mat)?;
        let outs = Array1::from(outcomes.to_vec());

        self.avg_outcome_sum += outs.sum();
        self.avg_count += outs.len() as u64;

        let n = outs.len();
        let one_percent = ((n as f32) * 0.01).floor() as usize;

        for &c in &self.col_array {
            let &(c1, c2) = self.tup_combos.get(&c).unwrap();
            let a = x_mapped.get(&c1).unwrap();
            let b = x_mapped.get(&c2).unwrap();
            let stacked =
                ndarray::stack(Axis(1), &[a.view(), b.view()]).map_err(|e| e.to_string())?;
            let rows: Vec<[i32; 2]> = stacked.rows().into_iter().map(|r| [r[0], r[1]]).collect();

            let mut freq: HashMap<(i32, i32), usize> = HashMap::new();
            for r in &rows {
                let k = (r[0], r[1]);
                *freq.entry(k).or_insert(0) += 1;
            }

            let mut frequent: Vec<(i32, i32)> = Vec::new();
            let mut rare: Vec<(i32, i32)> = Vec::new();
            for (k, cnt) in freq {
                if cnt > one_percent {
                    frequent.push(k);
                } else {
                    rare.push(k);
                }
            }

            for (v0, v1) in frequent {
                let unique_arr = [v0, v1];
                let unique_inv = [v1, v0];
                let unique_tup = (v0, v1);
                let unique_inv_tup = (v1, v0);

                let mut sum_direct = 0.0_f32;
                let mut sum_inv = 0.0_f32;
                let mut count_direct = 0_usize;
                let mut count_inv = 0_usize;
                for (i, r) in rows.iter().enumerate() {
                    if *r == unique_arr {
                        sum_direct += outs[i];
                        count_direct += 1;
                    }
                    if *r == unique_inv {
                        sum_inv += outs[i];
                        count_inv += 1;
                    }
                }

                let n_f = count_direct as f32;
                let n_inv = count_inv as f32;
                self.vals_map.entry(unique_tup).or_insert([0.0, 0.0])[0] += sum_direct;
                self.vals_map.entry(unique_inv_tup).or_insert([0.0, 0.0])[0] += sum_inv;
                self.vals_map.entry(unique_tup).or_insert([0.0, 0.0])[1] += n_f;
                self.vals_map.entry(unique_inv_tup).or_insert([0.0, 0.0])[1] += n_inv;
            }

            for (v0, v1) in rare {
                let unique_tup = (v0, v1);
                let unique_inv_tup = (v1, v0);
                self.vals_map.entry(unique_tup).or_insert([0.0, 0.0]);
                self.vals_map.entry(unique_inv_tup).or_insert([0.0, 0.0]);
            }
        }

        self.num_chunks += 1;
        Ok(())
    }

    pub fn finish_map(&mut self) {
        self.vals_map_avg.clear();
        let keys: Vec<(i32, i32)> = self.vals_map.keys().copied().collect();
        for val_tup in keys {
            let arr = self.vals_map[&val_tup];
            let (v0, v1) = val_tup;
            let count0 = arr[1];
            let div0 = if count0 <= 1.0 {
                0.0
            } else {
                (arr[0] / count0) * count0.log10()
            };
            self.vals_map_avg.insert(val_tup, div0);

            let inv = (v1, v0);
            let arr_inv = self.vals_map.get(&inv).copied().unwrap_or([0.0, 0.0]);
            let count1 = arr_inv[1];
            let div1 = if count1 <= 1.0 {
                0.0
            } else {
                (arr_inv[0] / count1) * count1.log10()
            };
            self.vals_map_avg.insert(inv, div1);
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
        let mat = self.convert_ff_to_string_matrix(x_processed)?;
        let n = mat.len();
        let m = self.col_array.len();
        let mut col_vals = Array2::<f32>::zeros((n, m));

        self.val_checking(&mat)?;
        let x_mapped = self.x_to_vec_mapped(&mat)?;

        for (mi, &combo_id) in self.col_array.iter().enumerate() {
            let &(c1, c2) = self.tup_combos.get(&combo_id).unwrap();
            let a = x_mapped.get(&c1).unwrap();
            let b = x_mapped.get(&c2).unwrap();
            for i in 0..n {
                let tup = (a[i], b[i]);
                let v = *self.vals_map_avg.get(&tup).unwrap_or(&0.0);
                col_vals[[i, mi]] = v;
            }
        }
        Ok(col_vals)
    }

    pub fn use_map(
        &mut self,
        x_processed: &HashMap<String, ColumnVec>,
        y: &[f32],
        outcomes: &[f32],
    ) -> Result<HashMap<String, ColumnVec>, String> {
        let col_vals_outcomes = self.make_cvto_inner(x_processed)?;
        let n = col_vals_outcomes.nrows();

        let mut col_combined: HashMap<usize, Array1<f32>> = HashMap::new();
        for &c in &self.col_array {
            let combo = self.tup_combos[&c];
            let c1 = combo.0;
            let c2 = combo.1;
            let col_vec = col_vals_outcomes.column(c).to_owned();
            col_combined
                .entry(c1)
                .and_modify(|e| *e += &col_vec)
                .or_insert_with(|| col_vec.clone());
            col_combined
                .entry(c2)
                .and_modify(|e| *e += &col_vec)
                .or_insert(col_vec);
        }
        let m = self.m_divisor.max(1.0);
        for v in col_combined.values_mut() {
            *v /= m;
        }

        let mut nnm: HashMap<String, ColumnVec> = HashMap::new();
        for name in &self.col_graph_names {
            let colvec = x_processed
                .get(name)
                .cloned()
                .unwrap_or_else(|| ColumnVec::Utf8(vec!["no data".into(); n]));
            nnm.insert(name.clone(), colvec);
        }

        for c_idx in 0..self.col_graph_names.len() {
            if let Some(arr) = col_combined.get(&c_idx) {
                let col_name = self.col_graph_names[c_idx].clone();
                let diffs: Vec<f32> = arr.iter().map(|x| x - self.avg_outcome).collect();
                nnm.insert(format!("{col_name}_effect"), ColumnVec::F32(diffs));
            }
        }

        nnm.insert("Actuals".into(), ColumnVec::F32(y.to_vec()));
        nnm.insert("outcomes_effect".into(), ColumnVec::F32(outcomes.to_vec()));
        Ok(nnm)
    }
}

impl Default for PairAggregator {
    fn default() -> Self {
        Self::new()
    }
}
