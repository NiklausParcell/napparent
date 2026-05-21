//! Optional stderr progress logging for long pipeline runs.

use std::time::{Duration, Instant};

pub(crate) fn progress_log(verbose: bool, msg: &str) {
    if verbose {
        eprintln!("napparent: {msg}");
    }
}

/// Log batch `index` (0-based) of `total` when verbose and throttling allows.
pub(crate) fn progress_batch(verbose: bool, index: usize, total: usize, msg: &str) {
    if verbose && should_log_batch(index, total) {
        eprintln!("napparent: {msg}");
    }
}

/// Throttle per-batch logs when there are many chunks (~5% steps + last batch).
pub(crate) fn should_log_batch(index: usize, total: usize) -> bool {
    if total <= 50 {
        return true;
    }
    let step = (total / 20).max(1);
    index.is_multiple_of(step) || index + 1 == total
}

pub(crate) struct ProgressTimer {
    start: Instant,
}

impl ProgressTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throttle_small_runs_all_batches() {
        assert!(should_log_batch(0, 10));
        assert!(should_log_batch(9, 10));
    }

    #[test]
    fn throttle_large_runs_sparse() {
        let total = 162;
        let step = (total / 20).max(1);
        assert!(should_log_batch(0, total));
        assert!(should_log_batch(total - 1, total));
        assert!(!should_log_batch(1, total) || 1 % step == 0);
    }
}
