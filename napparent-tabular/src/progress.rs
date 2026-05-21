//! Optional stderr progress reporting for long pipeline runs.

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::io::IsTerminal;
use std::time::{Duration, Instant};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProgressDisplay {
    Off,
    Bar,
    Lines,
}

pub(crate) struct ProgressReporter {
    mode: ProgressDisplay,
    bar: Option<ProgressBar>,
}

impl ProgressReporter {
    pub fn from_verbose(verbose: bool) -> Self {
        let mode = if !verbose {
            ProgressDisplay::Off
        } else if std::io::stderr().is_terminal() {
            ProgressDisplay::Bar
        } else {
            ProgressDisplay::Lines
        };
        Self { mode, bar: None }
    }

    #[cfg(test)]
    pub fn from_display(mode: ProgressDisplay) -> Self {
        Self { mode, bar: None }
    }

    #[cfg(test)]
    pub fn display_mode(&self) -> ProgressDisplay {
        self.mode
    }

    pub fn log(&self, msg: &str) {
        if self.mode == ProgressDisplay::Off {
            return;
        }
        if let Some(bar) = &self.bar {
            bar.println(format!("napparent: {msg}"));
        } else {
            eprintln!("napparent: {msg}");
        }
    }

    pub fn pass_start(&mut self, pass: u8, total_passes: u8, label: &str, batch_count: usize) {
        self.finish_bar();
        match self.mode {
            ProgressDisplay::Off => {}
            ProgressDisplay::Lines => {
                eprintln!("napparent: pass {pass}/{total_passes}: {label}");
            }
            ProgressDisplay::Bar => {
                let bar = ProgressBar::new(batch_count as u64);
                bar.set_draw_target(ProgressDrawTarget::stderr());
                bar.set_style(
                    ProgressStyle::with_template(
                        "napparent pass {prefix:.bold} [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {msg}",
                    )
                    .expect("progress template")
                    .progress_chars("█▓░"),
                );
                bar.set_prefix(format!("{pass}/{total_passes} {label}"));
                self.bar = Some(bar);
            }
        }
    }

    pub fn batch_tick(&mut self, index: usize, total: usize, rows: usize, line_msg: &str) {
        match self.mode {
            ProgressDisplay::Off => {}
            ProgressDisplay::Bar => {
                if let Some(bar) = &self.bar {
                    bar.set_position((index + 1) as u64);
                    bar.set_message(format!("batch {}/{} ({} rows)", index + 1, total, rows));
                }
            }
            ProgressDisplay::Lines => {
                if should_log_batch(index, total) {
                    eprintln!("napparent: {line_msg}");
                }
            }
        }
    }

    pub fn pass_finish(&mut self, msg: &str) {
        self.finish_bar();
        if self.mode != ProgressDisplay::Off {
            eprintln!("napparent: {msg}");
        }
    }

    pub fn finish(&mut self, msg: &str) {
        self.finish_bar();
        if self.mode != ProgressDisplay::Off {
            eprintln!("napparent: {msg}");
        }
    }

    pub fn abandon(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.abandon_with_message("interrupted");
        }
    }

    fn finish_bar(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
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
    fn verbose_off_is_off_mode() {
        let r = ProgressReporter::from_verbose(false);
        assert_eq!(r.display_mode(), ProgressDisplay::Off);
    }

    #[test]
    fn from_display_lines() {
        let r = ProgressReporter::from_display(ProgressDisplay::Lines);
        assert_eq!(r.display_mode(), ProgressDisplay::Lines);
    }

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
        assert!(!should_log_batch(1, total) || 1_usize.is_multiple_of(step));
    }
}
