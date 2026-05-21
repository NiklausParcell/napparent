//! Cooperative cancellation checked between pipeline batches.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub const INTERRUPT_MSG: &str = "interrupted by user";

type CancelHook = Arc<dyn Fn() -> Result<(), String> + Send + Sync>;

/// Checked at the start of each batch iteration; safe to share across threads.
pub struct CancelToken {
    stop: Arc<AtomicBool>,
    hook: Option<CancelHook>,
}

impl CancelToken {
    /// Never triggers cancellation.
    pub fn none() -> Self {
        Self {
            stop: Arc::new(AtomicBool::new(false)),
            hook: None,
        }
    }

    pub fn from_flag(stop: Arc<AtomicBool>) -> Self {
        Self { stop, hook: None }
    }

    pub fn with_hook(hook: CancelHook) -> Self {
        Self {
            stop: Arc::new(AtomicBool::new(false)),
            hook: Some(hook),
        }
    }

    pub fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop)
    }

    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    pub fn check(&self) -> Result<(), String> {
        if let Some(hook) = &self.hook {
            if let Err(e) = hook() {
                self.stop.store(true, Ordering::Relaxed);
                return Err(if e.contains("interrupted") {
                    e
                } else {
                    format!("{INTERRUPT_MSG}: {e}")
                });
            }
        }
        if self.stop.load(Ordering::Relaxed) {
            return Err(INTERRUPT_MSG.into());
        }
        Ok(())
    }
}

/// Restores a no-op SIGINT handler when dropped.
pub struct CtrlcGuard;

impl CancelToken {
    /// Install a process-wide SIGINT handler that sets the token stop flag.
    pub fn with_ctrlc_handler() -> Result<(Self, CtrlcGuard), String> {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_handler = Arc::clone(&stop);
        ctrlc::set_handler(move || {
            stop_handler.store(true, Ordering::Relaxed);
        })
        .map_err(|e| format!("ctrlc handler: {e}"))?;
        Ok((Self::from_flag(stop), CtrlcGuard))
    }
}

impl Drop for CtrlcGuard {
    fn drop(&mut self) {
        let _ = ctrlc::set_handler(|| {});
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_returns_err_when_flag_set() {
        let token = CancelToken::none();
        token.request_stop();
        let err = token.check().unwrap_err();
        assert!(err.contains("interrupted"));
    }
}
