//! The `CustomLogger` trait — the Rust mirror of Python
//! `litellm/integrations/custom_logger.py::CustomLogger`.
//!
//! Synchronous (no `async_trait`): callbacks are O(1) enqueue-and-return so the
//! realtime splice never blocks on a logger. Default bodies are no-ops so a
//! logger can implement only the events it cares about.

use crate::integrations::types::{LogError, LoggingError, StandardLoggingPayload};

pub trait CustomLogger: Send + Sync {
    /// Record a successful call. Default: no-op.
    fn log_success_event(&self, _payload: &StandardLoggingPayload) -> Result<(), LogError> {
        Ok(())
    }

    /// Record a failed call. Default: no-op.
    fn log_failure_event(
        &self,
        _payload: &StandardLoggingPayload,
        _error: &LoggingError,
    ) -> Result<(), LogError> {
        Ok(())
    }
}
