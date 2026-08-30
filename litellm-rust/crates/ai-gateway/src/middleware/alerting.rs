//! Alerting middleware for monitoring and alerting.
//!
//! Monitors error rates, latency, and other metrics, and sends alerts
//! when thresholds are exceeded.

use crate::alerting::{Alert, AlertManager, AlertSeverity, AlertType, AlertingConfig};
use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Alerting middleware state.
pub struct AlertingState {
    alert_manager: Arc<AlertManager>,
    request_count: Arc<Mutex<u64>>,
    error_count: Arc<Mutex<u64>>,
    last_reset: Arc<Mutex<Instant>>,
}

impl AlertingState {
    /// Create a new alerting state.
    pub fn new(config: AlertingConfig) -> Self {
        Self {
            alert_manager: Arc::new(AlertManager::new(config)),
            request_count: Arc::new(Mutex::new(0)),
            error_count: Arc::new(Mutex::new(0)),
            last_reset: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Record a request.
    pub async fn record_request(&self) {
        let mut count = self.request_count.lock().await;
        *count += 1;
    }

    /// Record an error.
    pub async fn record_error(&self) {
        let mut count = self.error_count.lock().await;
        *count += 1;
    }

    /// Get the error rate.
    pub async fn get_error_rate(&self) -> f64 {
        let request_count = *self.request_count.lock().await;
        let error_count = *self.error_count.lock().await;

        if request_count == 0 {
            0.0
        } else {
            error_count as f64 / request_count as f64
        }
    }

    /// Reset counters.
    pub async fn reset_counters(&self) {
        let mut request_count = self.request_count.lock().await;
        let mut error_count = self.error_count.lock().await;
        let mut last_reset = self.last_reset.lock().await;

        *request_count = 0;
        *error_count = 0;
        *last_reset = Instant::now();
    }

    /// Check thresholds and send alerts if needed.
    pub async fn check_thresholds(&self, error_rate_threshold: f64, latency_threshold_ms: u64) {
        let error_rate = self.get_error_rate().await;

        if error_rate > error_rate_threshold {
            let alert = Alert::new(
                AlertType::HighErrorRate,
                AlertSeverity::Critical,
                "High Error Rate Detected".to_string(),
                format!(
                    "Error rate {:.2}% exceeds threshold {:.2}%",
                    error_rate * 100.0,
                    error_rate_threshold * 100.0
                ),
            );
            let _ = self.alert_manager.send_alert(alert).await;
        }
    }
}

/// Alerting middleware that monitors metrics and sends alerts.
pub async fn alerting_middleware(
    State(state): State<Arc<AlertingState>>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // Record the request
    state.record_request().await;

    let start = Instant::now();

    // Process the request
    let response = next.run(request).await;

    let latency = start.elapsed();

    // Check if it was an error
    if response.status().is_server_error() {
        state.record_error().await;
    }

    // Check thresholds periodically (every 100 requests)
    let request_count = *state.request_count.lock().await;
    if request_count % 100 == 0 {
        state.check_thresholds(0.05, 5000).await; // 5% error rate, 5s latency
        state.reset_counters().await;
    }

    Ok(response)
}

/// Alerting middleware that extracts AlertingState from AppState.
/// For use in the production router where AppState is the router state.
pub async fn alerting_middleware_from_app_state(
    State(state): State<crate::state::AppState>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    state.alerting_state.record_request().await;

    let start = Instant::now();
    let response = next.run(request).await;
    let _latency = start.elapsed();

    if response.status().is_server_error() {
        state.alerting_state.record_error().await;
    }

    let request_count = *state.alerting_state.request_count.lock().await;
    if request_count % 100 == 0 && request_count > 0 {
        state.alerting_state.check_thresholds(0.05, 5000).await;
        state.alerting_state.reset_counters().await;
    }

    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_alerting_state_creation() {
        let config = AlertingConfig::default();
        let state = AlertingState::new(config);

        let error_rate = state.get_error_rate().await;
        assert_eq!(error_rate, 0.0);
    }

    #[tokio::test]
    async fn test_record_request_and_error() {
        let config = AlertingConfig::default();
        let state = AlertingState::new(config);

        state.record_request().await;
        state.record_request().await;
        state.record_error().await;

        let error_rate = state.get_error_rate().await;
        assert_eq!(error_rate, 0.5); // 1 error out of 2 requests
    }

    #[tokio::test]
    async fn test_reset_counters() {
        let config = AlertingConfig::default();
        let state = AlertingState::new(config);

        state.record_request().await;
        state.record_error().await;
        state.reset_counters().await;

        let error_rate = state.get_error_rate().await;
        assert_eq!(error_rate, 0.0);
    }
}
