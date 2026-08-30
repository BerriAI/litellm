//! Alerting system for monitoring and notifications.
//!
//! Sends alerts via webhooks and email when certain conditions are met,
//! such as high error rates, high latency, rate limit exceeded, or circuit breaker trips.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Alert severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Alert type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertType {
    HighErrorRate,
    HighLatency,
    RateLimitExceeded,
    CircuitBreakerTripped,
    HighMemoryUsage,
    LowDiskSpace,
    Custom(String),
}

impl std::fmt::Display for AlertType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertType::HighErrorRate => write!(f, "High Error Rate"),
            AlertType::HighLatency => write!(f, "High Latency"),
            AlertType::RateLimitExceeded => write!(f, "Rate Limit Exceeded"),
            AlertType::CircuitBreakerTripped => write!(f, "Circuit Breaker Tripped"),
            AlertType::HighMemoryUsage => write!(f, "High Memory Usage"),
            AlertType::LowDiskSpace => write!(f, "Low Disk Space"),
            AlertType::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// An alert message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub title: String,
    pub message: String,
    pub timestamp: u64,
    pub metadata: std::collections::HashMap<String, String>,
}

impl Alert {
    /// Create a new alert.
    pub fn new(
        alert_type: AlertType,
        severity: AlertSeverity,
        title: String,
        message: String,
    ) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id: format!("alert_{}", timestamp),
            alert_type,
            severity,
            title,
            message,
            timestamp,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata to the alert.
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Alert channel configuration.
#[derive(Debug, Clone)]
pub enum AlertChannel {
    Webhook {
        url: String,
        headers: std::collections::HashMap<String, String>,
    },
    Email {
        to: Vec<String>,
        from: String,
        smtp_server: String,
        smtp_port: u16,
    },
}

/// Alerting configuration.
#[derive(Debug, Clone)]
pub struct AlertingConfig {
    pub channels: Vec<AlertChannel>,
    pub error_rate_threshold: f64,
    pub latency_threshold_ms: u64,
    pub cooldown_duration: Duration,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            channels: Vec::new(),
            error_rate_threshold: 0.05,                  // 5%
            latency_threshold_ms: 5000,                  // 5 seconds
            cooldown_duration: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Alert manager.
pub struct AlertManager {
    config: AlertingConfig,
    last_alert: Arc<Mutex<std::collections::HashMap<AlertType, Instant>>>,
}

impl AlertManager {
    /// Create a new alert manager.
    pub fn new(config: AlertingConfig) -> Self {
        Self {
            config,
            last_alert: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    /// Check if an alert should be sent (respecting cooldown).
    fn should_alert(&self, alert_type: &AlertType) -> bool {
        let mut last_alert = self.last_alert.lock();
        let now = Instant::now();

        if let Some(last) = last_alert.get(alert_type) {
            if now.duration_since(*last) < self.config.cooldown_duration {
                return false;
            }
        }

        last_alert.insert(alert_type.clone(), now);
        true
    }

    /// Send an alert to all configured channels.
    pub async fn send_alert(&self, alert: Alert) -> Result<(), String> {
        if !self.should_alert(&alert.alert_type) {
            return Ok(());
        }

        for channel in &self.config.channels {
            match channel {
                AlertChannel::Webhook { url, headers } => {
                    self.send_webhook_alert(url, headers, &alert).await?;
                }
                AlertChannel::Email { .. } => {
                    // Email sending would require an SMTP client library
                    // For now, we'll just log it
                    tracing::info!("Email alert would be sent: {:?}", alert);
                }
            }
        }

        Ok(())
    }

    /// Send an alert via webhook.
    async fn send_webhook_alert(
        &self,
        url: &str,
        headers: &std::collections::HashMap<String, String>,
        alert: &Alert,
    ) -> Result<(), String> {
        let client = reqwest::Client::new();
        let mut request = client.post(url).json(alert);

        for (key, value) in headers {
            request = request.header(key, value);
        }

        let response = request
            .send()
            .await
            .map_err(|e| format!("Failed to send webhook alert: {}", e))?;

        if !response.status().is_success() {
            return Err(format!(
                "Webhook returned error status: {}",
                response.status()
            ));
        }

        Ok(())
    }

    /// Check metrics and send alerts if thresholds are exceeded.
    pub async fn check_and_alert(&self, error_rate: f64, latency_ms: u64) -> Result<(), String> {
        if error_rate > self.config.error_rate_threshold {
            let alert = Alert::new(
                AlertType::HighErrorRate,
                AlertSeverity::Critical,
                "High Error Rate Detected".to_string(),
                format!(
                    "Error rate {:.2}% exceeds threshold {:.2}%",
                    error_rate * 100.0,
                    self.config.error_rate_threshold * 100.0
                ),
            );
            self.send_alert(alert).await?;
        }

        if latency_ms > self.config.latency_threshold_ms {
            let alert = Alert::new(
                AlertType::HighLatency,
                AlertSeverity::Warning,
                "High Latency Detected".to_string(),
                format!(
                    "Latency {}ms exceeds threshold {}ms",
                    latency_ms, self.config.latency_threshold_ms
                ),
            );
            self.send_alert(alert).await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_creation() {
        let alert = Alert::new(
            AlertType::HighErrorRate,
            AlertSeverity::Critical,
            "Test Alert".to_string(),
            "This is a test alert".to_string(),
        );

        assert_eq!(alert.alert_type, AlertType::HighErrorRate);
        assert_eq!(alert.severity, AlertSeverity::Critical);
        assert_eq!(alert.title, "Test Alert");
    }

    #[test]
    fn test_alert_with_metadata() {
        let alert = Alert::new(
            AlertType::HighLatency,
            AlertSeverity::Warning,
            "Test".to_string(),
            "Test".to_string(),
        )
        .with_metadata("key1".to_string(), "value1".to_string())
        .with_metadata("key2".to_string(), "value2".to_string());

        assert_eq!(alert.metadata.len(), 2);
        assert_eq!(alert.metadata.get("key1"), Some(&"value1".to_string()));
    }

    #[test]
    fn test_alert_cooldown() {
        let config = AlertingConfig {
            cooldown_duration: Duration::from_secs(1),
            ..Default::default()
        };
        let manager = AlertManager::new(config);

        // First alert should be sent
        assert!(manager.should_alert(&AlertType::HighErrorRate));

        // Second alert within cooldown should not be sent
        assert!(!manager.should_alert(&AlertType::HighErrorRate));

        // Different alert type should be sent
        assert!(manager.should_alert(&AlertType::HighLatency));
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Critical);
    }
}
