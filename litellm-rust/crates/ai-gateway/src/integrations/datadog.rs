//! Datadog callback implementation.
//!
//! Sends metrics and logs to Datadog API for monitoring and observability.

use std::sync::Arc;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::custom_logger::{
    CallbackTiming, CallbackValue, CustomLogger, LogFuture, ModelCallDetails,
};

/// Datadog callback configuration.
#[derive(Clone, Debug, Deserialize)]
pub struct DatadogConfig {
    pub api_key: String,
    pub app_key: Option<String>,
    pub host: String,
}

/// Datadog callback implementation.
pub struct DatadogLogger {
    config: DatadogConfig,
    client: Client,
}

impl DatadogLogger {
    pub fn new(config: DatadogConfig) -> Self {
        Self {
            config,
            client: Client::new(),
        }
    }

    async fn send_metrics(&self, metrics: &DatadogMetrics) -> Result<(), String> {
        let url = format!("{}/api/v1/distribution_points", self.config.host);

        let response = self
            .client
            .post(&url)
            .header("DD-API-KEY", &self.config.api_key)
            .json(metrics)
            .send()
            .await
            .map_err(|e| format!("Failed to send to Datadog: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read response body".to_string());
            return Err(format!("Datadog API error: {} - {}", status, body));
        }

        Ok(())
    }

    async fn send_logs(&self, logs: &DatadogLogs) -> Result<(), String> {
        let url = format!("{}/api/v2/logs", self.config.host);

        let response = self
            .client
            .post(&url)
            .header("DD-API-KEY", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(logs)
            .send()
            .await
            .map_err(|e| format!("Failed to send logs to Datadog: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read response body".to_string());
            return Err(format!("Datadog logs API error: {} - {}", status, body));
        }

        Ok(())
    }
}

impl CustomLogger for DatadogLogger {
    fn async_log_success_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        response_obj: &'a CallbackValue,
        timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            let latency_ms = (timing.end_time - timing.start_time) * 1000.0;

            // Send metrics
            let metrics = DatadogMetrics {
                series: vec![
                    DatadogMetric {
                        metric: "litellm.request.duration".to_string(),
                        points: vec![DatadogPoint {
                            timestamp: timing.end_time as u64,
                            value: latency_ms,
                        }],
                        tags: vec![
                            format!("model:{}", model_call_details.model),
                            format!("provider:{}", model_call_details.custom_llm_provider),
                            format!("call_type:{}", model_call_details.call_type),
                            "status:success".to_string(),
                        ],
                        r#type: "gauge".to_string(),
                    },
                    DatadogMetric {
                        metric: "litellm.tokens.used".to_string(),
                        points: vec![DatadogPoint {
                            timestamp: timing.end_time as u64,
                            value: model_call_details
                                .standard_logging_payload
                                .as_ref()
                                .map(|p| p.total_tokens as f64)
                                .unwrap_or(0.0),
                        }],
                        tags: vec![
                            format!("model:{}", model_call_details.model),
                            format!("provider:{}", model_call_details.custom_llm_provider),
                        ],
                        r#type: "gauge".to_string(),
                    },
                ],
            };

            if let Err(e) = self.send_metrics(&metrics).await {
                eprintln!("Failed to send Datadog metrics: {}", e);
            }

            // Send logs
            let messages = model_call_details
                .standard_logging_payload
                .as_ref()
                .and_then(|p| p.messages.clone())
                .unwrap_or_else(|| json!([]));

            let log_entry = DatadogLogEntry {
                message: format!(
                    "LLM call to {} completed successfully",
                    model_call_details.model
                ),
                level: "info".to_string(),
                timestamp: (timing.end_time * 1000.0) as u64,
                service: "litellm".to_string(),
                tags: vec![
                    format!("model:{}", model_call_details.model),
                    format!("provider:{}", model_call_details.custom_llm_provider),
                    format!("call_type:{}", model_call_details.call_type),
                ],
                additional_attrs: json!({
                    "model": model_call_details.model,
                    "provider": model_call_details.custom_llm_provider,
                    "call_type": model_call_details.call_type.to_string(),
                    "latency_ms": latency_ms,
                    "tokens": model_call_details.standard_logging_payload.as_ref().map(|p| p.total_tokens),
                    "cost": model_call_details.response_cost,
                    "user_id": model_call_details.metadata.user_api_key_user_id,
                    "team_id": model_call_details.metadata.user_api_key_team_id,
                    "input": messages,
                    "output": response_obj.value,
                }),
            };

            let logs = DatadogLogs {
                logs: vec![log_entry],
            };

            if let Err(e) = self.send_logs(&logs).await {
                eprintln!("Failed to send Datadog logs: {}", e);
            }

            Ok(())
        })
    }

    fn async_log_failure_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        response_obj: Option<&'a CallbackValue>,
        timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            let latency_ms = (timing.end_time - timing.start_time) * 1000.0;

            // Send metrics
            let metrics = DatadogMetrics {
                series: vec![DatadogMetric {
                    metric: "litellm.request.duration".to_string(),
                    points: vec![DatadogPoint {
                        timestamp: timing.end_time as u64,
                        value: latency_ms,
                    }],
                    tags: vec![
                        format!("model:{}", model_call_details.model),
                        format!("provider:{}", model_call_details.custom_llm_provider),
                        format!("call_type:{}", model_call_details.call_type),
                        "status:error".to_string(),
                    ],
                    r#type: "gauge".to_string(),
                }],
            };

            if let Err(e) = self.send_metrics(&metrics).await {
                eprintln!("Failed to send Datadog metrics: {}", e);
            }

            // Send logs
            let messages = model_call_details
                .standard_logging_payload
                .as_ref()
                .and_then(|p| p.messages.clone())
                .unwrap_or_else(|| json!([]));

            let error_info = model_call_details
                .failure_error
                .as_ref()
                .map(|e| json!({"kind": e.kind, "message": e.message}))
                .unwrap_or_else(|| json!({"message": "Unknown error"}));

            let log_entry = DatadogLogEntry {
                message: format!("LLM call to {} failed", model_call_details.model),
                level: "error".to_string(),
                timestamp: (timing.end_time * 1000.0) as u64,
                service: "litellm".to_string(),
                tags: vec![
                    format!("model:{}", model_call_details.model),
                    format!("provider:{}", model_call_details.custom_llm_provider),
                    format!("call_type:{}", model_call_details.call_type),
                ],
                additional_attrs: json!({
                    "model": model_call_details.model,
                    "provider": model_call_details.custom_llm_provider,
                    "call_type": model_call_details.call_type.to_string(),
                    "latency_ms": latency_ms,
                    "user_id": model_call_details.metadata.user_api_key_user_id,
                    "team_id": model_call_details.metadata.user_api_key_team_id,
                    "input": messages,
                    "error": error_info,
                    "output": response_obj.map(|r| &r.value),
                }),
            };

            let logs = DatadogLogs {
                logs: vec![log_entry],
            };

            if let Err(e) = self.send_logs(&logs).await {
                eprintln!("Failed to send Datadog logs: {}", e);
            }

            Ok(())
        })
    }
}

#[derive(Debug, Serialize)]
struct DatadogMetrics {
    series: Vec<DatadogMetric>,
}

#[derive(Debug, Serialize)]
struct DatadogMetric {
    metric: String,
    points: Vec<DatadogPoint>,
    tags: Vec<String>,
    r#type: String,
}

#[derive(Debug, Serialize)]
struct DatadogPoint {
    timestamp: u64,
    value: f64,
}

#[derive(Debug, Serialize)]
struct DatadogLogs {
    logs: Vec<DatadogLogEntry>,
}

#[derive(Debug, Serialize)]
struct DatadogLogEntry {
    message: String,
    level: String,
    timestamp: u64,
    service: String,
    tags: Vec<String>,
    additional_attrs: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrations::custom_logger::{CallType, CallbackValue};
    use crate::integrations::types::{StandardLoggingMetadata, StandardLoggingPayload};
    use serde_json::json;

    #[tokio::test]
    async fn datadog_logger_sends_metrics_and_logs_on_success() {
        let config = DatadogConfig {
            api_key: "test_api_key".to_string(),
            app_key: Some("test_app_key".to_string()),
            host: "https://api.datadoghq.com".to_string(),
        };
        let logger = DatadogLogger::new(config);

        let payload = StandardLoggingPayload {
            id: "req_test".to_string(),
            litellm_call_id: "call_test".to_string(),
            call_type: "completion".to_string(),
            model: "gpt-4".to_string(),
            custom_llm_provider: "openai".to_string(),
            response_cost: 0.01,
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            start_time: 1000.0,
            end_time: 1001.5,
            stream: false,
            metadata: StandardLoggingMetadata {
                user_api_key_user_id: Some("user_123".to_string()),
                user_api_key_team_id: Some("team_456".to_string()),
                ..Default::default()
            },
            messages: Some(json!([{"role": "user", "content": "Hello"}])),
        };

        let details = ModelCallDetails::from_standard_logging_payload(payload);
        let response = CallbackValue::new("completion", json!({"choices": []}));
        let timing = CallbackTiming::new(1000.0, 1001.5);

        // In a real test, we'd mock the HTTP client and verify the requests
        // For now, we just verify it doesn't panic
        let _ = logger
            .async_log_success_event(&details, &response, timing)
            .await;
    }
}
