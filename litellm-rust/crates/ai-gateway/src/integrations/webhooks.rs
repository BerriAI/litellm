//! Webhooks callback implementation.
//!
//! Sends POST requests to configured URLs with request/response data.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::custom_logger::{
    CallbackTiming, CallbackValue, CustomLogger, LogFuture, ModelCallDetails,
};

/// Webhooks callback configuration.
#[derive(Clone, Debug, Deserialize)]
pub struct WebhooksConfig {
    pub url: String,
    pub headers: Option<std::collections::HashMap<String, String>>,
    pub auth_token: Option<String>,
}

/// Webhooks callback implementation.
pub struct WebhooksLogger {
    config: WebhooksConfig,
    client: Client,
}

impl WebhooksLogger {
    pub fn new(config: WebhooksConfig) -> Self {
        Self {
            config,
            client: Client::new(),
        }
    }

    async fn send_webhook(&self, payload: &WebhookPayload) -> Result<(), String> {
        let mut request = self.client.post(&self.config.url).json(payload);

        // Add custom headers
        if let Some(headers) = &self.config.headers {
            for (key, value) in headers {
                request = request.header(key, value);
            }
        }

        // Add auth token if provided
        if let Some(token) = &self.config.auth_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request
            .send()
            .await
            .map_err(|e| format!("Failed to send webhook: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read response body".to_string());
            return Err(format!("Webhook error: {} - {}", status, body));
        }

        Ok(())
    }
}

impl CustomLogger for WebhooksLogger {
    fn async_log_success_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        response_obj: &'a CallbackValue,
        timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            let messages = model_call_details
                .standard_logging_payload
                .as_ref()
                .and_then(|p| p.messages.clone())
                .unwrap_or_else(|| json!([]));

            let payload = WebhookPayload {
                event: "llm_call.success".to_string(),
                timestamp: timing.end_time,
                data: json!({
                    "model": model_call_details.model,
                    "provider": model_call_details.custom_llm_provider,
                    "call_type": model_call_details.call_type.to_string(),
                    "request_id": model_call_details.request_id,
                    "litellm_call_id": model_call_details.litellm_call_id,
                    "user_id": model_call_details.metadata.user_api_key_user_id,
                    "team_id": model_call_details.metadata.user_api_key_team_id,
                    "latency_ms": (timing.end_time - timing.start_time) * 1000.0,
                    "tokens": model_call_details.standard_logging_payload.as_ref().map(|p| p.total_tokens),
                    "cost": model_call_details.response_cost,
                    "input": messages,
                    "output": response_obj.value,
                    "status": "success",
                }),
            };

            self.send_webhook(&payload)
                .await
                .map_err(|e| super::custom_logger::LogError {
                    message: e,
                    kind: "WebhookError".to_string(),
                })
        })
    }

    fn async_log_failure_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        response_obj: Option<&'a CallbackValue>,
        timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
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

            let payload = WebhookPayload {
                event: "llm_call.failure".to_string(),
                timestamp: timing.end_time,
                data: json!({
                    "model": model_call_details.model,
                    "provider": model_call_details.custom_llm_provider,
                    "call_type": model_call_details.call_type.to_string(),
                    "request_id": model_call_details.request_id,
                    "litellm_call_id": model_call_details.litellm_call_id,
                    "user_id": model_call_details.metadata.user_api_key_user_id,
                    "team_id": model_call_details.metadata.user_api_key_team_id,
                    "latency_ms": (timing.end_time - timing.start_time) * 1000.0,
                    "input": messages,
                    "error": error_info,
                    "output": response_obj.map(|r| &r.value),
                    "status": "failure",
                }),
            };

            self.send_webhook(&payload)
                .await
                .map_err(|e| super::custom_logger::LogError {
                    message: e,
                    kind: "WebhookError".to_string(),
                })
        })
    }
}

#[derive(Debug, Serialize)]
struct WebhookPayload {
    event: String,
    timestamp: f64,
    data: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrations::custom_logger::CallbackValue;
    use crate::integrations::types::{StandardLoggingMetadata, StandardLoggingPayload};
    use serde_json::json;

    #[tokio::test]
    async fn webhooks_logger_sends_payload_on_success() {
        let config = WebhooksConfig {
            url: "http://localhost:8080/webhook".to_string(),
            headers: None,
            auth_token: Some("test_token".to_string()),
        };
        let logger = WebhooksLogger::new(config);

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

        // In a real test, we'd mock the HTTP client and verify the request
        // For now, we just verify it doesn't panic
        let _ = logger
            .async_log_success_event(&details, &response, timing)
            .await;
    }
}
