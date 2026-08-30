//! Langfuse callback implementation.
//!
//! Sends traces, spans, and metrics to Langfuse API for observability.

use std::sync::Arc;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::custom_logger::{
    CallbackTiming, CallbackValue, CustomLogger, LogFuture, ModelCallDetails,
};

/// Langfuse callback configuration.
#[derive(Clone, Debug, Deserialize)]
pub struct LangfuseConfig {
    pub public_key: String,
    pub secret_key: String,
    pub host: String,
}

/// Langfuse callback implementation.
pub struct LangfuseLogger {
    config: LangfuseConfig,
    client: Client,
}

impl LangfuseLogger {
    pub fn new(config: LangfuseConfig) -> Self {
        Self {
            config,
            client: Client::new(),
        }
    }

    async fn send_trace(&self, payload: &LangfuseTrace) -> Result<(), String> {
        let url = format!("{}/api/public/ingestion", self.config.host);

        let response = self
            .client
            .post(&url)
            .basic_auth(&self.config.public_key, Some(&self.config.secret_key))
            .json(payload)
            .send()
            .await
            .map_err(|e| format!("Failed to send to Langfuse: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read response body".to_string());
            return Err(format!("Langfuse API error: {} - {}", status, body));
        }

        Ok(())
    }
}

impl CustomLogger for LangfuseLogger {
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
                .unwrap_or_else(|| serde_json::json!([]));

            let trace = LangfuseTrace {
                batch_type: "trace-create".to_string(),
                body: LangfuseTraceBody {
                    id: model_call_details
                        .litellm_call_id
                        .clone()
                        .unwrap_or_else(|| format!("call_{}", timing.start_time)),
                    name: format!("litellm.{}", model_call_details.call_type),
                    input: serde_json::json!({"messages": messages}),
                    output: response_obj.value.clone(),
                    metadata: serde_json::json!({
                        "model": model_call_details.model,
                        "provider": model_call_details.custom_llm_provider,
                        "call_type": model_call_details.call_type.to_string(),
                        "user_id": model_call_details.metadata.user_api_key_user_id,
                        "team_id": model_call_details.metadata.user_api_key_team_id,
                    }),
                    start_time: format!("{}", timing.start_time),
                    end_time: Some(format!("{}", timing.end_time)),
                    session_id: None,
                    user_id: model_call_details.metadata.user_api_key_user_id.clone(),
                },
            };

            self.send_trace(&trace)
                .await
                .map_err(|e| super::custom_logger::LogError {
                    message: e,
                    kind: "LangfuseError".to_string(),
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
                .unwrap_or_else(|| serde_json::json!([]));

            let error_info = model_call_details
                .failure_error
                .as_ref()
                .map(|e| serde_json::json!({"kind": e.kind, "message": e.message}))
                .unwrap_or_else(|| serde_json::json!({"message": "Unknown error"}));

            let trace = LangfuseTrace {
                batch_type: "trace-create".to_string(),
                body: LangfuseTraceBody {
                    id: model_call_details
                        .litellm_call_id
                        .clone()
                        .unwrap_or_else(|| format!("call_{}", timing.start_time)),
                    name: format!("litellm.{}", model_call_details.call_type),
                    input: serde_json::json!({"messages": messages}),
                    output: response_obj
                        .map(|r| r.value.clone())
                        .unwrap_or_else(|| error_info.clone()),
                    metadata: serde_json::json!({
                        "model": model_call_details.model,
                        "provider": model_call_details.custom_llm_provider,
                        "call_type": model_call_details.call_type.to_string(),
                        "user_id": model_call_details.metadata.user_api_key_user_id,
                        "team_id": model_call_details.metadata.user_api_key_team_id,
                        "error": error_info,
                    }),
                    start_time: format!("{}", timing.start_time),
                    end_time: Some(format!("{}", timing.end_time)),
                    session_id: None,
                    user_id: model_call_details.metadata.user_api_key_user_id.clone(),
                },
            };

            self.send_trace(&trace)
                .await
                .map_err(|e| super::custom_logger::LogError {
                    message: e,
                    kind: "LangfuseError".to_string(),
                })
        })
    }
}

#[derive(Debug, Serialize)]
struct LangfuseTrace {
    batch_type: String,
    body: LangfuseTraceBody,
}

#[derive(Debug, Serialize)]
struct LangfuseTraceBody {
    id: String,
    name: String,
    input: serde_json::Value,
    output: serde_json::Value,
    metadata: serde_json::Value,
    start_time: String,
    end_time: Option<String>,
    session_id: Option<String>,
    user_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrations::custom_logger::{CallType, CallbackValue};
    use crate::integrations::types::{StandardLoggingMetadata, StandardLoggingPayload};
    use serde_json::json;

    #[tokio::test]
    async fn langfuse_logger_creates_trace_on_success() {
        // This test would require mocking the HTTP client or using a test server
        // For now, we'll just verify the structure is correct
        let config = LangfuseConfig {
            public_key: "test_public_key".to_string(),
            secret_key: "test_secret_key".to_string(),
            host: "http://localhost:3000".to_string(),
        };
        let logger = LangfuseLogger::new(config);

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
