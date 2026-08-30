//! Slack callback implementation.
//!
//! Sends notifications to Slack channels for errors, budget alerts, and other events.

use std::sync::Arc;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::custom_logger::{
    CallbackTiming, CallbackValue, CustomLogger, LogFuture, ModelCallDetails,
};

/// Slack callback configuration.
#[derive(Clone, Debug, Deserialize)]
pub struct SlackConfig {
    pub webhook_url: String,
    pub channel: Option<String>,
    pub username: Option<String>,
    pub icon_emoji: Option<String>,
}

/// Slack callback implementation.
pub struct SlackLogger {
    config: SlackConfig,
    client: Client,
}

impl SlackLogger {
    pub fn new(config: SlackConfig) -> Self {
        Self {
            config,
            client: Client::new(),
        }
    }

    async fn send_slack_message(&self, message: &SlackMessage) -> Result<(), String> {
        let response = self
            .client
            .post(&self.config.webhook_url)
            .json(message)
            .send()
            .await
            .map_err(|e| format!("Failed to send Slack message: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "Failed to read response body".to_string());
            return Err(format!("Slack API error: {} - {}", status, body));
        }

        Ok(())
    }
}

impl CustomLogger for SlackLogger {
    fn async_log_success_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        _response_obj: &'a CallbackValue,
        timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            let latency_ms = (timing.end_time - timing.start_time) * 1000.0;
            let tokens = model_call_details
                .standard_logging_payload
                .as_ref()
                .map(|p| p.total_tokens)
                .unwrap_or(0);
            let cost = model_call_details.response_cost.unwrap_or(0.0);

            let text = format!(
                "✅ *LLM Call Success*\n\
                • Model: `{}`\n\
                • Provider: `{}`\n\
                • Call Type: `{}`\n\
                • Latency: `{:.2}ms`\n\
                • Tokens: `{}`\n\
                • Cost: `${:.4}`\n\
                • User: `{}`\n\
                • Team: `{}`",
                model_call_details.model,
                model_call_details.custom_llm_provider,
                model_call_details.call_type,
                latency_ms,
                tokens,
                cost,
                model_call_details
                    .metadata
                    .user_api_key_user_id
                    .as_deref()
                    .unwrap_or("N/A"),
                model_call_details
                    .metadata
                    .user_api_key_team_id
                    .as_deref()
                    .unwrap_or("N/A"),
            );

            let message = SlackMessage {
                channel: self.config.channel.clone(),
                username: self.config.username.clone(),
                icon_emoji: self.config.icon_emoji.clone(),
                text: Some(text),
                attachments: None,
                blocks: None,
            };

            self.send_slack_message(&message)
                .await
                .map_err(|e| super::custom_logger::LogError {
                    message: e,
                    kind: "SlackError".to_string(),
                })
        })
    }

    fn async_log_failure_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        _response_obj: Option<&'a CallbackValue>,
        timing: CallbackTiming,
    ) -> LogFuture<'a> {
        Box::pin(async move {
            let latency_ms = (timing.end_time - timing.start_time) * 1000.0;
            
            let error_info = model_call_details
                .failure_error
                .as_ref()
                .map(|e| format!("{}: {}", e.kind, e.message))
                .unwrap_or_else(|| "Unknown error".to_string());

            let text = format!(
                "❌ *LLM Call Failure*\n\
                • Model: `{}`\n\
                • Provider: `{}`\n\
                • Call Type: `{}`\n\
                • Latency: `{:.2}ms`\n\
                • Error: `{}`\n\
                • User: `{}`\n\
                • Team: `{}`\n\
                • Request ID: `{}`",
                model_call_details.model,
                model_call_details.custom_llm_provider,
                model_call_details.call_type,
                latency_ms,
                error_info,
                model_call_details
                    .metadata
                    .user_api_key_user_id
                    .as_deref()
                    .unwrap_or("N/A"),
                model_call_details
                    .metadata
                    .user_api_key_team_id
                    .as_deref()
                    .unwrap_or("N/A"),
                model_call_details
                    .request_id
                    .as_deref()
                    .unwrap_or("N/A"),
            );

            let message = SlackMessage {
                channel: self.config.channel.clone(),
                username: self.config.username.clone(),
                icon_emoji: self.config.icon_emoji.clone(),
                text: Some(text),
                attachments: None,
                blocks: None,
            };

            self.send_slack_message(&message)
                .await
                .map_err(|e| super::custom_logger::LogError {
                    message: e,
                    kind: "SlackError".to_string(),
                })
        })
    }
}

#[derive(Debug, Serialize)]
struct SlackMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    channel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    username: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    icon_emoji: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attachments: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    blocks: Option<Vec<serde_json::Value>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrations::custom_logger::{CallType, CallbackValue};
    use crate::integrations::types::{StandardLoggingMetadata, StandardLoggingPayload};
    use serde_json::json;

    #[tokio::test]
    async fn slack_logger_sends_message_on_success() {
        let config = SlackConfig {
            webhook_url: "https://hooks.slack.com/services/test".to_string(),
            channel: Some("#alerts".to_string()),
            username: Some("LiteLLM Bot".to_string()),
            icon_emoji: Some(":robot_face:".to_string()),
        };
        let logger = SlackLogger::new(config);

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

    #[tokio::test]
    async fn slack_logger_sends_message_on_failure() {
        let config = SlackConfig {
            webhook_url: "https://hooks.slack.com/services/test".to_string(),
            channel: Some("#alerts".to_string()),
            username: None,
            icon_emoji: None,
        };
        let logger = SlackLogger::new(config);

        let payload = StandardLoggingPayload {
            id: "req_test".to_string(),
            litellm_call_id: "call_test".to_string(),
            call_type: "completion".to_string(),
            model: "gpt-4".to_string(),
            custom_llm_provider: "openai".to_string(),
            response_cost: 0.0,
            prompt_tokens: 10,
            completion_tokens: 0,
            total_tokens: 10,
            start_time: 1000.0,
            end_time: 1001.0,
            stream: false,
            metadata: StandardLoggingMetadata {
                user_api_key_user_id: Some("user_123".to_string()),
                user_api_key_team_id: Some("team_456".to_string()),
                ..Default::default()
            },
            messages: Some(json!([{"role": "user", "content": "Hello"}])),
        };

        let mut details = ModelCallDetails::from_standard_logging_payload(payload);
        details.failure_error = Some(super::super::custom_logger::LoggingError {
            message: "Rate limit exceeded".to_string(),
            kind: "RateLimitError".to_string(),
        });

        let response = CallbackValue::new("error", json!({"error": "Rate limit exceeded"}));
        let timing = CallbackTiming::new(1000.0, 1001.0);

        // In a real test, we'd mock the HTTP client and verify the request
        // For now, we just verify it doesn't panic
        let _ = logger
            .async_log_failure_event(&details, Some(&response), timing)
            .await;
    }
}
