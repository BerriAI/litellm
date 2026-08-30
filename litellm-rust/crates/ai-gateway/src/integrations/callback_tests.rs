//! Tests for callback integrations.

#[cfg(test)]
mod tests {
    use crate::integrations::custom_logger::{
        CallbackTiming, CallbackValue, CustomLogger, CustomLoggerRunner, ModelCallDetails,
    };
    use crate::integrations::custom_logger::CallType;
    use crate::integrations::types::{StandardLoggingMetadata, StandardLoggingPayload};
    use serde_json::json;
    use std::sync::Arc;

    #[derive(Default)]
    struct TestLogger {
        success_calls: std::sync::Mutex<Vec<String>>,
        failure_calls: std::sync::Mutex<Vec<String>>,
    }

    impl TestLogger {
        fn success_count(&self) -> usize {
            self.success_calls.lock().unwrap().len()
        }

        fn failure_count(&self) -> usize {
            self.failure_calls.lock().unwrap().len()
        }
    }

    impl CustomLogger for TestLogger {
        fn async_log_success_event<'a>(
            &'a self,
            model_call_details: &'a ModelCallDetails,
            _response_obj: &'a CallbackValue,
            _timing: CallbackTiming,
        ) -> crate::integrations::custom_logger::LogFuture<'a> {
            Box::pin(async move {
                self.success_calls
                    .lock()
                    .unwrap()
                    .push(model_call_details.model.clone());
                Ok(())
            })
        }

        fn async_log_failure_event<'a>(
            &'a self,
            model_call_details: &'a ModelCallDetails,
            _response_obj: Option<&'a CallbackValue>,
            _timing: CallbackTiming,
        ) -> crate::integrations::custom_logger::LogFuture<'a> {
            Box::pin(async move {
                self.failure_calls
                    .lock()
                    .unwrap()
                    .push(model_call_details.model.clone());
                Ok(())
            })
        }
    }

    fn create_test_payload() -> StandardLoggingPayload {
        StandardLoggingPayload {
            id: "req_test".to_string(),
            litellm_call_id: "call_test".to_string(),
            call_type: "chat_completion".to_string(),
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
        }
    }

    #[tokio::test]
    async fn test_callback_runner_dispatches_success_event() {
        let logger = Arc::new(TestLogger::default());
        let runner = CustomLoggerRunner::new(vec![logger.clone()]);

        let payload = create_test_payload();
        let details = ModelCallDetails::from_standard_logging_payload(payload);
        let response = CallbackValue::new("chat.completion", json!({"choices": []}));
        let timing = CallbackTiming::new(1000.0, 1001.5);

        let report = runner
            .async_log_success_event(&details, &response, timing)
            .await;

        assert_eq!(report.invoked, 1);
        assert_eq!(report.dropped, 0);
        assert_eq!(logger.success_count(), 1);
    }

    #[tokio::test]
    async fn test_callback_runner_dispatches_failure_event() {
        let logger = Arc::new(TestLogger::default());
        let runner = CustomLoggerRunner::new(vec![logger.clone()]);

        let payload = create_test_payload();
        let details = ModelCallDetails::from_standard_logging_payload(payload);
        let timing = CallbackTiming::new(1000.0, 1001.0);

        let report = runner
            .async_log_failure_event(&details, None, timing)
            .await;

        assert_eq!(report.invoked, 1);
        assert_eq!(report.dropped, 0);
        assert_eq!(logger.failure_count(), 1);
    }

    #[tokio::test]
    async fn test_callback_runner_handles_multiple_loggers() {
        let logger1 = Arc::new(TestLogger::default());
        let logger2 = Arc::new(TestLogger::default());
        let runner = CustomLoggerRunner::new(vec![logger1.clone(), logger2.clone()]);

        let payload = create_test_payload();
        let details = ModelCallDetails::from_standard_logging_payload(payload);
        let response = CallbackValue::new("chat.completion", json!({"choices": []}));
        let timing = CallbackTiming::new(1000.0, 1001.5);

        let report = runner
            .async_log_success_event(&details, &response, timing)
            .await;

        assert_eq!(report.invoked, 2);
        assert_eq!(report.dropped, 0);
        assert_eq!(logger1.success_count(), 1);
        assert_eq!(logger2.success_count(), 1);
    }

    #[tokio::test]
    async fn test_callback_runner_with_empty_loggers() {
        let runner = CustomLoggerRunner::new(vec![]);

        let payload = create_test_payload();
        let details = ModelCallDetails::from_standard_logging_payload(payload);
        let response = CallbackValue::new("chat.completion", json!({"choices": []}));
        let timing = CallbackTiming::new(1000.0, 1001.5);

        let report = runner
            .async_log_success_event(&details, &response, timing)
            .await;

        assert_eq!(report.invoked, 0);
        assert_eq!(report.dropped, 0);
    }

    #[test]
    #[cfg(feature = "server")]
    fn test_callback_config_deserialization() {
        use crate::config::CallbackConfig;

        let yaml = r#"
type: langfuse
public_key: pk_test
secret_key: sk_test
host: https://langfuse.example.com
"#;
        let config: CallbackConfig = serde_yaml::from_str(yaml).unwrap();
        match config {
            CallbackConfig::Langfuse {
                public_key,
                secret_key,
                host,
            } => {
                assert_eq!(public_key, "pk_test");
                assert_eq!(secret_key, "sk_test");
                assert_eq!(host, "https://langfuse.example.com");
            }
            _ => panic!("Expected Langfuse config"),
        }

        let yaml = r#"
type: datadog
api_key: dd_api_key
host: https://api.datadoghq.com
"#;
        let config: CallbackConfig = serde_yaml::from_str(yaml).unwrap();
        match config {
            CallbackConfig::Datadog {
                api_key,
                app_key,
                host,
            } => {
                assert_eq!(api_key, "dd_api_key");
                assert!(app_key.is_none());
                assert_eq!(host, "https://api.datadoghq.com");
            }
            _ => panic!("Expected Datadog config"),
        }

        let yaml = r#"
type: webhooks
url: https://webhook.example.com
auth_token: secret_token
"#;
        let config: CallbackConfig = serde_yaml::from_str(yaml).unwrap();
        match config {
            CallbackConfig::Webhooks {
                url,
                headers,
                auth_token,
            } => {
                assert_eq!(url, "https://webhook.example.com");
                assert!(headers.is_none());
                assert_eq!(auth_token, Some("secret_token".to_string()));
            }
            _ => panic!("Expected Webhooks config"),
        }

        let yaml = r#"
type: slack
webhook_url: https://hooks.slack.com/services/test
channel: '#alerts'
username: LiteLLM Bot
"#;
        let config: CallbackConfig = serde_yaml::from_str(yaml).unwrap();
        match config {
            CallbackConfig::Slack {
                webhook_url,
                channel,
                username,
                icon_emoji,
            } => {
                assert_eq!(webhook_url, "https://hooks.slack.com/services/test");
                assert_eq!(channel, Some("#alerts".to_string()));
                assert_eq!(username, Some("LiteLLM Bot".to_string()));
                assert!(icon_emoji.is_none());
            }
            _ => panic!("Expected Slack config"),
        }
    }
}
