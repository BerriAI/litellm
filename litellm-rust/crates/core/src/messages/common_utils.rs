use litellm_http::request::string_headers as shared_string_headers;
pub(super) use litellm_http::request::truncate_error_body;
use litellm_llms::{
    anthropic::experimental_pass_through::messages::transformation::ANTHROPIC_MESSAGES_CONFIG,
    azure_ai::anthropic::messages_transformation::AZURE_ANTHROPIC_MESSAGES_CONFIG,
    base_llm::anthropic_messages::transformation::BaseAnthropicMessagesConfig,
};
use serde_json::{Map, Value};

use super::Error;

const HEADER_CONTEXT: &str = "messages";

pub(super) fn messages_provider_config(
    provider: &str,
) -> Option<&'static dyn BaseAnthropicMessagesConfig> {
    match provider {
        "anthropic" => Some(&ANTHROPIC_MESSAGES_CONFIG),
        "azure_ai" => Some(&AZURE_ANTHROPIC_MESSAGES_CONFIG),
        _ => None,
    }
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> Result<Vec<(String, String)>, Error> {
    shared_string_headers(HEADER_CONTEXT, extra_headers).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{messages_provider_config, string_headers, truncate_error_body};
    use crate::messages::Error;

    #[test]
    fn provider_config_resolves_anthropic_and_azure_ai() {
        assert!(messages_provider_config("anthropic").is_some());
        assert!(messages_provider_config("azure_ai").is_some());
        assert!(messages_provider_config("openai").is_none());
    }

    #[test]
    fn truncate_error_body_caps_long_payloads() {
        let body = "x".repeat(400);
        let truncated = truncate_error_body(&body);
        assert!(truncated.ends_with("... (truncated)"));
        let prefix_chars = truncated
            .strip_suffix("... (truncated)")
            .expect("truncated marker present")
            .chars()
            .count();
        assert_eq!(prefix_chars, 256);
    }

    #[test]
    fn string_headers_rejects_non_string_values() {
        let headers = json!({"x-count": 3}).as_object().unwrap().clone();
        let err = string_headers(Some(headers)).expect_err("non-string header rejected");
        assert_eq!(
            err,
            Error::Headers(litellm_http::request::HeaderError {
                context: "messages",
                name: "x-count".to_string(),
                actual: "number",
            })
        );
    }
}
