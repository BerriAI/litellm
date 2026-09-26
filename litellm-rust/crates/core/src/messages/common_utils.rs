use litellm_http::request::string_headers as shared_string_headers;
pub(super) use litellm_http::request::truncate_error_body;
use litellm_llms::{
    anthropic::messages::transformation::ANTHROPIC_MESSAGES_CONFIG,
    azure_ai::anthropic::messages_transformation::AZURE_ANTHROPIC_MESSAGES_CONFIG,
    base_llm::anthropic_messages::transformation::BaseAnthropicMessagesConfig,
    bedrock::messages::invoke_transformations::anthropic_claude3_transformation::BEDROCK_ANTHROPIC_MESSAGES_CONFIG,
};
use serde_json::{Map, Value};
use strum::{EnumString, IntoStaticStr};

use super::Error;

const HEADER_CONTEXT: &str = "messages";

#[derive(Clone, Copy, Debug, EnumString, IntoStaticStr, PartialEq, Eq)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum MessagesProvider {
    Anthropic,
    AzureAi,
    Bedrock,
}

impl MessagesProvider {
    pub(crate) fn as_str(self) -> &'static str {
        self.into()
    }

    pub(crate) fn config(self) -> &'static dyn BaseAnthropicMessagesConfig {
        match self {
            Self::Anthropic => &ANTHROPIC_MESSAGES_CONFIG,
            Self::AzureAi => &AZURE_ANTHROPIC_MESSAGES_CONFIG,
            Self::Bedrock => &BEDROCK_ANTHROPIC_MESSAGES_CONFIG,
        }
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

    use rstest::rstest;

    use super::{MessagesProvider, string_headers, truncate_error_body};
    use crate::messages::Error;

    #[rstest]
    #[case::anthropic("anthropic", MessagesProvider::Anthropic)]
    #[case::azure_ai("azure_ai", MessagesProvider::AzureAi)]
    #[case::bedrock("bedrock", MessagesProvider::Bedrock)]
    fn provider_round_trips_through_its_python_name(
        #[case] name: &str,
        #[case] provider: MessagesProvider,
    ) {
        assert_eq!(name.parse::<MessagesProvider>(), Ok(provider));
        assert_eq!(provider.as_str(), name);
    }

    #[test]
    fn provider_without_a_messages_config_is_rejected() {
        assert!("openai".parse::<MessagesProvider>().is_err());
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
