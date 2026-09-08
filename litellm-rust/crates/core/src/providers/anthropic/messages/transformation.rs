use crate::error::Error;
use crate::messages::transformation::{AnthropicMessagesProviderConfig, MessagesAuthStrategy};
pub use crate::providers::anthropic::auth::{
    complete_anthropic_url, non_empty, resolve_anthropic_api_key,
};

pub struct AnthropicMessagesConfig;

pub const ANTHROPIC_MESSAGES_CONFIG: AnthropicMessagesConfig = AnthropicMessagesConfig;

impl AnthropicMessagesProviderConfig for AnthropicMessagesConfig {
    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(complete_anthropic_url(api_base, env_lookup))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        resolve_anthropic_api_key(api_key, env_lookup)
    }

    fn auth_strategy(&self) -> MessagesAuthStrategy {
        MessagesAuthStrategy::Header("x-api-key")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_defaults_to_public_anthropic_endpoint() {
        assert_eq!(
            complete_anthropic_url(None, &|_| None),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn url_appends_messages_suffix_to_custom_base() {
        assert_eq!(
            complete_anthropic_url(Some("https://proxy.internal"), &|_| None),
            "https://proxy.internal/v1/messages"
        );
    }

    #[test]
    fn url_leaves_complete_messages_endpoint_untouched() {
        assert_eq!(
            complete_anthropic_url(Some("https://proxy.internal/v1/messages"), &|_| None),
            "https://proxy.internal/v1/messages"
        );
    }

    #[test]
    fn url_falls_back_to_env_base() {
        let with_env =
            |key: &str| (key == "ANTHROPIC_API_BASE").then(|| "https://env.anthropic".to_string());
        assert_eq!(
            complete_anthropic_url(Some("  "), &with_env),
            "https://env.anthropic/v1/messages"
        );
    }

    #[test]
    fn api_key_prefers_param_then_env_then_errors() {
        assert_eq!(
            resolve_anthropic_api_key(Some("sk-param"), &|_| None).unwrap(),
            "sk-param"
        );
        let with_env = |key: &str| (key == "ANTHROPIC_API_KEY").then(|| "sk-env".to_string());
        assert_eq!(
            resolve_anthropic_api_key(Some("  "), &with_env).unwrap(),
            "sk-env"
        );
        assert!(matches!(
            resolve_anthropic_api_key(None, &|_| None).expect_err("missing key"),
            Error::Auth(_)
        ));
    }

    #[test]
    fn auth_strategy_and_default_headers_match_anthropic() {
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.auth_strategy().header_name(),
            "x-api-key"
        );
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.default_headers(),
            &[
                ("anthropic-version", "2023-06-01"),
                ("content-type", "application/json"),
            ]
        );
    }
}
