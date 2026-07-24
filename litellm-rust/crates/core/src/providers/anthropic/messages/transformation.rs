use crate::error::{CoreError, CoreResult};
use crate::messages::transformation::{AnthropicMessagesProviderConfig, MessagesAuthStrategy};

const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_API_BASE_ENV: &str = "ANTHROPIC_API_BASE";
const DEFAULT_ANTHROPIC_API_BASE: &str = "https://api.anthropic.com";
const MESSAGES_PATH_SUFFIX: &str = "/v1/messages";

pub struct AnthropicMessagesConfig;

pub const ANTHROPIC_MESSAGES_CONFIG: AnthropicMessagesConfig = AnthropicMessagesConfig;

pub fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

pub fn resolve_anthropic_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    non_empty(api_key)
        .map(str::to_string)
        .or_else(|| env_lookup(ANTHROPIC_API_KEY_ENV).filter(|value| !value.trim().is_empty()))
        .ok_or_else(|| {
            CoreError::Auth(
                "Missing Anthropic API Key - Set `api_key` or the ANTHROPIC_API_KEY \
                 environment variable"
                    .to_string(),
            )
        })
}

pub fn complete_anthropic_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let api_base = non_empty(api_base)
        .map(str::to_string)
        .or_else(|| env_lookup(ANTHROPIC_API_BASE_ENV).filter(|value| !value.trim().is_empty()))
        .unwrap_or_else(|| DEFAULT_ANTHROPIC_API_BASE.to_string());

    let api_base = api_base.trim_end_matches('/');
    if api_base.ends_with(MESSAGES_PATH_SUFFIX) {
        return api_base.to_string();
    }
    format!("{api_base}{MESSAGES_PATH_SUFFIX}")
}

impl AnthropicMessagesProviderConfig for AnthropicMessagesConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        Ok(complete_anthropic_url(api_base, env_lookup))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
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
        let with_env = |key: &str| {
            (key == ANTHROPIC_API_BASE_ENV).then(|| "https://env.anthropic".to_string())
        };
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
        let with_env = |key: &str| (key == ANTHROPIC_API_KEY_ENV).then(|| "sk-env".to_string());
        assert_eq!(
            resolve_anthropic_api_key(Some("  "), &with_env).unwrap(),
            "sk-env"
        );
        assert!(matches!(
            resolve_anthropic_api_key(None, &|_| None).expect_err("missing key"),
            CoreError::Auth(_)
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
