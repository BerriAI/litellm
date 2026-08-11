use serde_json::Value;

use crate::error::{CoreError, CoreResult};
use crate::messages::transformation::{AnthropicMessagesProviderConfig, MessagesAuthStrategy};
use crate::messages::types::{AnthropicMessagesRequest, SystemPrompt};
use crate::providers::anthropic::messages::transformation::non_empty;

const DEEPSEEK_API_KEY_ENV: &str = "DEEPSEEK_API_KEY";
const DEEPSEEK_ANTHROPIC_API_BASE_ENV: &str = "DEEPSEEK_ANTHROPIC_API_BASE";
const DEEPSEEK_API_BASE_ENV: &str = "DEEPSEEK_API_BASE";
const DEFAULT_DEEPSEEK_ANTHROPIC_API_BASE: &str = "https://api.deepseek.com/anthropic";
const MESSAGES_PATH_SUFFIX: &str = "/v1/messages";
const BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header:";

pub struct DeepSeekAnthropicMessagesConfig;

pub const DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG: DeepSeekAnthropicMessagesConfig =
    DeepSeekAnthropicMessagesConfig;

fn env_value(key: &str, env_lookup: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    env_lookup(key).filter(|value| !value.trim().is_empty())
}

fn resolve_deepseek_api_base(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    non_empty(api_base)
        .map(str::to_string)
        .or_else(|| env_value(DEEPSEEK_ANTHROPIC_API_BASE_ENV, env_lookup))
        .or_else(|| env_value(DEEPSEEK_API_BASE_ENV, env_lookup))
        .unwrap_or_else(|| DEFAULT_DEEPSEEK_ANTHROPIC_API_BASE.to_string())
}

pub fn complete_deepseek_messages_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let base_url = resolve_deepseek_api_base(api_base, env_lookup);
    let base_url = base_url.trim_end_matches('/');

    if base_url.ends_with(MESSAGES_PATH_SUFFIX) && base_url.contains("/anthropic/") {
        return base_url.to_string();
    }

    let base_url = base_url
        .strip_suffix(MESSAGES_PATH_SUFFIX)
        .unwrap_or(base_url);
    let base_url = base_url.strip_suffix("/v1").unwrap_or(base_url);
    let base_url = base_url.strip_suffix("/beta").unwrap_or(base_url);

    let base_url = if base_url.ends_with("/anthropic") || base_url.contains("/anthropic/") {
        base_url.to_string()
    } else {
        format!("{base_url}/anthropic")
    };

    format!("{base_url}{MESSAGES_PATH_SUFFIX}")
}

pub fn resolve_deepseek_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    non_empty(api_key)
        .map(str::to_string)
        .or_else(|| env_value(DEEPSEEK_API_KEY_ENV, env_lookup))
        .ok_or_else(|| {
            CoreError::Auth(
                "Missing DeepSeek API Key - Set `api_key` or the DEEPSEEK_API_KEY environment variable"
                    .to_string(),
            )
        })
}

fn sanitize_tool(tool: Value) -> Value {
    match tool {
        Value::Object(entries) if entries.get("type").and_then(Value::as_str) == Some("custom") => {
            Value::Object(
                entries
                    .into_iter()
                    .filter(|(key, _)| key != "type")
                    .collect(),
            )
        }
        value => value,
    }
}

fn strip_billing_system(system: Option<&SystemPrompt>) -> Option<SystemPrompt> {
    match system {
        Some(SystemPrompt::Text(text)) if text.starts_with(BILLING_HEADER_PREFIX) => None,
        Some(SystemPrompt::Text(text)) => Some(SystemPrompt::Text(text.clone())),
        Some(SystemPrompt::Blocks(blocks)) => {
            let blocks = blocks
                .iter()
                .filter(|block| {
                    let is_text = block.extra.get("type").and_then(Value::as_str) == Some("text");
                    let is_billing = block
                        .extra
                        .get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| text.starts_with(BILLING_HEADER_PREFIX));
                    !(is_text && is_billing)
                })
                .cloned()
                .collect::<Vec<_>>();

            (!blocks.is_empty()).then_some(SystemPrompt::Blocks(blocks))
        }
        None => None,
    }
}

impl AnthropicMessagesProviderConfig for DeepSeekAnthropicMessagesConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        Ok(complete_deepseek_messages_url(api_base, env_lookup))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        resolve_deepseek_api_key(api_key, env_lookup)
    }

    fn auth_strategy(&self) -> MessagesAuthStrategy {
        MessagesAuthStrategy::Header("x-api-key")
    }

    fn accepts_authorization_header(&self) -> bool {
        true
    }

    fn transform_request(
        &self,
        request: AnthropicMessagesRequest,
    ) -> CoreResult<AnthropicMessagesRequest> {
        let tools = request
            .tools
            .as_ref()
            .map(|tools| tools.iter().cloned().map(sanitize_tool).collect::<Vec<_>>());
        let system = strip_billing_system(request.system.as_ref());

        Ok(AnthropicMessagesRequest {
            tools,
            system,
            ..request
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn url_defaults_to_deepseek_anthropic_endpoint() {
        assert_eq!(
            complete_deepseek_messages_url(None, &|_| None),
            "https://api.deepseek.com/anthropic/v1/messages"
        );
    }

    #[test]
    fn url_normalization_matches_python() {
        for api_base in [
            "https://api.deepseek.com/anthropic/v1",
            "https://api.deepseek.com/anthropic",
            "https://api.deepseek.com",
            "https://api.deepseek.com/v1",
            "https://api.deepseek.com/v1/messages",
        ] {
            assert_eq!(
                complete_deepseek_messages_url(Some(api_base), &|_| None),
                "https://api.deepseek.com/anthropic/v1/messages"
            );
        }
    }

    #[test]
    fn url_prefers_anthropic_env_then_general_env() {
        let with_both = |key: &str| match key {
            DEEPSEEK_ANTHROPIC_API_BASE_ENV => Some("https://anthropic.deepseek.test".to_string()),
            DEEPSEEK_API_BASE_ENV => Some("https://general.deepseek.test".to_string()),
            _ => None,
        };

        assert_eq!(
            complete_deepseek_messages_url(None, &with_both),
            "https://anthropic.deepseek.test/anthropic/v1/messages"
        );

        let with_general = |key: &str| {
            (key == DEEPSEEK_API_BASE_ENV).then(|| "https://general.deepseek.test".to_string())
        };

        assert_eq!(
            complete_deepseek_messages_url(None, &with_general),
            "https://general.deepseek.test/anthropic/v1/messages"
        );
    }

    #[test]
    fn api_key_prefers_param_then_environment() {
        assert_eq!(
            resolve_deepseek_api_key(Some("sk-param"), &|_| None).unwrap(),
            "sk-param"
        );

        let with_env =
            |key: &str| (key == DEEPSEEK_API_KEY_ENV).then(|| "sk-deepseek-env".to_string());

        assert_eq!(
            resolve_deepseek_api_key(None, &with_env).unwrap(),
            "sk-deepseek-env"
        );
    }

    #[test]
    fn transform_preserves_thinking_and_sanitizes_custom_tools() {
        let request: AnthropicMessagesRequest = serde_json::from_value(json!({
            "model": "deepseek-v4-pro",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "I should call the tool.",
                            "signature": "sig"
                        }
                    ]
                }
            ],
            "max_tokens": 100,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024
            },
            "tools": [
                {
                    "type": "custom",
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object"}
                },
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": 1
                }
            ]
        }))
        .unwrap();

        let transformed = DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG
            .transform_request(request)
            .unwrap();
        let value = serde_json::to_value(transformed).unwrap();

        assert_eq!(
            value["thinking"],
            json!({"type": "enabled", "budget_tokens": 1024})
        );
        assert_eq!(
            value["tools"][0],
            json!({
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object"}
            })
        );
        assert_eq!(value["tools"][1]["type"], "web_search_20260209");
        assert_eq!(value["messages"][0]["content"][0]["type"], "thinking");
    }

    #[test]
    fn transform_strips_billing_system_blocks() {
        let request: AnthropicMessagesRequest = serde_json::from_value(json!({
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100,
            "system": [
                {
                    "type": "text",
                    "text": "x-anthropic-billing-header: cc_version=1"
                },
                {
                    "type": "text",
                    "text": "Keep this"
                }
            ]
        }))
        .unwrap();

        let transformed = DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG
            .transform_request(request)
            .unwrap();
        let value = serde_json::to_value(transformed).unwrap();

        assert_eq!(
            value["system"],
            json!([{"type": "text", "text": "Keep this"}])
        );
    }
}
