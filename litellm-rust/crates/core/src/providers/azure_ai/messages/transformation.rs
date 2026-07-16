use serde_json::{Map, Value};

use crate::error::{json_type_name, CoreError, CoreResult};
use crate::messages::transformation::{AnthropicMessagesProviderConfig, MessagesAuthStrategy};
use crate::messages::types::MessagesRequestData;

const AZURE_API_KEY_ENV: &str = "AZURE_API_KEY";
const AZURE_API_BASE_ENV: &str = "AZURE_API_BASE";

pub struct AzureAnthropicMessagesConfig;

pub const AZURE_ANTHROPIC_MESSAGES_CONFIG: AzureAnthropicMessagesConfig =
    AzureAnthropicMessagesConfig;

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

pub fn resolve_azure_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    non_empty(api_key)
        .map(str::to_string)
        .or_else(|| env_lookup(AZURE_API_KEY_ENV).filter(|value| !value.trim().is_empty()))
        .ok_or_else(|| {
            CoreError::Auth(
                "Missing Azure API Key - Set `api_key` or the AZURE_API_KEY environment variable"
                    .to_string(),
            )
        })
}

pub fn complete_azure_anthropic_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    let api_base = non_empty(api_base)
        .map(str::to_string)
        .or_else(|| env_lookup(AZURE_API_BASE_ENV).filter(|value| !value.trim().is_empty()))
        .ok_or_else(|| {
            CoreError::Auth(
                "Missing Azure API Base - Set `api_base` or the AZURE_API_BASE environment variable. \
                 Expected format: https://<resource-name>.services.ai.azure.com/anthropic"
                    .to_string(),
            )
        })?;

    let api_base = api_base.trim_end_matches('/');

    if api_base.ends_with("/v1/messages") || api_base.ends_with("/anthropic/v1/messages") {
        return Ok(api_base.to_string());
    }

    let with_anthropic = match api_base.split_once("/anthropic") {
        Some((prefix, _)) => format!("{prefix}/anthropic"),
        None => format!("{api_base}/anthropic"),
    };
    Ok(format!("{with_anthropic}/v1/messages"))
}

fn remove_scope_from_content_blocks(content: &mut [Value]) {
    for item in content.iter_mut() {
        if let Some(cache_control) = item
            .as_object_mut()
            .and_then(|block| block.get_mut("cache_control"))
            .and_then(Value::as_object_mut)
        {
            cache_control.remove("scope");
        }
    }
}

fn remove_scope_from_cache_control(body: &mut Map<String, Value>) {
    if let Some(Value::Array(system)) = body.get_mut("system") {
        remove_scope_from_content_blocks(system);
    }
    if let Some(Value::Array(messages)) = body.get_mut("messages") {
        for message in messages.iter_mut() {
            if let Some(Value::Array(content)) = message
                .as_object_mut()
                .and_then(|message| message.get_mut("content"))
            {
                remove_scope_from_content_blocks(content);
            }
        }
    }
}

impl AnthropicMessagesProviderConfig for AzureAnthropicMessagesConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        complete_azure_anthropic_url(api_base, env_lookup)
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        resolve_azure_api_key(api_key, env_lookup)
    }

    fn auth_strategy(&self) -> MessagesAuthStrategy {
        MessagesAuthStrategy::Header("x-api-key")
    }

    fn transform_request(&self, body: Value) -> CoreResult<MessagesRequestData> {
        let mut body = match body {
            Value::Object(body) => body,
            other => {
                return Err(CoreError::InvalidType {
                    expected: "object",
                    actual: json_type_name(&other),
                })
            }
        };
        remove_scope_from_cache_control(&mut body);
        Ok(MessagesRequestData {
            body: Value::Object(body),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn url_appends_anthropic_and_messages_suffix() {
        let url =
            complete_azure_anthropic_url(Some("https://resource.services.ai.azure.com"), &|_| None)
                .expect("url builds");
        assert_eq!(
            url,
            "https://resource.services.ai.azure.com/anthropic/v1/messages"
        );
    }

    #[test]
    fn url_keeps_existing_anthropic_segment() {
        let url = complete_azure_anthropic_url(
            Some("https://resource.services.ai.azure.com/anthropic"),
            &|_| None,
        )
        .expect("url builds");
        assert_eq!(
            url,
            "https://resource.services.ai.azure.com/anthropic/v1/messages"
        );
    }

    #[test]
    fn url_leaves_complete_messages_endpoint_untouched() {
        for base in [
            "https://resource.services.ai.azure.com/anthropic/v1/messages",
            "https://resource.services.ai.azure.com/v1/messages",
        ] {
            assert_eq!(
                complete_azure_anthropic_url(Some(base), &|_| None).expect("url builds"),
                base
            );
        }
    }

    #[test]
    fn url_trims_trailing_slash_and_truncates_after_anthropic() {
        let url = complete_azure_anthropic_url(
            Some("https://resource.services.ai.azure.com/anthropic/extra/"),
            &|_| None,
        )
        .expect("url builds");
        assert_eq!(
            url,
            "https://resource.services.ai.azure.com/anthropic/v1/messages"
        );
    }

    #[test]
    fn url_falls_back_to_env_then_errors_when_absent() {
        let with_env = |key: &str| {
            (key == AZURE_API_BASE_ENV).then(|| "https://env.services.ai.azure.com".to_string())
        };
        assert_eq!(
            complete_azure_anthropic_url(None, &with_env).expect("url builds"),
            "https://env.services.ai.azure.com/anthropic/v1/messages"
        );
        let err = complete_azure_anthropic_url(Some("  "), &|_| None).expect_err("missing base");
        assert!(matches!(err, CoreError::Auth(_)));
    }

    #[test]
    fn resolve_api_key_prefers_param_then_env() {
        assert_eq!(
            resolve_azure_api_key(Some("sk-param"), &|_| None).unwrap(),
            "sk-param"
        );
        let with_env = |key: &str| (key == AZURE_API_KEY_ENV).then(|| "sk-env".to_string());
        assert_eq!(
            resolve_azure_api_key(Some("  "), &with_env).unwrap(),
            "sk-env"
        );
        assert!(matches!(
            resolve_azure_api_key(None, &|_| None).expect_err("missing key"),
            CoreError::Auth(_)
        ));
    }

    #[test]
    fn auth_strategy_is_x_api_key() {
        assert_eq!(
            AZURE_ANTHROPIC_MESSAGES_CONFIG
                .auth_strategy()
                .header_name(),
            "x-api-key"
        );
    }

    #[test]
    fn default_headers_match_python() {
        assert_eq!(
            AZURE_ANTHROPIC_MESSAGES_CONFIG.default_headers(),
            &[
                ("anthropic-version", "2023-06-01"),
                ("content-type", "application/json"),
            ]
        );
    }

    #[test]
    fn transform_request_strips_scope_from_system_and_messages() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": "sys",
                    "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "global"}
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "ephemeral", "scope": "global"}
                        },
                        {"type": "text", "text": "no cache control"}
                    ]
                }
            ]
        });

        let transformed = AZURE_ANTHROPIC_MESSAGES_CONFIG
            .transform_request(body)
            .expect("request transforms")
            .body;

        assert_eq!(
            transformed["system"][0]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
        assert_eq!(
            transformed["messages"][0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert_eq!(
            transformed["messages"][0]["content"][1],
            json!({"type": "text", "text": "no cache control"})
        );
    }

    #[test]
    fn transform_request_is_idempotent_and_preserves_string_system() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "system": "plain string system",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let once = AZURE_ANTHROPIC_MESSAGES_CONFIG
            .transform_request(body)
            .expect("request transforms")
            .body;
        let twice = AZURE_ANTHROPIC_MESSAGES_CONFIG
            .transform_request(once.clone())
            .expect("request transforms")
            .body;
        assert_eq!(once, twice);
        assert_eq!(once["system"], json!("plain string system"));
    }

    #[test]
    fn transform_request_rejects_non_object_body() {
        let err = AZURE_ANTHROPIC_MESSAGES_CONFIG
            .transform_request(json!("bad"))
            .expect_err("non-object body should error");
        assert_eq!(
            err,
            CoreError::InvalidType {
                expected: "object",
                actual: "string",
            }
        );
    }

    #[test]
    fn transform_response_passes_through_object() {
        let response = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2}
        });
        let transformed = AZURE_ANTHROPIC_MESSAGES_CONFIG
            .transform_response("claude-sonnet-4-5", response.clone())
            .expect("response transforms")
            .into_json();
        assert_eq!(transformed, response);
    }

    #[test]
    fn transform_response_rejects_non_object() {
        let err = AZURE_ANTHROPIC_MESSAGES_CONFIG
            .transform_response("claude-sonnet-4-5", json!([1, 2, 3]))
            .expect_err("array response should error");
        assert_eq!(
            err,
            CoreError::InvalidType {
                expected: "object",
                actual: "array",
            }
        );
    }
}
