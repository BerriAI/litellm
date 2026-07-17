use crate::error::{CoreError, CoreResult};
use crate::messages::transformation::{AnthropicMessagesProviderConfig, MessagesAuthStrategy};
use crate::messages::types::{
    AnthropicMessage, AnthropicMessagesRequest, AnthropicMessagesResponse, ContentBlock,
    MessageContent, SystemPrompt,
};
use crate::providers::anthropic::messages::transformation::{
    non_empty, AnthropicMessagesConfig, ANTHROPIC_MESSAGES_CONFIG,
};
use serde_json::{Map, Value};

const AZURE_API_KEY_ENV: &str = "AZURE_API_KEY";
const AZURE_API_BASE_ENV: &str = "AZURE_API_BASE";
const ANTHROPIC_PATH_SEGMENT: &str = "/anthropic";
const MESSAGES_PATH_SUFFIX: &str = "/v1/messages";
const SYSTEM_ROLE: &str = "system";
const TEXT_BLOCK_TYPE: &str = "text";

pub struct AzureAnthropicMessagesConfig {
    anthropic: AnthropicMessagesConfig,
}

pub const AZURE_ANTHROPIC_MESSAGES_CONFIG: AzureAnthropicMessagesConfig =
    AzureAnthropicMessagesConfig {
        anthropic: ANTHROPIC_MESSAGES_CONFIG,
    };

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

    if api_base.ends_with(MESSAGES_PATH_SUFFIX) {
        return Ok(api_base.to_string());
    }

    let with_anthropic = match api_base.split_once(ANTHROPIC_PATH_SEGMENT) {
        Some((prefix, _)) => format!("{prefix}{ANTHROPIC_PATH_SEGMENT}"),
        None => format!("{api_base}{ANTHROPIC_PATH_SEGMENT}"),
    };
    Ok(format!("{with_anthropic}{MESSAGES_PATH_SUFFIX}"))
}

fn strip_scope_from_block(block: &mut ContentBlock) {
    if let Some(cache_control) = block.cache_control.as_mut() {
        cache_control.scope = None;
    }
}

fn strip_scope_from_system(system: &mut SystemPrompt) {
    if let SystemPrompt::Blocks(blocks) = system {
        blocks.iter_mut().for_each(strip_scope_from_block);
    }
}

fn strip_scope_from_message(message: &mut AnthropicMessage) {
    if let MessageContent::Blocks(blocks) = &mut message.content {
        blocks.iter_mut().for_each(strip_scope_from_block);
    }
}

fn text_content_block(text: String) -> ContentBlock {
    let extra = Map::from_iter([
        (
            "type".to_string(),
            Value::String(TEXT_BLOCK_TYPE.to_string()),
        ),
        ("text".to_string(), Value::String(text)),
    ]);
    ContentBlock {
        cache_control: None,
        extra,
    }
}

fn content_into_blocks(content: MessageContent) -> Vec<ContentBlock> {
    match content {
        MessageContent::Text(text) => vec![text_content_block(text)],
        MessageContent::Blocks(blocks) => blocks,
    }
}

fn system_into_blocks(system: Option<SystemPrompt>) -> Vec<ContentBlock> {
    match system {
        None => Vec::new(),
        Some(SystemPrompt::Text(text)) => vec![text_content_block(text)],
        Some(SystemPrompt::Blocks(blocks)) => blocks,
    }
}

fn fold_system_role_messages(request: AnthropicMessagesRequest) -> AnthropicMessagesRequest {
    if !request.messages.iter().any(|msg| msg.role == SYSTEM_ROLE) {
        return request;
    }

    let (system_messages, chat_messages): (Vec<AnthropicMessage>, Vec<AnthropicMessage>) = request
        .messages
        .into_iter()
        .partition(|msg| msg.role == SYSTEM_ROLE);

    let folded_system: Vec<ContentBlock> = system_into_blocks(request.system)
        .into_iter()
        .chain(
            system_messages
                .into_iter()
                .flat_map(|msg| content_into_blocks(msg.content)),
        )
        .collect();

    AnthropicMessagesRequest {
        messages: chat_messages,
        system: (!folded_system.is_empty()).then_some(SystemPrompt::Blocks(folded_system)),
        ..request
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
        self.anthropic.auth_strategy()
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        self.anthropic.default_headers()
    }

    fn transform_request(
        &self,
        request: AnthropicMessagesRequest,
    ) -> CoreResult<AnthropicMessagesRequest> {
        let mut request = fold_system_role_messages(request);
        if let Some(system) = request.system.as_mut() {
            strip_scope_from_system(system);
        }
        request
            .messages
            .iter_mut()
            .for_each(strip_scope_from_message);
        self.anthropic.transform_request(request)
    }

    fn transform_response(
        &self,
        model: &str,
        response: AnthropicMessagesResponse,
    ) -> CoreResult<AnthropicMessagesResponse> {
        self.anthropic.transform_response(model, response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request_from(value: serde_json::Value) -> AnthropicMessagesRequest {
        serde_json::from_value(value).expect("valid request")
    }

    fn to_value(request: AnthropicMessagesRequest) -> serde_json::Value {
        serde_json::to_value(request).expect("serializable request")
    }

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
        let request = request_from(json!({
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
        }));

        let transformed = to_value(
            AZURE_ANTHROPIC_MESSAGES_CONFIG
                .transform_request(request)
                .expect("request transforms"),
        );

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
        let request = request_from(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "system": "plain string system",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let once = AZURE_ANTHROPIC_MESSAGES_CONFIG
            .transform_request(request)
            .expect("request transforms");
        let twice = AZURE_ANTHROPIC_MESSAGES_CONFIG
            .transform_request(once.clone())
            .expect("request transforms");
        assert_eq!(once, twice);
        assert_eq!(to_value(once)["system"], json!("plain string system"));
    }

    #[test]
    fn transform_request_preserves_all_supported_params() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}],
            "system": "be terse",
            "metadata": {"user_id": "u1"},
            "stop_sequences": ["STOP"],
            "stream": false,
            "temperature": 0.4,
            "top_p": 0.9,
            "top_k": 40,
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "service_tier": "auto",
            "container": {"id": "c1"},
            "mcp_servers": [{"type": "url", "url": "https://mcp.example", "name": "x"}],
            "context_management": {"edits": []},
            "output_format": {"type": "json_schema"},
            "output_config": {"effort": "high"},
            "speed": "fast",
            "inference_geo": "us",
            "litellm_metadata": {"trace": "abc"}
        });
        let transformed = to_value(
            AZURE_ANTHROPIC_MESSAGES_CONFIG
                .transform_request(request_from(body.clone()))
                .expect("request transforms"),
        );
        assert_eq!(transformed, body);
    }

    #[test]
    fn transform_request_folds_system_role_message_into_top_level_system() {
        let request = request_from(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 256,
            "system": [{"type": "text", "text": "base system"}],
            "messages": [
                {"role": "user", "content": "fix the bug"},
                {"role": "system", "content": "Available agent types: claude"}
            ]
        }));

        let transformed = to_value(
            AZURE_ANTHROPIC_MESSAGES_CONFIG
                .transform_request(request)
                .expect("request transforms"),
        );

        assert_eq!(
            transformed["messages"],
            json!([{"role": "user", "content": "fix the bug"}])
        );
        assert_eq!(
            transformed["system"],
            json!([
                {"type": "text", "text": "base system"},
                {"type": "text", "text": "Available agent types: claude"}
            ])
        );
    }

    #[test]
    fn transform_request_folds_system_role_when_no_top_level_system() {
        let request = request_from(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "system", "content": [{"type": "text", "text": "sys block"}]}
            ]
        }));

        let transformed = to_value(
            AZURE_ANTHROPIC_MESSAGES_CONFIG
                .transform_request(request)
                .expect("request transforms"),
        );

        assert_eq!(
            transformed["messages"],
            json!([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        );
        assert_eq!(
            transformed["system"],
            json!([{"type": "text", "text": "sys block"}])
        );
    }

    #[test]
    fn transform_request_leaves_requests_without_system_role_untouched() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 256,
            "system": "be terse",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}
            ]
        });
        let transformed = to_value(
            AZURE_ANTHROPIC_MESSAGES_CONFIG
                .transform_request(request_from(body.clone()))
                .expect("request transforms"),
        );
        assert_eq!(transformed, body);
    }

    #[test]
    fn transform_request_rejects_non_object_body() {
        let err = serde_json::from_value::<AnthropicMessagesRequest>(json!("bad"))
            .expect_err("non-object body should error");
        assert!(err.is_data());
    }

    #[test]
    fn transform_response_passes_through() {
        let response: AnthropicMessagesResponse = serde_json::from_value(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 2}
        }))
        .expect("valid response");
        let transformed = AZURE_ANTHROPIC_MESSAGES_CONFIG
            .transform_response("claude-sonnet-4-5", response)
            .expect("response transforms");
        let value = serde_json::to_value(transformed).expect("serializable");
        assert_eq!(value["stop_reason"], json!("end_turn"));
        assert_eq!(value["stop_sequence"], json!(null));
        assert_eq!(value["content"][0]["text"], json!("hello"));
    }
}
