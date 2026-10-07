use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_http::request::{has_bearer_auth, has_header};
use litellm_llms_types::formats::messages::{
    CacheControl, ContentBlock, Message, MessageContent, MessagesOptionalParams, MessagesRequest,
    SystemPrompt,
};

use crate::{
    Error,
    anthropic::{
        common_utils::DEFAULT_ANTHROPIC_HEADERS,
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::{transform_messages_request, update_headers_with_anthropic_beta},
        },
    },
    azure_ai::common_utils::{
        AZURE_API_BASE_ENV, AZURE_API_KEY_ENV, resolve_azure_api_base, resolve_azure_api_key,
    },
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        messages::{
            context::MessagesTransformContext,
            normalization::fold_system_role_messages,
            transformation::{BaseMessagesConfig, MESSAGES_PATH_SUFFIX},
        },
    },
};

const API_KEY_PLACEMENT: CredentialPlacement = CredentialPlacement::Header("x-api-key");
const ANTHROPIC_PATH_SEGMENT: &str = "/anthropic";

pub struct AzureAnthropicMessagesConfig;

pub const AZURE_ANTHROPIC_MESSAGES_CONFIG: AzureAnthropicMessagesConfig =
    AzureAnthropicMessagesConfig;

impl BaseMessagesConfig for AzureAnthropicMessagesConfig {
    fn shape_request(
        &self,
        request: MessagesRequest,
        reasoning_auto_summary: bool,
    ) -> Result<MessagesRequest, Error> {
        shape_anthropic_messages_request(request, reasoning_auto_summary)
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        complete_azure_anthropic_url(api_base, env_lookup)
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        let request = fold_system_role_messages(request);
        transform_messages_request(
            MessagesRequest {
                messages: request
                    .messages
                    .into_iter()
                    .map(strip_scope_from_message)
                    .collect(),
                params: MessagesOptionalParams {
                    system: request.params.system.map(strip_scope_from_system),
                    ..request.params
                },
                ..request
            },
            context,
        )
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[AZURE_API_KEY_ENV, AZURE_API_BASE_ENV]
    }

    /// A forwarded `x-api-key` or a non-blank bearer (an Entra ID token) is the credential;
    /// otherwise the Azure key goes in `x-api-key`.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if has_header(&headers, API_KEY_PLACEMENT.header_name()) || has_bearer_auth(&headers) {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        let auth = AuthScheme::Credential {
            placement: API_KEY_PLACEMENT,
            secret: SecretValue::new(resolve_azure_api_key(api_key, env_lookup)?),
        };
        Ok(ValidatedEnvironment { headers, auth })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_ANTHROPIC_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        update_headers_with_anthropic_beta(headers, request)
    }
}

pub fn complete_azure_anthropic_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let api_base = resolve_azure_api_base(api_base, env_lookup)?;

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

fn strip_scope_from_block(block: ContentBlock) -> ContentBlock {
    ContentBlock {
        cache_control: block
            .cache_control
            .map(|cache_control| match cache_control {
                litellm_llms_types::serde_compat::Nullable::Value(cache_control) => {
                    litellm_llms_types::serde_compat::Nullable::Value(CacheControl {
                        scope: None,
                        ..cache_control
                    })
                }
                other => other,
            }),
        ..block
    }
}

fn strip_scope_from_system(system: SystemPrompt) -> SystemPrompt {
    match system {
        SystemPrompt::Blocks(blocks) => {
            SystemPrompt::Blocks(blocks.into_iter().map(strip_scope_from_block).collect())
        }
        text => text,
    }
}

fn strip_scope_from_message(message: Message) -> Message {
    Message {
        content: match message.content {
            MessageContent::Blocks(blocks) => {
                MessageContent::Blocks(blocks.into_iter().map(strip_scope_from_block).collect())
            }
            text => text,
        },
        ..message
    }
}

#[cfg(test)]
mod tests {
    use litellm_llms_types::formats::messages::MessagesResponse;
    use rstest::rstest;
    use serde_json::json;

    use litellm_auth::CredentialPlacement;

    use super::*;
    use crate::base_llm::messages::context::MessagesModelCapabilities;

    fn request_from(value: serde_json::Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid request")
    }

    fn to_value(request: MessagesRequest) -> serde_json::Value {
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
        assert!(matches!(err, Error::Auth(_)));
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
            Error::Auth(_)
        ));
    }

    fn validated(forwarded: &[(&str, &str)], api_key: Option<&str>) -> ValidatedEnvironment {
        AZURE_ANTHROPIC_MESSAGES_CONFIG
            .validate_environment(
                forwarded
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.to_string()))
                    .collect(),
                api_key,
                "claude",
                &|_| None,
            )
            .unwrap()
    }

    #[test]
    fn the_azure_key_goes_in_x_api_key() {
        assert!(matches!(
            validated(&[], Some("sk-azure")).auth,
            AuthScheme::Credential {
                placement: CredentialPlacement::Header("x-api-key"),
                ref secret
            } if secret.expose() == "sk-azure"
        ));
    }

    #[rstest]
    #[case::x_api_key(&[("X-Api-Key", "caller")])]
    #[case::entra_id_bearer(&[("Authorization", "Bearer eyJ-token")])]
    fn a_forwarded_key_or_bearer_is_the_credential(#[case] forwarded: &[(&str, &str)]) {
        assert!(matches!(
            validated(forwarded, Some("sk-azure")).auth,
            AuthScheme::Forwarded
        ));
    }

    #[test]
    fn a_blank_bearer_does_not_count_as_a_credential() {
        assert!(matches!(
            validated(&[("Authorization", "Bearer  ")], Some("sk-azure")).auth,
            AuthScheme::Credential { .. }
        ));
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
                .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
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
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .expect("request transforms");
        let twice = AZURE_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                once.clone(),
                &MessagesTransformContext::default(),
            )
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
        let context = MessagesTransformContext::with_lookup(
            MessagesModelCapabilities {
                supports_reasoning: true,
                supports_adaptive_thinking: true,
                supports_legacy_thinking: true,
                supports_output_config: true,
                supports_speed: true,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        let transformed = to_value(
            AZURE_ANTHROPIC_MESSAGES_CONFIG
                .transform_anthropic_messages_request(request_from(body.clone()), &context)
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
                .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
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
                .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
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
        let context = MessagesTransformContext::with_lookup(
            MessagesModelCapabilities {
                supports_reasoning: true,
                supports_adaptive_thinking: true,
                supports_legacy_thinking: true,
                supports_output_config: true,
                supports_speed: true,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        let transformed = to_value(
            AZURE_ANTHROPIC_MESSAGES_CONFIG
                .transform_anthropic_messages_request(request_from(body.clone()), &context)
                .expect("request transforms"),
        );
        assert_eq!(transformed, body);
    }

    #[test]
    fn transform_request_rejects_non_object_body() {
        let err = serde_json::from_value::<MessagesRequest>(json!("bad"))
            .expect_err("non-object body should error");
        assert!(err.is_data());
    }

    #[rstest]
    #[case::compact_context_management_edit(
        json!({"context_management": {"edits": [{"type": "compact_20260112"}]}}),
        &[],
        &[("x-api-key", "k"), ("anthropic-beta", "compact-2026-01-12")]
    )]
    #[case::forwarded_beta_merged_with_structured_output(
        json!({"output_config": {"format": {"type": "json_schema"}}}),
        &[("anthropic-beta", "web-search-2025-03-05")],
        &[("x-api-key", "k"), ("anthropic-beta", "structured-outputs-2025-11-13,web-search-2025-03-05")]
    )]
    #[case::no_feature_needs_a_beta(json!({}), &[], &[("x-api-key", "k")])]
    fn request_headers_carry_the_anthropic_feature_betas(
        #[case] fields: serde_json::Value,
        #[case] forwarded: &[(&str, &str)],
        #[case] expected: &[(&str, &str)],
    ) {
        let pairs = |pairs: &[(&str, &str)]| -> Vec<(String, String)> {
            pairs
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect()
        };
        let serde_json::Value::Object(fields) = fields else {
            panic!("case fields are an object")
        };
        let request = request_from(serde_json::Value::Object(
            [
                ("model".to_string(), json!("claude-sonnet")),
                ("max_tokens".to_string(), json!(16)),
                (
                    "messages".to_string(),
                    json!([{"role": "user", "content": "hi"}]),
                ),
            ]
            .into_iter()
            .chain(fields)
            .collect(),
        ));
        assert_eq!(
            AZURE_ANTHROPIC_MESSAGES_CONFIG.request_headers(
                pairs(&[("x-api-key", "k")])
                    .into_iter()
                    .chain(pairs(forwarded))
                    .collect(),
                &request
            ),
            pairs(expected)
        );
    }

    #[test]
    fn transform_response_passes_through() {
        let response: MessagesResponse = serde_json::from_value(json!({
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
            .transform_anthropic_messages_response("claude-sonnet-4-5", response)
            .expect("response transforms");
        let value = serde_json::to_value(transformed).expect("serializable");
        assert_eq!(value["stop_reason"], json!("end_turn"));
        assert_eq!(value["stop_sequence"], json!(null));
        assert_eq!(value["content"][0]["text"], json!("hello"));
    }

    #[test]
    fn secret_names_cover_every_credential_and_base_lookup() {
        let requested = std::cell::RefCell::new(Vec::<String>::new());
        let record = |name: &str| -> Option<String> {
            requested.borrow_mut().push(name.to_string());
            None
        };
        let _ = AZURE_ANTHROPIC_MESSAGES_CONFIG.validate_environment(
            Vec::new(),
            None,
            "claude",
            &record,
        );
        let _ = AZURE_ANTHROPIC_MESSAGES_CONFIG.get_complete_url(None, "claude", &record);
        let requested = requested.into_inner();
        assert!(!requested.is_empty());
        let undeclared: Vec<&String> = requested
            .iter()
            .filter(|name| {
                !AZURE_ANTHROPIC_MESSAGES_CONFIG
                    .secret_names()
                    .contains(&name.as_str())
            })
            .collect();
        assert_eq!(undeclared, Vec::<&String>::new());
    }
}
