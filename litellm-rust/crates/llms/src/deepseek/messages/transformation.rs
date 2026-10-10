use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_core_utils::settings::resolve_non_empty;
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest, MessagesTool},
    providers::anthropic::{DEFAULT_HEADERS, MESSAGES_PATH},
    recognized::Recognized,
};
use litellm_router_types::LitellmParams;
use serde_json::Value;

use crate::{
    Error,
    anthropic::{
        common_utils::{filter_billing_headers_from_system, has_anthropic_credential},
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::{transform_messages_request, update_headers_with_anthropic_beta},
        },
    },
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        messages::{context::MessagesTransformContext, transformation::BaseMessagesConfig},
    },
};

pub const DEEPSEEK_API_KEY_ENV: &str = "DEEPSEEK_API_KEY";
pub const DEEPSEEK_ANTHROPIC_API_BASE_ENV: &str = "DEEPSEEK_ANTHROPIC_API_BASE";
pub const DEEPSEEK_API_BASE_ENV: &str = "DEEPSEEK_API_BASE";
pub const DEFAULT_DEEPSEEK_ANTHROPIC_API_BASE: &str = "https://api.deepseek.com/anthropic";

const ANTHROPIC_PATH_SEGMENT: &str = "/anthropic";
const CUSTOM_TOOL_TYPE: &str = "custom";

/// DeepSeek's Anthropic-compatible Messages API: the Anthropic payload shaping, a key in
/// `x-api-key`, and a host that rejects the explicit `custom` tool discriminator and the
/// first-party billing system blocks.
pub struct DeepSeekAnthropicMessagesConfig;

pub const DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG: DeepSeekAnthropicMessagesConfig =
    DeepSeekAnthropicMessagesConfig;

impl BaseMessagesConfig for DeepSeekAnthropicMessagesConfig {
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
        _litellm_params: &LitellmParams,
        _stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(complete_deepseek_anthropic_url(api_base, env_lookup))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        let request = transform_messages_request(
            MessagesRequest {
                params: MessagesOptionalParams {
                    system: request
                        .params
                        .system
                        .and_then(filter_billing_headers_from_system),
                    ..request.params
                },
                ..request
            },
            context,
        )?;
        Ok(MessagesRequest {
            params: MessagesOptionalParams {
                tools: request
                    .params
                    .tools
                    .map(|tools| tools.into_iter().map(sanitize_tool).collect()),
                ..request.params
            },
            ..request
        })
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[
            DEEPSEEK_API_KEY_ENV,
            DEEPSEEK_ANTHROPIC_API_BASE_ENV,
            DEEPSEEK_API_BASE_ENV,
        ]
    }

    /// A forwarded `x-api-key` or `authorization` is the credential; otherwise the DeepSeek
    /// key goes in `x-api-key`.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _litellm_params: &LitellmParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if has_anthropic_credential(&headers) {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        let key = get_api_key(api_key, env_lookup).ok_or(Error::Auth(
            litellm_auth::Error::MissingApiKey {
                provider: "DeepSeek",
                environment_variable: DEEPSEEK_API_KEY_ENV,
            },
        ))?;
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::Credential {
                placement: CredentialPlacement::Header("x-api-key"),
                secret: SecretValue::new(key),
            },
        })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        update_headers_with_anthropic_beta(headers, request)
    }
}

pub fn get_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    resolve_non_empty(api_key, env_lookup, &[DEEPSEEK_API_KEY_ENV])
}

pub fn get_api_base(api_base: Option<&str>, env_lookup: &dyn Fn(&str) -> Option<String>) -> String {
    resolve_non_empty(
        api_base,
        env_lookup,
        &[DEEPSEEK_ANTHROPIC_API_BASE_ENV, DEEPSEEK_API_BASE_ENV],
    )
    .unwrap_or_else(|| DEFAULT_DEEPSEEK_ANTHROPIC_API_BASE.to_string())
}

/// Python's `get_complete_url`: a base already ending in the Anthropic messages path is kept,
/// otherwise the OpenAI compatible suffixes are peeled off and `/anthropic/v1/messages` is
/// appended once.
pub fn complete_deepseek_anthropic_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let api_base = get_api_base(api_base, env_lookup);
    let api_base = api_base.trim_end_matches('/');
    if api_base.ends_with(MESSAGES_PATH) && api_base.contains("/anthropic/") {
        return api_base.to_string();
    }
    let api_base = [MESSAGES_PATH, "/v1", "/beta"]
        .into_iter()
        .fold(api_base, |base, suffix| {
            base.strip_suffix(suffix).unwrap_or(base)
        });
    if api_base.ends_with(ANTHROPIC_PATH_SEGMENT) || api_base.contains("/anthropic/") {
        return format!("{api_base}{MESSAGES_PATH}");
    }
    format!("{api_base}{ANTHROPIC_PATH_SEGMENT}{MESSAGES_PATH}")
}

/// DeepSeek rejects Anthropic's explicit `{"type": "custom"}` tool discriminator, so it is
/// dropped; every other tool is sent as given.
fn sanitize_tool(tool: Recognized<MessagesTool>) -> Recognized<MessagesTool> {
    match tool {
        Recognized::Unrecognized(Value::Object(fields))
            if fields.get("type").and_then(Value::as_str) == Some(CUSTOM_TOOL_TYPE) =>
        {
            Recognized::Unrecognized(Value::Object(
                fields
                    .into_iter()
                    .filter(|(name, _)| name != "type")
                    .collect(),
            ))
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use litellm_auth::CredentialPlacement;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn request_from(value: Value) -> MessagesRequest {
        serde_json::from_value(value).unwrap()
    }

    fn transformed(value: Value) -> Value {
        serde_json::to_value(
            DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG
                .transform_anthropic_messages_request(
                    request_from(value),
                    &MessagesTransformContext::default(),
                )
                .unwrap(),
        )
        .unwrap()
    }

    #[rstest]
    #[case::no_base(None)]
    #[case::anthropic_v1("https://api.deepseek.com/anthropic/v1")]
    #[case::anthropic("https://api.deepseek.com/anthropic")]
    #[case::anthropic_with_trailing_slash("https://api.deepseek.com/anthropic/")]
    #[case::host_only("https://api.deepseek.com")]
    #[case::openai_v1("https://api.deepseek.com/v1")]
    #[case::openai_messages("https://api.deepseek.com/v1/messages")]
    #[case::openai_beta("https://api.deepseek.com/beta")]
    #[case::complete("https://api.deepseek.com/anthropic/v1/messages")]
    fn every_base_spelling_reaches_the_anthropic_messages_endpoint(
        #[case] api_base: impl Into<Option<&'static str>>,
    ) {
        assert_eq!(
            complete_deepseek_anthropic_url(api_base.into(), &no_env),
            "https://api.deepseek.com/anthropic/v1/messages"
        );
    }

    #[rstest]
    #[case::proxy_under_anthropic_prefix(
        "https://proxy.example/anthropic/v1/messages",
        "https://proxy.example/anthropic/v1/messages"
    )]
    #[case::messages_path_without_the_anthropic_segment_is_rebuilt(
        "https://proxy.example/v1/messages",
        "https://proxy.example/anthropic/v1/messages"
    )]
    #[case::anthropic_segment_in_the_middle_is_kept(
        "https://proxy.example/anthropic/tenant",
        "https://proxy.example/anthropic/tenant/v1/messages"
    )]
    fn custom_bases_follow_the_python_url_rules(#[case] api_base: &str, #[case] expected: &str) {
        assert_eq!(
            complete_deepseek_anthropic_url(Some(api_base), &no_env),
            expected
        );
    }

    #[rstest]
    #[case::anthropic_base_wins(
        &[("DEEPSEEK_ANTHROPIC_API_BASE", "https://anthropic.example"), ("DEEPSEEK_API_BASE", "https://openai.example/v1")],
        "https://anthropic.example/anthropic/v1/messages"
    )]
    #[case::openai_base_is_the_fallback(
        &[("DEEPSEEK_API_BASE", "https://openai.example/v1")],
        "https://openai.example/anthropic/v1/messages"
    )]
    #[case::blank_env_is_absent(
        &[("DEEPSEEK_ANTHROPIC_API_BASE", " "), ("DEEPSEEK_API_BASE", "")],
        "https://api.deepseek.com/anthropic/v1/messages"
    )]
    fn the_base_comes_from_the_environment_in_python_order(
        #[case] env: &[(&str, &str)],
        #[case] expected: &str,
    ) {
        let lookup = |name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        };
        assert_eq!(complete_deepseek_anthropic_url(None, &lookup), expected);
    }

    fn validated(
        forwarded: &[(&str, &str)],
        api_key: Option<&str>,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG.validate_environment(
            forwarded
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
            api_key,
            "deepseek-v4-pro",
            &LitellmParams::default(),
            env,
        )
    }

    #[rstest]
    #[case::param(Some("sk-deepseek"), &[], "sk-deepseek")]
    #[case::env(None, &[("DEEPSEEK_API_KEY", "sk-env")], "sk-env")]
    #[case::blank_param_falls_back_to_env(Some(" "), &[("DEEPSEEK_API_KEY", "sk-env")], "sk-env")]
    fn the_deepseek_key_goes_in_x_api_key(
        #[case] api_key: Option<&str>,
        #[case] env: &[(&str, &str)],
        #[case] expected: &str,
    ) {
        let lookup = |name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        };
        let validated = validated(&[], api_key, &lookup).unwrap();
        assert!(matches!(
            validated.auth,
            AuthScheme::Credential {
                placement: CredentialPlacement::Header("x-api-key"),
                ref secret
            } if secret.expose() == expected
        ));
        assert!(validated.headers.is_empty());
    }

    #[rstest]
    #[case::x_api_key(&[("X-Api-Key", "caller")])]
    #[case::bearer(&[("Authorization", "Bearer caller")])]
    fn a_forwarded_credential_is_sent_as_is(#[case] forwarded: &[(&str, &str)]) {
        assert!(matches!(
            validated(forwarded, Some("sk-deepseek"), &no_env)
                .unwrap()
                .auth,
            AuthScheme::Forwarded
        ));
    }

    #[rstest]
    fn a_call_without_a_key_names_the_deepseek_variable() {
        assert!(matches!(
            validated(&[], None, &no_env).unwrap_err(),
            Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "DeepSeek",
                environment_variable: "DEEPSEEK_API_KEY",
            })
        ));
    }

    #[rstest]
    fn secret_names_cover_the_key_and_both_bases() {
        assert_eq!(
            DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG.secret_names(),
            &[
                "DEEPSEEK_API_KEY",
                "DEEPSEEK_ANTHROPIC_API_BASE",
                "DEEPSEEK_API_BASE"
            ]
        );
    }

    #[rstest]
    fn default_headers_are_anthropic_version_and_json() {
        assert_eq!(
            DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG.default_headers(),
            &[
                ("anthropic-version", "2023-06-01"),
                ("content-type", "application/json"),
            ]
        );
    }

    #[rstest]
    fn the_custom_discriminator_is_dropped_and_other_tools_are_kept() {
        let body = transformed(json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Use the tool."}],
            "tools": [
                {"type": "custom", "name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}},
                {"name": "untyped", "input_schema": {"type": "object"}},
                {"type": "web_search_20260209", "name": "web_search", "max_uses": 1},
                {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
            ]
        }));
        assert_eq!(
            body["tools"],
            json!([
                {"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}},
                {"name": "untyped", "input_schema": {"type": "object"}},
                {"type": "web_search_20260209", "name": "web_search", "max_uses": 1},
                {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
            ])
        );
    }

    #[rstest]
    #[case::billing_block_among_others(
        json!([
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
            {"type": "text", "text": "real system prompt"}
        ]),
        Some(json!([{"type": "text", "text": "real system prompt"}]))
    )]
    #[case::only_billing_blocks(
        json!([{"type": "text", "text": "x-anthropic-billing-header: cc_version=1"}]),
        None
    )]
    #[case::billing_string(json!("x-anthropic-billing-header: cc_version=1"), None)]
    #[case::ordinary_string(json!("be terse"), Some(json!("be terse")))]
    #[case::billing_text_in_a_non_text_block_is_kept(
        json!([{"type": "other", "text": "x-anthropic-billing-header: cc_version=1"}]),
        Some(json!([{"type": "other", "text": "x-anthropic-billing-header: cc_version=1"}]))
    )]
    fn billing_metadata_is_stripped_from_the_system_prompt(
        #[case] system: Value,
        #[case] expected: Option<Value>,
    ) {
        let body = transformed(json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 100,
            "system": system,
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(body.get("system").cloned(), expected);
    }

    #[rstest]
    fn thinking_history_and_params_pass_through() {
        let messages = json!([
            {"role": "user", "content": "Use the tool."},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I should call the tool.", "signature": "sig"},
                {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"city": "Sao Paulo"}}
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_123", "content": "Sunny"}]}
        ]);
        let body = transformed(json!({
            "model": "deepseek-v4-pro",
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": messages
        }));
        assert_eq!(body["messages"], messages);
        assert_eq!(
            body["thinking"],
            json!({"type": "enabled", "budget_tokens": 1024})
        );
    }
}
