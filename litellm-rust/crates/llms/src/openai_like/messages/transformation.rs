use litellm_auth::{
    CredentialPlacement, CredentialPlanKind, CredentialRule, ExistingHeaderBehavior,
    ProviderAuthPolicy, SecretValue,
};
use litellm_llms_types::{
    formats::messages::{
        BlockContent, CacheControl, ContentBlock, ContentBlockPayload, ContentBlockType,
        MessagesOptionalParams, MessagesRequest, MessagesTool, ToolDefinition,
    },
    recognized::Recognized,
    serde_compat::Nullable,
};
use serde_json::{Map, Value};

use crate::{
    Error,
    anthropic::{
        common_utils::DEFAULT_ANTHROPIC_HEADERS,
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::{
                BillingMetadata, COMPATIBLE_HOST_REQUEST_POLICY, RequestPolicy,
                transform_messages_request_with, update_headers_with_anthropic_beta,
            },
        },
    },
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        messages::{
            context::MessagesTransformContext,
            normalization::map_request_blocks,
            transformation::{BaseMessagesConfig, complete_messages_url},
        },
    },
    openai_like::common_utils::{JsonProvider, non_blank},
};

pub const OPENAI_LIKE_MESSAGES_AUTH_POLICY: ProviderAuthPolicy = ProviderAuthPolicy {
    rules: &[CredentialRule {
        kind: CredentialPlanKind::Static,
        placement: CredentialPlacement::Bearer,
    }],
    accepted_existing_headers: &["authorization", "x-api-key"],
    existing_header_behavior: ExistingHeaderBehavior::Preserve,
    scope: None,
    audience: None,
};

/// Python's `OpenAILikeAnthropicMessagesConfig` (`Deployment`, opted into through
/// `model_info.supported_endpoints`) and `JSONProviderAnthropicMessagesConfig` (`Registry`, a
/// `providers.json` entry listing `/v1/messages`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenAILikeMessagesConfig {
    Deployment { cache_control_ttl: bool },
    Registry(&'static JsonProvider),
}

pub const OPENAI_LIKE_MESSAGES_CONFIG: OpenAILikeMessagesConfig =
    OpenAILikeMessagesConfig::Deployment {
        cache_control_ttl: false,
    };

pub const OPENAI_LIKE_MESSAGES_CONFIG_WITH_CACHE_CONTROL_TTL: OpenAILikeMessagesConfig =
    OpenAILikeMessagesConfig::Deployment {
        cache_control_ttl: true,
    };

impl OpenAILikeMessagesConfig {
    pub fn request_policy(self) -> RequestPolicy {
        RequestPolicy {
            billing_metadata: match self {
                Self::Deployment { .. } => BillingMetadata::Forward,
                Self::Registry(_) => BillingMetadata::Strip,
            },
            ..COMPATIBLE_HOST_REQUEST_POLICY
        }
    }

    pub fn keeps_cache_control_ttl(self) -> bool {
        match self {
            Self::Deployment { cache_control_ttl } => cache_control_ttl,
            Self::Registry(provider) => provider.cache_control_ttl,
        }
    }

    pub fn resolve_api_key(
        self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String> {
        match self {
            Self::Deployment { .. } => non_blank(api_key).map(str::to_string),
            Self::Registry(provider) => provider.resolve_api_key(api_key, env_lookup),
        }
    }
}

impl BaseMessagesConfig for OpenAILikeMessagesConfig {
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
        match self {
            Self::Deployment { .. } => non_blank(api_base)
                .map(complete_messages_url)
                .ok_or(Error::MissingField("api_base")),
            Self::Registry(provider) => Ok(complete_messages_url(
                &provider.resolve_api_base(api_base, env_lookup),
            )),
        }
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        let request = transform_messages_request_with(request, context, self.request_policy())?;
        if self.keeps_cache_control_ttl() {
            return Ok(request);
        }
        Ok(with_portable_cache_control(request))
    }

    fn secret_names(&self) -> &'static [&'static str] {
        match self {
            Self::Deployment { .. } => &[],
            Self::Registry(provider) => provider.secret_names(),
        }
    }

    /// Keyless servers are valid, so a missing key sends no credential instead of failing.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let key = self.resolve_api_key(api_key, env_lookup);
        let auth = match key {
            Some(key) if !OPENAI_LIKE_MESSAGES_AUTH_POLICY.has_existing_credential(&headers) => {
                AuthScheme::Credential {
                    placement: CredentialPlacement::Bearer,
                    secret: SecretValue::new(key),
                }
            }
            _ => AuthScheme::Forwarded,
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

/// Python's `normalize_cache_control_in_anthropic_payload`: every `cache_control` the Messages
/// API defines (request, system blocks, tools, message blocks, `tool_result` content) becomes
/// `{"type": <its type, or "ephemeral">}`, and a non-object one is dropped. Application data
/// such as `tool_use.input` and `input_schema` is never touched.
pub fn with_portable_cache_control(request: MessagesRequest) -> MessagesRequest {
    let request = map_request_blocks(request, portable_block, portable_message_block);
    MessagesRequest {
        params: MessagesOptionalParams {
            cache_control: request
                .params
                .cache_control
                .and_then(portable_recognized_cache),
            tools: request
                .params
                .tools
                .map(|tools| tools.into_iter().map(portable_tool).collect()),
            ..request.params
        },
        ..request
    }
}

fn portable_cache_control(cache_control: CacheControl) -> CacheControl {
    CacheControl {
        cache_type: Some(Nullable::Value(
            cache_control
                .cache_type
                .and_then(Nullable::into_value)
                .unwrap_or_else(|| "ephemeral".to_string()),
        )),
        ..CacheControl::default()
    }
}

fn portable_recognized_cache(cache: Recognized<CacheControl>) -> Option<Recognized<CacheControl>> {
    match cache {
        Recognized::Known(cache) => Some(Recognized::Known(portable_cache_control(cache))),
        Recognized::Unrecognized(value) => {
            portable_cache_value(value).map(Recognized::Unrecognized)
        }
    }
}

fn portable_cache_value(cache_control: Value) -> Option<Value> {
    let Value::Object(cache_control) = cache_control else {
        return None;
    };
    let cache_type = cache_control
        .into_iter()
        .find_map(|(key, value)| match (key.as_str(), value) {
            ("type", Value::String(value)) => Some(value),
            _ => None,
        })
        .unwrap_or_else(|| "ephemeral".to_string());
    Some(Value::Object(
        [("type".to_string(), Value::String(cache_type))]
            .into_iter()
            .collect(),
    ))
}

fn portable_tool(tool: Recognized<MessagesTool>) -> Recognized<MessagesTool> {
    match tool {
        Recognized::Known(tool) => {
            Recognized::Known(tool.map_definition(|definition| ToolDefinition {
                cache_control: definition.cache_control.and_then(portable_recognized_cache),
                ..definition
            }))
        }
        Recognized::Unrecognized(Value::Object(fields)) => {
            Recognized::Unrecognized(Value::Object(portable_object(fields)))
        }
        other => other,
    }
}

fn portable_block(block: ContentBlock) -> ContentBlock {
    ContentBlock {
        cache_control: block.cache_control.and_then(|cache| match cache {
            Nullable::Value(cache) => Some(Nullable::Value(portable_cache_control(cache))),
            Nullable::Null => None,
        }),
        ..block
    }
}

fn portable_object(block: Map<String, Value>) -> Map<String, Value> {
    if !block.contains_key("cache_control") {
        return block;
    }
    block
        .into_iter()
        .filter_map(|(key, value)| match key.as_str() {
            "cache_control" => portable_cache_value(value).map(|value| (key, value)),
            _ => Some((key, value)),
        })
        .collect()
}

fn portable_unknown_blocks(blocks: Value) -> Value {
    match blocks {
        Value::Array(blocks) => blocks
            .into_iter()
            .map(|block| match block {
                Value::Object(block) => Value::Object(portable_object(block)),
                other => other,
            })
            .collect(),
        other => other,
    }
}

fn portable_message_block(block: ContentBlock) -> ContentBlock {
    let block = portable_block(block);
    if !block.is_type(ContentBlockType::ToolResult) {
        return block;
    }
    ContentBlock {
        payload: ContentBlockPayload {
            content: block.payload.content.map(|content| match content {
                Recognized::Known(BlockContent::Blocks(blocks)) => {
                    Recognized::Known(BlockContent::Blocks(
                        blocks
                            .into_iter()
                            .map(|block| match block {
                                Recognized::Known(block) => {
                                    Recognized::Known(portable_block(block))
                                }
                                Recognized::Unrecognized(Value::Object(fields)) => {
                                    Recognized::Unrecognized(Value::Object(portable_object(fields)))
                                }
                                other => other,
                            })
                            .collect(),
                    ))
                }
                Recognized::Unrecognized(value) => {
                    Recognized::Unrecognized(portable_unknown_blocks(value))
                }
                other => other,
            }),
            ..block.payload
        },
        ..block
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use litellm_llms_types::formats::messages::MessagesResponse;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    static REGISTRY_PROVIDER: JsonProvider = JsonProvider::new(
        "https://registry.example/v1",
        "REGISTRY_API_KEY",
        Some("REGISTRY_API_BASE"),
        false,
    );
    static REGISTRY_PROVIDER_WITHOUT_BASE_ENV: JsonProvider =
        JsonProvider::new("https://keyonly.example", "KEYONLY_API_KEY", None, false);
    static REGISTRY_PROVIDER_WITH_TTL: JsonProvider = JsonProvider::new(
        "https://ttl.example/v1",
        "TTL_API_KEY",
        Some("TTL_API_BASE"),
        true,
    );

    const REGISTRY: OpenAILikeMessagesConfig =
        OpenAILikeMessagesConfig::Registry(&REGISTRY_PROVIDER);

    fn request_from(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid request")
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn transformed(config: OpenAILikeMessagesConfig, body: Value) -> Value {
        serde_json::to_value(
            config
                .transform_anthropic_messages_request(
                    request_from(body),
                    &MessagesTransformContext::default(),
                )
                .expect("request transforms"),
        )
        .expect("serializable request")
    }

    fn cache_controlled_body(cache_control: Value) -> Value {
        json!({
            "model": "m",
            "max_tokens": 16,
            "cache_control": cache_control,
            "system": [{"type": "text", "text": "sys", "cache_control": cache_control}],
            "tools": [{
                "name": "lookup",
                "input_schema": {"type": "object", "properties": {"cache_control": {"type": "string", "ttl": "1h"}}},
                "cache_control": cache_control
            }],
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi", "cache_control": cache_control},
                    {"type": "tool_result", "tool_use_id": "t1", "cache_control": cache_control, "content": [
                        {"type": "text", "text": "out", "cache_control": cache_control}
                    ]}
                ]},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t2", "name": "lookup", "input": {"cache_control": {"ttl": "1h"}}}
                ]},
                {"role": "user", "content": "a plain string message"}
            ]
        })
    }

    fn cache_controls(body: &Value) -> Vec<Value> {
        [
            &body["cache_control"],
            &body["system"][0]["cache_control"],
            &body["tools"][0]["cache_control"],
            &body["messages"][0]["content"][0]["cache_control"],
            &body["messages"][0]["content"][1]["cache_control"],
            &body["messages"][0]["content"][1]["content"][0]["cache_control"],
        ]
        .into_iter()
        .cloned()
        .collect()
    }

    #[rstest]
    #[case::deployment_default(OPENAI_LIKE_MESSAGES_CONFIG)]
    #[case::registry_default(REGISTRY)]
    fn ttl_is_stripped_from_every_scoped_cache_control(#[case] config: OpenAILikeMessagesConfig) {
        let body = transformed(
            config,
            cache_controlled_body(json!({"type": "ephemeral", "ttl": "1h", "scope": "global"})),
        );
        assert_eq!(cache_controls(&body), vec![json!({"type": "ephemeral"}); 6]);
        assert_eq!(
            body["messages"][2],
            json!({"role": "user", "content": "a plain string message"})
        );
        assert_eq!(
            body["tools"][0]["input_schema"]["properties"]["cache_control"],
            json!({"type": "string", "ttl": "1h"})
        );
        assert_eq!(
            body["messages"][1]["content"][0]["input"],
            json!({"cache_control": {"ttl": "1h"}})
        );
    }

    #[rstest]
    #[case::deployment_opt_in(OPENAI_LIKE_MESSAGES_CONFIG_WITH_CACHE_CONTROL_TTL)]
    #[case::registry_constraint(OpenAILikeMessagesConfig::Registry(&REGISTRY_PROVIDER_WITH_TTL))]
    fn ttl_is_kept_when_the_deployment_or_registry_allows_it(
        #[case] config: OpenAILikeMessagesConfig,
    ) {
        let cache_control = json!({"type": "ephemeral", "ttl": "1h"});
        let body = transformed(config, cache_controlled_body(cache_control.clone()));
        assert_eq!(cache_controls(&body), vec![cache_control; 6]);
    }

    #[test]
    fn a_cache_control_without_a_type_becomes_ephemeral_and_a_null_one_is_dropped() {
        let body = transformed(
            OPENAI_LIKE_MESSAGES_CONFIG,
            json!({
                "model": "m",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "a", "cache_control": {"ttl": "5m"}},
                    {"type": "text", "text": "b", "cache_control": null}
                ]}]
            }),
        );
        assert_eq!(
            body["messages"][0]["content"],
            json!([
                {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "b"}
            ])
        );
    }

    #[rstest]
    #[case::unknown_tool(
        json!({"model": "m", "messages": [], "tools": [{
            "type": "future_tool", "name": "lookup", "cache_control": {"type": "future_cache", "ttl": "1h"},
            "input_schema": {"cache_control": {"ttl": "schema-data"}},
            "custom": {"cache_control": {"ttl": "application-data"}}
        }]}),
        json!({"model": "m", "messages": [], "tools": [{
            "type": "future_tool", "name": "lookup", "cache_control": {"type": "future_cache"},
            "input_schema": {"cache_control": {"ttl": "schema-data"}},
            "custom": {"cache_control": {"ttl": "application-data"}}
        }]})
    )]
    #[case::malformed_nested_block(
        json!({"model": "m", "messages": [{"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "t", "content": [
                {"type": 9, "cache_control": {"type": null, "ttl": "1h"}, "extra": "kept"},
                false
            ]
        }]}]}),
        json!({"model": "m", "messages": [{"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "t", "content": [
                {"type": 9, "cache_control": {"type": "ephemeral"}, "extra": "kept"},
                false
            ]
        }]}]})
    )]
    #[case::malformed_and_null_cache_values(
        json!({"model": "m", "messages": [], "cache_control": {"type": 9}, "tools": [
            {"type": "web_search_20250305", "name": "web_search", "cache_control": false},
            {"type": "future_tool", "cache_control": null},
            null
        ]}),
        json!({"model": "m", "messages": [], "cache_control": {"type": "ephemeral"}, "tools": [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "future_tool"},
            null
        ]})
    )]
    #[case::content_only_recurses_at_message_tool_results(
        json!({"model": "m", "system": [{"type": "tool_result", "content": [{
            "cache_control": {"ttl": "system-content-data"}
        }]}], "messages": [{"role": "user", "content": [
            {"type": "future_block", "content": [{"cache_control": {"ttl": "unknown-content-data"}}]},
            {"type": "tool_result", "content": [{"type": "tool_result", "cache_control": {"ttl": "1h"}, "content": [
                {"cache_control": {"ttl": "deep-content-data"}}
            ]}]}
        ]}]}),
        json!({"model": "m", "system": [{"type": "tool_result", "content": [{
            "cache_control": {"ttl": "system-content-data"}
        }]}], "messages": [{"role": "user", "content": [
            {"type": "future_block", "content": [{"cache_control": {"ttl": "unknown-content-data"}}]},
            {"type": "tool_result", "content": [{"type": "tool_result", "cache_control": {"type": "ephemeral"}, "content": [
                {"cache_control": {"ttl": "deep-content-data"}}
            ]}]}
        ]}]})
    )]
    fn portable_cache_control_preserves_unrecognized_payloads_and_application_data(
        #[case] input: Value,
        #[case] expected: Value,
    ) {
        let once = with_portable_cache_control(request_from(input));
        assert_eq!(serde_json::to_value(&once).unwrap(), expected);
        assert_eq!(with_portable_cache_control(once.clone()), once);
    }

    #[rstest]
    #[case::deployment_keeps_them(OPENAI_LIKE_MESSAGES_CONFIG, true)]
    #[case::registry_strips_them(REGISTRY, false)]
    fn billing_metadata_system_blocks(
        #[case] config: OpenAILikeMessagesConfig,
        #[case] kept: bool,
    ) {
        let billing = json!({"type": "text", "text": "x-anthropic-billing-header: cc_version=1"});
        let body = transformed(
            config,
            json!({
                "model": "m",
                "max_tokens": 16,
                "system": [billing.clone(), {"type": "text", "text": "be terse"}],
                "messages": [{"role": "user", "content": "hi"}]
            }),
        );
        assert_eq!(
            body["system"]
                .as_array()
                .expect("system blocks")
                .contains(&billing),
            kept
        );
        assert_eq!(
            body["system"].as_array().expect("system blocks").last(),
            Some(&json!({"type": "text", "text": "be terse"}))
        );
    }

    #[rstest]
    #[case::deployment(OPENAI_LIKE_MESSAGES_CONFIG)]
    #[case::registry(REGISTRY)]
    fn thinking_and_temperature_pass_through_untouched(#[case] config: OpenAILikeMessagesConfig) {
        for thinking in [
            json!({"type": "disabled"}),
            json!({"type": "enabled", "budget_tokens": 1024}),
            json!({"type": "adaptive"}),
        ] {
            let body = json!({
                "model": "m",
                "max_tokens": 4096,
                "temperature": 0.3,
                "thinking": thinking,
                "messages": [{"role": "user", "content": "hi"}]
            });
            assert_eq!(transformed(config, body.clone()), body);
        }
    }

    #[test]
    fn max_tokens_is_still_required() {
        assert_eq!(
            OPENAI_LIKE_MESSAGES_CONFIG.transform_anthropic_messages_request(
                request_from(
                    json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
                ),
                &MessagesTransformContext::default(),
            ),
            Err(Error::MissingField("max_tokens"))
        );
    }

    #[rstest]
    #[case::bare_host("https://h.example", "https://h.example/v1/messages")]
    #[case::version_no_slash("https://host/v1", "https://host/v1/messages")]
    #[case::anthropic_surface(
        "https://api.deepseek.com/anthropic",
        "https://api.deepseek.com/anthropic/v1/messages"
    )]
    #[case::anthropic_version(
        "https://api.deepseek.com/anthropic/v1",
        "https://api.deepseek.com/anthropic/v1/messages"
    )]
    #[case::version_suffix("https://h.example/v1/", "https://h.example/v1/messages")]
    #[case::complete_endpoint("https://h.example/x/v1/messages", "https://h.example/x/v1/messages")]
    fn deployment_url_completes_the_given_base(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(
            OPENAI_LIKE_MESSAGES_CONFIG.get_complete_url(Some(base), "m", &|_| None),
            Ok(expected.to_string())
        );
    }

    #[rstest]
    #[case::absent(None)]
    #[case::blank(Some(" "))]
    fn deployment_url_requires_an_api_base_and_ignores_env(#[case] base: Option<&str>) {
        assert_eq!(
            OPENAI_LIKE_MESSAGES_CONFIG
                .get_complete_url(base, "m", &|_| Some("https://env.example".to_string())),
            Err(Error::MissingField("api_base"))
        );
    }

    #[rstest]
    #[case::param_wins(
        Some("https://param.example"),
        Some("https://env.example"),
        "https://param.example/v1/messages"
    )]
    #[case::env_over_default(
        None,
        Some("https://env.example/v1"),
        "https://env.example/v1/messages"
    )]
    #[case::blank_param_falls_through(
        Some(""),
        Some("https://env.example"),
        "https://env.example/v1/messages"
    )]
    #[case::registry_default(None, None, "https://registry.example/v1/messages")]
    fn registry_url_resolves_param_then_env_then_default(
        #[case] param: Option<&str>,
        #[case] env: Option<&str>,
        #[case] expected: &str,
    ) {
        let lookup = |name: &str| {
            (name == "REGISTRY_API_BASE")
                .then_some(env)
                .flatten()
                .map(str::to_string)
        };
        assert_eq!(
            REGISTRY.get_complete_url(param, "m", &lookup),
            Ok(expected.to_string())
        );
    }

    fn auth_for(
        config: OpenAILikeMessagesConfig,
        forwarded: &[(&str, &str)],
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> ValidatedEnvironment {
        config
            .validate_environment(headers(forwarded), api_key, "m", env_lookup)
            .expect("environment validates")
    }

    fn bearer_secret(validated: &ValidatedEnvironment) -> Option<String> {
        match &validated.auth {
            AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret,
            } => Some(secret.expose().to_string()),
            _ => None,
        }
    }

    #[rstest]
    #[case::deployment(OPENAI_LIKE_MESSAGES_CONFIG)]
    #[case::registry(REGISTRY)]
    fn the_key_is_sent_as_a_bearer(#[case] config: OpenAILikeMessagesConfig) {
        let validated = auth_for(config, &[("x-trace", "1")], Some("sk-param"), &|_| None);
        assert_eq!(bearer_secret(&validated), Some("sk-param".to_string()));
        assert_eq!(validated.headers, headers(&[("x-trace", "1")]));
    }

    #[rstest]
    #[case::authorization(&[("Authorization", "Bearer caller")])]
    #[case::x_api_key(&[("X-Api-Key", "caller")])]
    fn a_caller_credential_is_never_overridden(#[case] forwarded: &[(&str, &str)]) {
        for config in [OPENAI_LIKE_MESSAGES_CONFIG, REGISTRY] {
            let validated = auth_for(config, forwarded, Some("sk-param"), &|_| None);
            assert!(matches!(validated.auth, AuthScheme::Forwarded));
            assert_eq!(validated.headers, headers(forwarded));
        }
    }

    #[rstest]
    #[case::deployment(OPENAI_LIKE_MESSAGES_CONFIG)]
    #[case::registry(REGISTRY)]
    fn a_keyless_server_gets_no_credential(#[case] config: OpenAILikeMessagesConfig) {
        for api_key in [None, Some("  ")] {
            let validated = auth_for(config, &[], api_key, &|_| None);
            assert!(matches!(validated.auth, AuthScheme::Forwarded));
            assert_eq!(validated.headers, Headers::new());
        }
    }

    #[test]
    fn only_the_registry_falls_back_to_its_key_env() {
        let lookup = |name: &str| (name == "REGISTRY_API_KEY").then(|| "sk-env".to_string());
        assert_eq!(
            bearer_secret(&auth_for(REGISTRY, &[], None, &lookup)),
            Some("sk-env".to_string())
        );
        assert_eq!(
            bearer_secret(&auth_for(REGISTRY, &[], Some("sk-param"), &lookup)),
            Some("sk-param".to_string())
        );
        assert_eq!(
            bearer_secret(&auth_for(
                OPENAI_LIKE_MESSAGES_CONFIG,
                &[],
                None,
                &|_| Some("sk-env".to_string())
            )),
            None
        );
    }

    #[test]
    fn default_headers_are_the_anthropic_defaults() {
        for config in [OPENAI_LIKE_MESSAGES_CONFIG, REGISTRY] {
            assert_eq!(config.default_headers(), DEFAULT_ANTHROPIC_HEADERS);
        }
    }

    #[test]
    fn betas_are_merged_but_never_filtered() {
        let request = request_from(json!({
            "model": "m",
            "max_tokens": 16,
            "context_management": {"edits": [{"type": "compact_20260112"}]},
            "messages": [{"role": "user", "content": "hi"}]
        }));
        for config in [OPENAI_LIKE_MESSAGES_CONFIG, REGISTRY] {
            assert_eq!(
                config.request_headers(
                    headers(&[("Anthropic-Beta", "server-only-beta-2099-01-01")]),
                    &request
                ),
                headers(&[(
                    "anthropic-beta",
                    "compact-2026-01-12,server-only-beta-2099-01-01"
                )])
            );
        }
    }

    #[test]
    fn responses_pass_through() {
        let response: MessagesResponse = serde_json::from_value(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": "m",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 2}
        }))
        .expect("valid response");
        for config in [OPENAI_LIKE_MESSAGES_CONFIG, REGISTRY] {
            assert_eq!(
                config.transform_anthropic_messages_response("m", response.clone()),
                Ok(response.clone())
            );
            assert!(config.stream_decoder().is_none());
        }
    }

    fn requested_env(config: OpenAILikeMessagesConfig) -> Vec<String> {
        let requested = RefCell::new(Vec::<String>::new());
        let record = |name: &str| -> Option<String> {
            requested.borrow_mut().push(name.to_string());
            None
        };
        let _ = config.validate_environment(Vec::new(), None, "m", &record);
        let _ = config.get_complete_url(None, "m", &record);
        requested.into_inner()
    }

    #[rstest]
    #[case::deployment(OPENAI_LIKE_MESSAGES_CONFIG)]
    #[case::registry(REGISTRY)]
    #[case::registry_without_base_env(OpenAILikeMessagesConfig::Registry(
        &REGISTRY_PROVIDER_WITHOUT_BASE_ENV
    ))]
    fn secret_names_are_exactly_the_env_lookups(#[case] config: OpenAILikeMessagesConfig) {
        let requested = requested_env(config);
        assert_eq!(
            requested,
            config
                .secret_names()
                .iter()
                .map(|name| name.to_string())
                .collect::<Vec<_>>()
        );
    }
    #[rstest]
    #[case::deployment(OPENAI_LIKE_MESSAGES_CONFIG)]
    #[case::registry(REGISTRY)]
    fn test_reasoning_effort_budget_capped_for_openai_like_messages_upstream(
        #[case] config: OpenAILikeMessagesConfig,
    ) {
        use litellm_llms_types::formats::chat_completions::ReasoningEffort;
        use litellm_llms_types::formats::messages::{
            Message, MessageContent, MessageRole, ThinkingConfig,
        };
        let request = MessagesRequest {
            model: "unregistered-model".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text("hello".into()),
                extra: Map::new(),
            }],
            params: MessagesOptionalParams {
                max_tokens: Some(4000),
                reasoning_effort: Some(Recognized::Known(ReasoningEffort::Xhigh)),
                ..Default::default()
            },
        };
        let result = config
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        assert_eq!(
            result.params.thinking,
            Some(Recognized::Known(ThinkingConfig::enabled(3999)))
        );
        assert!(result.params.reasoning_effort.is_none());
        assert_eq!(result.params.max_tokens, Some(4000));
    }
    fn native_request(params: MessagesOptionalParams) -> MessagesRequest {
        use litellm_llms_types::formats::messages::{Message, MessageContent, MessageRole};
        MessagesRequest {
            model: "some-model".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text("hello".into()),
                extra: Map::new(),
            }],
            params,
        }
    }

    #[rstest]
    #[case::context_management(
        Some(litellm_llms_types::formats::messages::ContextManagement {
            edits: Some(vec![Recognized::Known(litellm_llms_types::formats::messages::ContextEdit::ClearToolUses { extra: Map::new() })]),
            ..Default::default()
        }), None, &[], "context-management-2025-06-27"
    )]
    #[case::fast_mode(None, Some(litellm_llms_types::formats::messages::Speed::Fast), &[], "fast-mode-2026-02-01")]
    #[case::caller_and_fast(None, Some(litellm_llms_types::formats::messages::Speed::Fast), &[("Anthropic-Beta", "caller-flag")], "caller-flag,fast-mode-2026-02-01")]
    fn compatible_feature_betas_are_merged_without_filtering(
        #[values(OPENAI_LIKE_MESSAGES_CONFIG, REGISTRY)] config: OpenAILikeMessagesConfig,
        #[case] context: Option<litellm_llms_types::formats::messages::ContextManagement>,
        #[case] speed: Option<litellm_llms_types::formats::messages::Speed>,
        #[case] caller: &[(&str, &str)],
        #[case] expected: &str,
    ) {
        let request = native_request(MessagesOptionalParams {
            max_tokens: Some(16),
            context_management: context.map(Recognized::Known),
            speed: speed.map(Recognized::Known),
            ..Default::default()
        });
        assert_eq!(
            config.request_headers(headers(caller), &request),
            headers(&[("anthropic-beta", expected)])
        );
    }

    #[rstest]
    #[case::adaptive(litellm_llms_types::formats::messages::ThinkingConfig::adaptive(None))]
    #[case::enabled(litellm_llms_types::formats::messages::ThinkingConfig::Enabled(
        Default::default()
    ))]
    #[case::disabled(litellm_llms_types::formats::messages::ThinkingConfig::Disabled(
        Default::default()
    ))]
    fn native_thinking_and_effort_survive_without_catalog_registration(
        #[case] thinking: litellm_llms_types::formats::messages::ThinkingConfig,
    ) {
        use litellm_llms_types::formats::messages::{EffortLevel, OutputConfig};
        let request = native_request(MessagesOptionalParams {
            max_tokens: Some(4096),
            thinking: Some(Recognized::Known(thinking)),
            output_config: Some(Recognized::Known(OutputConfig {
                effort: Some(Recognized::Known(EffortLevel::High)),
                ..Default::default()
            })),
            ..Default::default()
        });
        let expected = request.clone();
        let result = OPENAI_LIKE_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        assert_eq!(result, expected);
    }

    #[rstest]
    fn request_retains_the_anthropic_shape() {
        use litellm_llms_types::formats::messages::{
            Message, MessageContent, MessageRole, SystemPrompt, ThinkingConfig,
        };
        use litellm_llms_types::json_schema::{JsonSchema, JsonSchemaObject, JsonSchemaType};
        let request = MessagesRequest {
            model: "some-model".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Blocks(vec![ContentBlock {
                    cache_control: Some(Nullable::Value(CacheControl {
                        cache_type: Some(Nullable::Value("ephemeral".into())),
                        ..Default::default()
                    })),
                    ..ContentBlock::text("Summarize this")
                }]),
                extra: Map::new(),
            }],
            params: MessagesOptionalParams {
                max_tokens: Some(256),
                system: Some(SystemPrompt::Text("You are a careful assistant".into())),
                thinking: Some(Recognized::Known(ThinkingConfig::enabled(1024))),
                temperature: Some(0.3),
                stream: Some(false),
                tools: Some(vec![Recognized::Known(MessagesTool::Custom(
                    litellm_llms_types::formats::messages::CustomTool::try_from(ToolDefinition {
                        name: Some(Recognized::Known("lookup".into())),
                        input_schema: Some(Recognized::Known(JsonSchema::Object(Box::new(
                            JsonSchemaObject {
                                schema_type: Some(Recognized::Known(JsonSchemaType::Name(
                                    "object".into(),
                                ))),
                                ..Default::default()
                            },
                        )))),
                        ..Default::default()
                    })
                    .unwrap(),
                ))]),
                ..Default::default()
            },
        };
        let expected = request.clone();
        let result = OPENAI_LIKE_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        assert_eq!(result, expected);
        let body = OPENAI_LIKE_MESSAGES_CONFIG.request_body(&result).unwrap();
        assert_eq!(
            body,
            json!({
                "model": "some-model", "max_tokens": 256,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Summarize this", "cache_control": {"type": "ephemeral"}}]}],
                "system": "You are a careful assistant", "thinking": {"type": "enabled", "budget_tokens": 1024},
                "temperature": 0.3, "tools": [{"name": "lookup", "input_schema": {"type": "object"}}], "stream": false
            })
        );
    }

    #[rstest]
    #[case::lowercase(&[("authorization", "Bearer caller-token"), ("anthropic-version", "2024-10-22"), ("content-type", "application/json")])]
    #[case::standard_case(&[("Authorization", "Bearer caller-token"), ("Anthropic-Version", "2024-10-22"), ("Content-Type", "application/json")])]
    fn caller_headers_keep_their_values_and_casing(#[case] caller: &[(&str, &str)]) {
        let validated = auth_for(
            OPENAI_LIKE_MESSAGES_CONFIG,
            caller,
            Some("sk-test"),
            &|_| None,
        );
        assert!(matches!(validated.auth, AuthScheme::Forwarded));
        assert_eq!(
            litellm_http::request::with_default_headers(
                validated.headers,
                OPENAI_LIKE_MESSAGES_CONFIG.default_headers()
            ),
            headers(caller)
        );
    }
    #[rstest]
    fn compatible_cache_projection_preserves_the_callers_input() {
        use litellm_llms_types::formats::messages::{Message, MessageContent, MessageRole};
        let block = ContentBlock {
            cache_control: Some(Nullable::Value(CacheControl {
                cache_type: Some(Nullable::Value("ephemeral".into())),
                ttl: Some(Nullable::Value("1h".into())),
                ..Default::default()
            })),
            ..ContentBlock::text("hi")
        };
        let original = MessagesRequest {
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Blocks(vec![block]),
                extra: Map::new(),
            }],
            ..native_request(MessagesOptionalParams {
                max_tokens: Some(16),
                system: Some(litellm_llms_types::formats::messages::SystemPrompt::Blocks(
                    vec![ContentBlock {
                        cache_control: Some(Nullable::Value(CacheControl {
                            cache_type: Some(Nullable::Value("ephemeral".into())),
                            ttl: Some(Nullable::Value("5m".into())),
                            ..Default::default()
                        })),
                        ..ContentBlock::text("System")
                    }],
                )),
                ..Default::default()
            })
        };
        let snapshot = original.clone();
        let result = OPENAI_LIKE_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                original.clone(),
                &MessagesTransformContext::default(),
            )
            .unwrap();
        assert_eq!(original, snapshot);
        let native = crate::anthropic::messages::transformation::ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                original.clone(),
                &MessagesTransformContext::default(),
            )
            .unwrap();
        assert_eq!(native, original);
        assert_eq!(
            serde_json::to_value(&native).unwrap()["system"][0]["cache_control"],
            json!({"type": "ephemeral", "ttl": "5m"})
        );
        assert_eq!(
            serde_json::to_value(&original).unwrap()["messages"][0]["content"][0]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
        assert_eq!(
            serde_json::to_value(result).unwrap()["messages"][0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
    }

    #[rstest]
    fn compatible_system_output_config_injects_per_turn_beta() {
        use litellm_llms_types::formats::messages::{Message, MessageContent, MessageRole};
        let request = MessagesRequest {
            messages: vec![Message {
                role: MessageRole::System,
                content: MessageContent::Text("# Environment".into()),
                extra: [("output_config".into(), json!({"effort": "low"}))]
                    .into_iter()
                    .collect(),
            }],
            ..native_request(MessagesOptionalParams {
                max_tokens: Some(16),
                ..Default::default()
            })
        };
        assert_eq!(
            REGISTRY.request_headers(Vec::new(), &request),
            headers(&[("anthropic-beta", "per-turn-control-2026-07-01")])
        );
    }
    #[rstest]
    fn advisor_history_is_removed_on_the_compatible_path() {
        use litellm_llms_types::formats::messages::{Message, MessageContent, MessageRole};
        let text = ContentBlock::text("thinking out loud");
        let request = MessagesRequest {
            messages: vec![Message {
                role: MessageRole::Assistant,
                content: MessageContent::Blocks(vec![
                    text.clone(),
                    ContentBlock {
                        block_type: Some(Nullable::Value(ContentBlockType::ServerToolUse)),
                        payload: ContentBlockPayload {
                            id: Some(Nullable::Value("advisor_1".into())),
                            name: Some(Nullable::Value("advisor".into())),
                            input: Some(Recognized::Known(Map::new())),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    ContentBlock {
                        block_type: Some(Nullable::Value(ContentBlockType::AdvisorToolResult)),
                        tool_use_id: Some(Nullable::Value("advisor_1".into())),
                        payload: ContentBlockPayload {
                            content: Some(Recognized::Known(BlockContent::Text("stale".into()))),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                ]),
                extra: Map::new(),
            }],
            ..native_request(MessagesOptionalParams {
                max_tokens: Some(64),
                ..Default::default()
            })
        };
        let result = OPENAI_LIKE_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        assert_eq!(
            result.messages[0].content,
            MessageContent::Blocks(vec![text])
        );
    }
}
