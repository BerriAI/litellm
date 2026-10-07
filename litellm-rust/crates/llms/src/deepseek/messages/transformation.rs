use litellm_auth::{
    CredentialPlacement, CredentialPlanKind, CredentialRule, ExistingHeaderBehavior,
    ProviderAuthPolicy, SecretValue,
};
use litellm_llms_types::{
    formats::messages::{
        BuiltinMessagesTool, CustomTool, MessagesOptionalParams, MessagesRequest, MessagesTool,
    },
    recognized::Recognized,
};

use crate::{
    Error,
    anthropic::{
        common_utils::DEFAULT_ANTHROPIC_HEADERS,
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::{COMPATIBLE_HOST_REQUEST_POLICY, transform_messages_request_with},
        },
    },
    base_llm::{
        auth::{Headers, ValidatedEnvironment},
        messages::{
            context::MessagesTransformContext,
            transformation::{BaseMessagesConfig, MESSAGES_PATH_SUFFIX},
        },
    },
    deepseek::common_utils::{
        DEEPSEEK_ANTHROPIC_API_BASE_ENV, DEEPSEEK_API_BASE_ENV, DEEPSEEK_API_KEY_ENV,
        resolve_deepseek_anthropic_api_base, resolve_deepseek_api_key,
    },
};

const ANTHROPIC_SURFACE: &str = "/anthropic";
const OPENAI_STYLE_SUFFIXES: [&str; 3] = [MESSAGES_PATH_SUFFIX, "/v1", "/beta"];

const DEEPSEEK_AUTH_POLICY: ProviderAuthPolicy = ProviderAuthPolicy {
    rules: &[CredentialRule {
        kind: CredentialPlanKind::Static,
        placement: CredentialPlacement::Header("x-api-key"),
    }],
    accepted_existing_headers: &["x-api-key", "authorization"],
    existing_header_behavior: ExistingHeaderBehavior::Preserve,
    scope: None,
    audience: None,
};

pub struct DeepSeekMessagesConfig;

pub const DEEPSEEK_MESSAGES_CONFIG: DeepSeekMessagesConfig = DeepSeekMessagesConfig;

impl BaseMessagesConfig for DeepSeekMessagesConfig {
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
        Ok(complete_deepseek_messages_url(
            &resolve_deepseek_anthropic_api_base(api_base, env_lookup),
        ))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        let request =
            transform_messages_request_with(request, context, COMPATIBLE_HOST_REQUEST_POLICY)?;
        Ok(MessagesRequest {
            params: MessagesOptionalParams {
                tools: request
                    .params
                    .tools
                    .map(|tools| tools.into_iter().map(without_custom_type).collect()),
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

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        Ok(ValidatedEnvironment::with_api_key(
            &DEEPSEEK_AUTH_POLICY,
            headers,
            resolve_deepseek_api_key(api_key, env_lookup).map(SecretValue::new),
            litellm_auth::Error::MissingApiKey {
                provider: "DeepSeek",
                environment_variable: DEEPSEEK_API_KEY_ENV,
            },
        )?)
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_ANTHROPIC_HEADERS
    }

    fn request_headers(&self, headers: Headers, _request: &MessagesRequest) -> Headers {
        litellm_http::request::without_headers(headers, &["anthropic-beta"])
    }
}

/// Python's `get_complete_url`: an OpenAI-style base such as `https://api.deepseek.com/v1`
/// is moved onto the `/anthropic` surface, so one `DEEPSEEK_API_BASE` serves both routes.
pub fn complete_deepseek_messages_url(api_base: &str) -> String {
    let api_base = api_base.trim_end_matches('/');
    let on_anthropic_surface = |base: &str| base.contains(&format!("{ANTHROPIC_SURFACE}/"));
    if api_base.ends_with(MESSAGES_PATH_SUFFIX) && on_anthropic_surface(api_base) {
        return api_base.to_string();
    }
    let api_base = OPENAI_STYLE_SUFFIXES.iter().fold(api_base, |base, suffix| {
        base.strip_suffix(suffix).unwrap_or(base)
    });
    if api_base.ends_with(ANTHROPIC_SURFACE) || on_anthropic_surface(api_base) {
        return format!("{api_base}{MESSAGES_PATH_SUFFIX}");
    }
    format!("{api_base}{ANTHROPIC_SURFACE}{MESSAGES_PATH_SUFFIX}")
}

fn without_custom_type(tool: Recognized<MessagesTool>) -> Recognized<MessagesTool> {
    match tool {
        Recognized::Known(MessagesTool::Builtin(BuiltinMessagesTool::Custom(definition))) => {
            Recognized::Known(MessagesTool::Custom(CustomTool { definition }))
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use litellm_auth::CredentialPlacement;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        base_llm::{auth::AuthScheme, messages::context::MessagesModelCapabilities},
        deepseek::common_utils::DEEPSEEK_ANTHROPIC_DEFAULT_API_BASE,
    };

    fn request_from(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid request")
    }

    fn transformed(body: Value, context: &MessagesTransformContext) -> Value {
        serde_json::to_value(
            DEEPSEEK_MESSAGES_CONFIG
                .transform_anthropic_messages_request(request_from(body), context)
                .expect("request transforms"),
        )
        .expect("serializable request")
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn reasoning_context() -> MessagesTransformContext {
        MessagesTransformContext::with_lookup(
            MessagesModelCapabilities {
                supports_reasoning: true,
                supports_legacy_thinking: true,
                thinking_always_on: true,
                supports_sampling_params: true,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        )
    }

    #[rstest]
    #[case::openai_style_v1(
        "https://api.deepseek.com/v1",
        "https://api.deepseek.com/anthropic/v1/messages"
    )]
    #[case::bare_host(
        "https://api.deepseek.com",
        "https://api.deepseek.com/anthropic/v1/messages"
    )]
    #[case::beta_surface(
        "https://api.deepseek.com/beta",
        "https://api.deepseek.com/anthropic/v1/messages"
    )]
    #[case::anthropic_surface(
        "https://api.deepseek.com/anthropic/",
        "https://api.deepseek.com/anthropic/v1/messages"
    )]
    #[case::anthropic_v1(
        "https://api.deepseek.com/anthropic/v1",
        "https://api.deepseek.com/anthropic/v1/messages"
    )]
    #[case::complete_endpoint(
        "https://api.deepseek.com/anthropic/v1/messages/",
        "https://api.deepseek.com/anthropic/v1/messages"
    )]
    #[case::messages_endpoint_off_the_anthropic_surface(
        "https://api.deepseek.com/v1/messages",
        "https://api.deepseek.com/anthropic/v1/messages"
    )]
    #[case::nested_under_anthropic(
        "https://proxy.example/anthropic/deepseek",
        "https://proxy.example/anthropic/deepseek/v1/messages"
    )]
    fn url_lands_on_the_anthropic_messages_surface(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(
            DEEPSEEK_MESSAGES_CONFIG
                .get_complete_url(Some(base), "deepseek-chat", &|_| None)
                .expect("url builds"),
            expected
        );
    }

    #[rstest]
    #[case::param_wins(Some("https://param.example"), &[(DEEPSEEK_ANTHROPIC_API_BASE_ENV, "https://anthropic-env.example"), (DEEPSEEK_API_BASE_ENV, "https://env.example/v1")], "https://param.example/anthropic/v1/messages")]
    #[case::anthropic_env_before_shared_env(Some(" "), &[(DEEPSEEK_ANTHROPIC_API_BASE_ENV, "https://anthropic-env.example"), (DEEPSEEK_API_BASE_ENV, "https://env.example/v1")], "https://anthropic-env.example/anthropic/v1/messages")]
    #[case::shared_env(None, &[(DEEPSEEK_API_BASE_ENV, "https://env.example/v1")], "https://env.example/anthropic/v1/messages")]
    fn api_base_resolves_param_then_anthropic_env_then_shared_env(
        #[case] api_base: Option<&str>,
        #[case] env: &[(&str, &str)],
        #[case] expected: &str,
    ) {
        let lookup = |name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        };
        assert_eq!(
            DEEPSEEK_MESSAGES_CONFIG
                .get_complete_url(api_base, "deepseek-chat", &lookup)
                .expect("url builds"),
            expected
        );
    }

    #[test]
    fn api_base_defaults_to_the_anthropic_surface() {
        assert_eq!(
            DEEPSEEK_MESSAGES_CONFIG
                .get_complete_url(None, "deepseek-chat", &|_| None)
                .expect("url builds"),
            format!("{DEEPSEEK_ANTHROPIC_DEFAULT_API_BASE}{MESSAGES_PATH_SUFFIX}")
        );
    }

    fn validated(
        forwarded: &[(&str, &str)],
        api_key: Option<&str>,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        DEEPSEEK_MESSAGES_CONFIG.validate_environment(
            headers(forwarded),
            api_key,
            "deepseek-chat",
            env,
        )
    }

    #[rstest]
    #[case::x_api_key(&[("X-Api-Key", "caller")])]
    #[case::authorization(&[("Authorization", "Bearer caller")])]
    fn a_caller_credential_header_wins_over_the_resolved_key(#[case] forwarded: &[(&str, &str)]) {
        let validated = validated(forwarded, Some("sk-deepseek"), &|_| None).unwrap();
        assert!(matches!(validated.auth, AuthScheme::Forwarded));
        assert_eq!(validated.headers, headers(forwarded));
    }

    #[rstest]
    #[case::param(Some("sk-param"), "sk-param")]
    #[case::env_when_param_blank(Some(" "), "sk-env")]
    #[case::env_when_param_absent(None, "sk-env")]
    fn the_resolved_key_goes_in_x_api_key(#[case] api_key: Option<&str>, #[case] expected: &str) {
        let env = |name: &str| (name == DEEPSEEK_API_KEY_ENV).then(|| "sk-env".to_string());
        assert!(matches!(
            validated(&[], api_key, &env).unwrap().auth,
            AuthScheme::Credential {
                placement: CredentialPlacement::Header("x-api-key"),
                ref secret
            } if secret.expose() == expected
        ));
    }

    #[test]
    fn a_missing_key_names_the_deepseek_env_var() {
        assert!(matches!(
            validated(&[], None, &|_| None),
            Err(Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "DeepSeek",
                environment_variable: DEEPSEEK_API_KEY_ENV,
            }))
        ));
    }

    #[test]
    fn default_headers_are_the_anthropic_defaults() {
        assert_eq!(
            DEEPSEEK_MESSAGES_CONFIG.default_headers(),
            DEFAULT_ANTHROPIC_HEADERS
        );
    }

    #[test]
    fn only_custom_typed_tools_lose_their_type() {
        let other_tools = json!([
            {"name": "untyped", "input_schema": {"type": "object"}},
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 2},
            {"type": "bash_20250124", "name": "bash"}
        ]);
        let tools = [json!({
            "type": "custom",
            "name": "get_weather",
            "description": "weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
        })]
        .into_iter()
        .chain(other_tools.as_array().unwrap().iter().cloned())
        .collect::<Vec<_>>();
        let body = json!({
            "model": "deepseek-chat",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools
        });
        let expected = [json!({
            "name": "get_weather",
            "description": "weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
        })]
        .into_iter()
        .chain(other_tools.as_array().unwrap().iter().cloned())
        .collect::<Vec<_>>();
        assert_eq!(
            transformed(body, &MessagesTransformContext::default())["tools"],
            json!(expected)
        );
    }

    #[test]
    fn thinking_blocks_in_assistant_history_are_forwarded() {
        let messages = json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "let me think", "signature": "sig"},
                {"type": "text", "text": "hello"}
            ]},
            {"role": "user", "content": "again"}
        ]);
        let body = json!({"model": "deepseek-chat", "max_tokens": 64, "messages": messages});
        assert_eq!(
            transformed(body, &MessagesTransformContext::default())["messages"],
            messages
        );
    }

    #[rstest]
    #[case::disabled_thinking_is_kept(json!({"type": "disabled"}))]
    #[case::temperature_survives_enabled_thinking(json!({"type": "enabled", "budget_tokens": 1024}))]
    fn claude_thinking_rules_do_not_apply(#[case] thinking: Value) {
        let body = json!({
            "model": "deepseek-reasoner",
            "max_tokens": 2048,
            "temperature": 0.5,
            "thinking": thinking,
            "messages": [{"role": "user", "content": "hi"}]
        });
        assert_eq!(transformed(body.clone(), &reasoning_context()), body);
    }

    #[rstest]
    #[case::only_billing(json!("x-anthropic-billing-header: cc_version=1"), None)]
    #[case::billing_among_blocks(
        json!([
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
            {"type": "text", "text": "be terse"}
        ]),
        Some(json!([{"type": "text", "text": "be terse"}]))
    )]
    fn billing_metadata_is_stripped_and_an_emptied_system_omitted(
        #[case] system: Value,
        #[case] expected: Option<Value>,
    ) {
        let body = json!({
            "model": "deepseek-chat",
            "max_tokens": 64,
            "system": system,
            "messages": [{"role": "user", "content": "hi"}]
        });
        assert_eq!(
            transformed(body, &MessagesTransformContext::default())
                .get("system")
                .cloned(),
            expected
        );
    }

    #[test]
    fn every_anthropic_beta_is_dropped_including_feature_betas() {
        let request = request_from(json!({
            "model": "deepseek-chat",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "context_management": {"edits": [{"type": "compact_20260112"}]},
            "output_config": {"format": {"type": "json_schema"}}
        }));
        assert_eq!(
            DEEPSEEK_MESSAGES_CONFIG.request_headers(
                headers(&[
                    ("x-api-key", "k"),
                    ("Anthropic-Beta", "web-search-2025-03-05"),
                    ("anthropic-beta", "compact-2026-01-12"),
                ]),
                &request
            ),
            headers(&[("x-api-key", "k")])
        );
    }

    #[test]
    fn secret_names_cover_every_credential_and_base_lookup() {
        let requested = RefCell::new(Vec::<String>::new());
        let record = |name: &str| -> Option<String> {
            requested.borrow_mut().push(name.to_string());
            None
        };
        let _ = DEEPSEEK_MESSAGES_CONFIG.validate_environment(
            Vec::new(),
            None,
            "deepseek-chat",
            &record,
        );
        let _ = DEEPSEEK_MESSAGES_CONFIG.get_complete_url(None, "deepseek-chat", &record);
        let requested = requested.into_inner();
        assert!(!requested.is_empty());
        let undeclared: Vec<&String> = requested
            .iter()
            .filter(|name| {
                !DEEPSEEK_MESSAGES_CONFIG
                    .secret_names()
                    .contains(&name.as_str())
            })
            .collect();
        assert_eq!(undeclared, Vec::<&String>::new());
    }
    #[rstest]
    #[case::adaptive(litellm_llms_types::formats::messages::ThinkingConfig::adaptive(None))]
    #[case::disabled(litellm_llms_types::formats::messages::ThinkingConfig::Disabled(
        Default::default()
    ))]
    #[case::enabled(litellm_llms_types::formats::messages::ThinkingConfig::enabled(2048))]
    fn test_native_thinking_effort_and_sampling_do_not_depend_on_model_registration(
        #[case] thinking: litellm_llms_types::formats::messages::ThinkingConfig,
    ) {
        use litellm_llms_types::formats::messages::{
            EffortLevel, Message, MessageContent, MessageRole, MessagesOptionalParams, OutputConfig,
        };
        use litellm_llms_types::recognized::Recognized;
        let params = MessagesOptionalParams {
            max_tokens: Some(4096),
            thinking: Some(Recognized::Known(thinking)),
            output_config: Some(Recognized::Known(OutputConfig {
                effort: Some(Recognized::Known(EffortLevel::High)),
                ..Default::default()
            })),
            temperature: Some(0.7),
            ..Default::default()
        };
        let request = MessagesRequest {
            model: "unregistered-native-model".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text("hello".into()),
                extra: Default::default(),
            }],
            params: params.clone(),
        };
        let result = DEEPSEEK_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        assert_eq!(result.params.thinking, params.thinking);
        assert_eq!(result.params.output_config, params.output_config);
        assert_eq!(result.params.temperature, params.temperature);
    }
    #[rstest]
    fn native_tools_and_thinking_survive_a_small_output_limit() {
        use litellm_llms_types::json_schema::{JsonSchema, JsonSchemaObject, JsonSchemaType};
        use litellm_llms_types::{
            formats::messages::{
                BuiltinMessagesTool, ContentBlock, ContentBlockPayload, ContentBlockType,
                CustomTool, Message, MessageContent, MessageRole, MessagesOptionalParams,
                ThinkingConfig, ToolDefinition,
            },
            recognized::Recognized,
            serde_compat::Nullable,
        };
        let custom = ToolDefinition {
            name: Some(Recognized::Known("get_weather".into())),
            description: Some(Recognized::Known("Get weather".into())),
            input_schema: Some(Recognized::Known(JsonSchema::Object(Box::new(
                JsonSchemaObject {
                    schema_type: Some(Recognized::Known(JsonSchemaType::Name("object".into()))),
                    ..Default::default()
                },
            )))),
            ..Default::default()
        };
        let web = ToolDefinition {
            name: Some(Recognized::Known("web_search".into())),
            max_uses: Some(Recognized::Known(1)),
            ..Default::default()
        };
        let messages = vec![
            Message {
                role: MessageRole::User,
                content: MessageContent::Text("Use the tool.".into()),
                extra: Default::default(),
            },
            Message {
                role: MessageRole::Assistant,
                content: MessageContent::Blocks(vec![
                    ContentBlock {
                        block_type: Some(Nullable::Value(ContentBlockType::Thinking)),
                        payload: ContentBlockPayload {
                            thinking: Some(Nullable::Value("I should call the tool.".into())),
                            signature: Some(Nullable::Value("sig".into())),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    ContentBlock {
                        block_type: Some(Nullable::Value(ContentBlockType::ToolUse)),
                        payload: ContentBlockPayload {
                            id: Some(Nullable::Value("toolu_123".into())),
                            name: Some(Nullable::Value("get_weather".into())),
                            input: Some(Recognized::Known(
                                [("city".into(), json!("Sao Paulo"))].into_iter().collect(),
                            )),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                ]),
                extra: Default::default(),
            },
            Message {
                role: MessageRole::User,
                content: MessageContent::Blocks(vec![ContentBlock {
                    block_type: Some(Nullable::Value(ContentBlockType::ToolResult)),
                    tool_use_id: Some(Nullable::Value("toolu_123".into())),
                    payload: ContentBlockPayload {
                        content: Some(Recognized::Known(
                            litellm_llms_types::formats::messages::BlockContent::Text(
                                "Sunny".into(),
                            ),
                        )),
                        ..Default::default()
                    },
                    ..Default::default()
                }]),
                extra: Default::default(),
            },
        ];
        let request = MessagesRequest {
            model: "deepseek-v4-pro".into(),
            messages: messages.clone(),
            params: MessagesOptionalParams {
                max_tokens: Some(100),
                thinking: Some(Recognized::Known(ThinkingConfig::enabled(1024))),
                tools: Some(vec![
                    Recognized::Known(MessagesTool::Builtin(BuiltinMessagesTool::Custom(
                        custom.clone(),
                    ))),
                    Recognized::Known(MessagesTool::Builtin(
                        BuiltinMessagesTool::WebSearch20260209(web.clone()),
                    )),
                ]),
                ..Default::default()
            },
        };
        let result = DEEPSEEK_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        assert_eq!(result.messages, messages);
        assert_eq!(
            result.params.thinking,
            Some(Recognized::Known(ThinkingConfig::enabled(1024)))
        );
        assert_eq!(result.params.max_tokens, Some(100));
        assert_eq!(
            result.params.tools,
            Some(vec![
                Recognized::Known(MessagesTool::Custom(CustomTool::try_from(custom).unwrap())),
                Recognized::Known(MessagesTool::Builtin(
                    BuiltinMessagesTool::WebSearch20260209(web)
                ))
            ])
        );
    }
}
