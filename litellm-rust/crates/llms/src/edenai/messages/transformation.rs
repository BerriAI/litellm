use litellm_auth::{
    CredentialPlacement, CredentialPlanKind, CredentialRule, ExistingHeaderBehavior,
    ProviderAuthPolicy,
};
use litellm_llms_types::formats::messages::{MessagesRequest, MessagesResponse};

use crate::{
    Error,
    anthropic::{
        beta_headers::BetaPolicy,
        common_utils::DEFAULT_ANTHROPIC_HEADERS,
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::{
                COMPATIBLE_HOST_REQUEST_POLICY, transform_messages_request_with,
                update_headers_with_anthropic_beta,
            },
        },
    },
    base_llm::messages::{
        context::MessagesTransformContext,
        transformation::{
            BaseMessagesConfig, Headers, ValidatedEnvironment, complete_messages_url,
        },
    },
    edenai::common_utils::{
        EDENAI_API_BASE_ENV, EDENAI_API_KEY_ENV, EDENAI_COST_FIELD, missing_edenai_api_key,
        reported_cost, resolve_edenai_api_base, resolve_edenai_api_key,
    },
};

const EDENAI_AUTH_POLICY: ProviderAuthPolicy = ProviderAuthPolicy {
    rules: &[CredentialRule {
        kind: CredentialPlanKind::Static,
        placement: CredentialPlacement::Bearer,
    }],
    accepted_existing_headers: &["authorization", "x-api-key"],
    existing_header_behavior: ExistingHeaderBehavior::Preserve,
    scope: None,
    audience: None,
};

const EDENAI_BETA_POLICY: BetaPolicy = BetaPolicy::Forward;

pub struct EdenAIMessagesConfig;

pub const EDENAI_MESSAGES_CONFIG: EdenAIMessagesConfig = EdenAIMessagesConfig;

impl EdenAIMessagesConfig {
    /// The spend Eden reports beside the Anthropic body, which replaces the price map estimate.
    pub fn reported_response_cost(&self, response: &MessagesResponse) -> Option<f64> {
        response
            .extra
            .get(EDENAI_COST_FIELD)
            .and_then(reported_cost)
    }
}

impl BaseMessagesConfig for EdenAIMessagesConfig {
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
        Ok(complete_messages_url(&resolve_edenai_api_base(
            api_base, env_lookup,
        )))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        transform_messages_request_with(request, context, COMPATIBLE_HOST_REQUEST_POLICY)
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[EDENAI_API_KEY_ENV, EDENAI_API_BASE_ENV]
    }

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        Ok(ValidatedEnvironment::with_api_key(
            &EDENAI_AUTH_POLICY,
            headers,
            resolve_edenai_api_key(api_key, env_lookup),
            missing_edenai_api_key(),
        )?)
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_ANTHROPIC_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        EDENAI_BETA_POLICY.apply(update_headers_with_anthropic_beta(headers, request))
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
        base_llm::auth::AuthScheme,
        edenai::common_utils::{EDENAI_API_BASE, EDENAI_API_KEY_ENV},
    };

    fn request_from(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid request")
    }

    fn transformed(body: Value) -> Value {
        serde_json::to_value(
            EDENAI_MESSAGES_CONFIG
                .transform_anthropic_messages_request(
                    request_from(body),
                    &MessagesTransformContext::default(),
                )
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

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn response_from(extra: Value) -> MessagesResponse {
        let Value::Object(extra) = extra else {
            panic!("extra fields are an object")
        };
        serde_json::from_value(Value::Object(
            json!({
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "openai/gpt-4.1-nano",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 12, "output_tokens": 1}
            })
            .as_object()
            .expect("object")
            .clone()
            .into_iter()
            .chain(extra)
            .collect(),
        ))
        .expect("valid response")
    }

    #[rstest]
    #[case::default_base_keeps_its_version(None, &format!("{EDENAI_API_BASE}/v1/messages"))]
    #[case::trailing_slash(
        Some("https://eden.internal/v3/"),
        "https://eden.internal/v3/v1/messages"
    )]
    #[case::trailing_v1_is_dropped(
        Some("https://eden.internal/v1"),
        "https://eden.internal/v1/messages"
    )]
    #[case::complete_endpoint(
        Some("https://eden.internal/v3/v1/messages"),
        "https://eden.internal/v3/v1/messages"
    )]
    fn url_appends_the_messages_path_to_the_eden_base(
        #[case] api_base: Option<&str>,
        #[case] expected: &str,
    ) {
        assert_eq!(
            EDENAI_MESSAGES_CONFIG
                .get_complete_url(api_base, "openai/gpt-4.1-nano", &no_env)
                .expect("url builds"),
            expected
        );
    }

    #[test]
    fn url_uses_the_env_base_when_none_is_given() {
        let env = |name: &str| {
            (name == EDENAI_API_BASE_ENV).then(|| "https://api.eu.example/v3".to_string())
        };
        assert_eq!(
            EDENAI_MESSAGES_CONFIG
                .get_complete_url(None, "openai/gpt-4.1-nano", &env)
                .expect("url builds"),
            "https://api.eu.example/v3/v1/messages"
        );
    }

    fn validated(
        forwarded: &[(&str, &str)],
        api_key: Option<&str>,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        EDENAI_MESSAGES_CONFIG.validate_environment(
            headers(forwarded),
            api_key,
            "openai/gpt-4.1-nano",
            env,
        )
    }

    #[rstest]
    #[case::explicit(Some("sk-explicit"), "sk-explicit")]
    #[case::env(None, "sk-env")]
    fn the_resolved_key_is_sent_as_a_bearer(#[case] api_key: Option<&str>, #[case] expected: &str) {
        let env = |name: &str| (name == EDENAI_API_KEY_ENV).then(|| "sk-env".to_string());
        assert!(matches!(
            validated(&[], api_key, &env).expect("key resolves").auth,
            AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                ref secret,
            } if secret.expose() == expected
        ));
    }

    #[rstest]
    #[case::authorization(&[("Authorization", "Bearer caller")])]
    #[case::x_api_key(&[("X-Api-Key", "caller")])]
    fn a_caller_credential_header_wins_over_the_resolved_key(#[case] forwarded: &[(&str, &str)]) {
        let environment =
            validated(forwarded, Some("sk-explicit"), &no_env).expect("forwarded credential");
        assert!(matches!(environment.auth, AuthScheme::Forwarded));
        assert_eq!(environment.headers, headers(forwarded));
    }

    #[rstest]
    #[case::absent(None)]
    #[case::blank(Some("  "))]
    fn a_missing_key_is_an_authentication_error(#[case] api_key: Option<&str>) {
        assert!(matches!(
            validated(&[], api_key, &no_env),
            Err(Error::Auth(litellm_auth::Error::MissingApiKey {
                environment_variable: EDENAI_API_KEY_ENV,
                ..
            }))
        ));
    }

    #[test]
    fn default_headers_carry_the_anthropic_version() {
        assert!(
            EDENAI_MESSAGES_CONFIG
                .default_headers()
                .iter()
                .any(|(name, _)| *name == "anthropic-version")
        );
    }

    #[rstest]
    #[case::bare("gpt-4.1-nano")]
    #[case::seller_prefixed("anthropic/claude-sonnet-latest")]
    #[case::eden_alias("anthropic/claude-sonnet-latest@edenai")]
    fn the_model_is_sent_as_given(#[case] model: &str) {
        let body = transformed(json!({
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(body["model"], json!(model));
    }

    #[rstest]
    #[case::five_minutes("5m")]
    #[case::one_hour("1h")]
    fn cache_control_keeps_its_ttl(#[case] ttl: &str) {
        let cache_control = json!({"type": "ephemeral", "ttl": ttl});
        let body = transformed(json!({
            "model": "openai/gpt-4.1-nano",
            "max_tokens": 16,
            "system": [{"type": "text", "text": "sys", "cache_control": cache_control}],
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hi", "cache_control": cache_control}]
            }]
        }));
        assert_eq!(body["system"][0]["cache_control"], cache_control);
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"],
            cache_control
        );
    }

    #[rstest]
    #[case::enabled_with_budget(json!({"type": "enabled", "budget_tokens": 1024}))]
    #[case::disabled(json!({"type": "disabled"}))]
    fn thinking_is_forwarded_without_the_anthropic_rewrites(#[case] thinking: Value) {
        let body = transformed(json!({
            "model": "openai/gpt-4.1-nano",
            "max_tokens": 2048,
            "temperature": 0.5,
            "thinking": thinking,
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(body["thinking"], thinking);
        assert_eq!(body["temperature"], json!(0.5));
    }

    #[test]
    fn billing_metadata_blocks_are_stripped_from_the_system_prompt() {
        let kept =
            json!({"type": "text", "text": "Be terse", "cache_control": {"type": "ephemeral"}});
        let body = transformed(json!({
            "model": "openai/gpt-4.1-nano",
            "max_tokens": 16,
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.0; cc_entrypoint=cli"},
                kept
            ],
            "messages": [{"role": "user", "content": "hi"}]
        }));
        assert_eq!(body["system"], json!([kept]));
    }

    #[test]
    fn a_missing_max_tokens_is_rejected() {
        assert!(matches!(
            EDENAI_MESSAGES_CONFIG.transform_anthropic_messages_request(
                request_from(json!({
                    "model": "openai/gpt-4.1-nano",
                    "messages": [{"role": "user", "content": "hi"}]
                })),
                &MessagesTransformContext::default(),
            ),
            Err(Error::MissingField("max_tokens"))
        ));
    }

    #[rstest]
    #[case::unknown_forwarded_beta_survives(
        json!({}),
        &[("anthropic-beta", "made-up-beta-2099-01-01")],
        &[("x-trace", "1"), ("anthropic-beta", "made-up-beta-2099-01-01")]
    )]
    #[case::feature_beta_merged_with_forwarded_casings(
        json!({"output_config": {"format": {"type": "json_schema"}}}),
        &[("Anthropic-Beta", "made-up-beta-2099-01-01"), ("anthropic-beta", "web-search-2025-03-05")],
        &[
            ("x-trace", "1"),
            ("anthropic-beta", "made-up-beta-2099-01-01,structured-outputs-2025-11-13,web-search-2025-03-05")
        ]
    )]
    #[case::no_beta(json!({}), &[], &[("x-trace", "1")])]
    fn betas_are_merged_into_one_header_without_filtering(
        #[case] fields: Value,
        #[case] forwarded: &[(&str, &str)],
        #[case] expected: &[(&str, &str)],
    ) {
        let Value::Object(fields) = fields else {
            panic!("case fields are an object")
        };
        let request = request_from(Value::Object(
            json!({
                "model": "openai/gpt-4.1-nano",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}]
            })
            .as_object()
            .expect("object")
            .clone()
            .into_iter()
            .chain(fields)
            .collect(),
        ));
        assert_eq!(
            EDENAI_MESSAGES_CONFIG.request_headers(
                headers(&[("x-trace", "1")])
                    .into_iter()
                    .chain(headers(forwarded))
                    .collect(),
                &request
            ),
            headers(expected)
        );
    }

    #[test]
    fn the_response_passes_through_with_the_reported_cost() {
        let response = EDENAI_MESSAGES_CONFIG
            .transform_anthropic_messages_response(
                "openai/gpt-4.1-nano",
                response_from(json!({"cost": 0.0042})),
            )
            .expect("response transforms");
        assert_eq!(
            EDENAI_MESSAGES_CONFIG.reported_response_cost(&response),
            Some(0.0042)
        );
        let body = serde_json::to_value(response).expect("serializable");
        assert_eq!(body["cost"], json!(0.0042));
        assert_eq!(body["content"], json!([{"type": "text", "text": "OK"}]));
    }

    #[rstest]
    #[case::absent(json!({}))]
    #[case::null(json!({"cost": null}))]
    #[case::malformed(json!({"cost": "free"}))]
    fn a_missing_or_malformed_cost_leaves_pricing_to_the_price_map(#[case] extra: Value) {
        assert_eq!(
            EDENAI_MESSAGES_CONFIG.reported_response_cost(&response_from(extra)),
            None
        );
    }

    #[test]
    fn secret_names_cover_every_credential_and_base_lookup() {
        let requested = RefCell::new(Vec::<String>::new());
        let record = |name: &str| -> Option<String> {
            requested.borrow_mut().push(name.to_string());
            None
        };
        let _ = EDENAI_MESSAGES_CONFIG.validate_environment(Vec::new(), None, "m", &record);
        let _ = EDENAI_MESSAGES_CONFIG.get_complete_url(None, "m", &record);
        let requested = requested.into_inner();
        assert!(!requested.is_empty());
        let undeclared: Vec<&String> = requested
            .iter()
            .filter(|name| {
                !EDENAI_MESSAGES_CONFIG
                    .secret_names()
                    .contains(&name.as_str())
            })
            .collect();
        assert_eq!(undeclared, Vec::<&String>::new());
    }
    #[rstest]
    fn explicit_endpoint_wins_over_the_environment() {
        let lookup =
            |name: &str| (name == EDENAI_API_BASE_ENV).then(|| "https://api.eu.example/v3".into());
        assert_eq!(
            EDENAI_MESSAGES_CONFIG
                .get_complete_url(Some("https://eden.internal/v3/"), "native-model", &lookup)
                .unwrap(),
            "https://eden.internal/v3/v1/messages"
        );
    }

    #[rstest]
    fn native_payload_is_forwarded_untranslated() {
        use litellm_llms_types::{
            formats::messages::{
                CacheControl, ContentBlock, Message, MessageContent, MessageRole,
                MessagesOptionalParams, SystemPrompt, ThinkingConfig,
            },
            recognized::Recognized,
            serde_compat::Nullable,
        };
        let request = MessagesRequest {
            model: "openai/gpt-4.1-nano".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text("Say OK".into()),
                extra: Default::default(),
            }],
            params: MessagesOptionalParams {
                max_tokens: Some(16),
                system: Some(SystemPrompt::Blocks(vec![ContentBlock {
                    cache_control: Some(Nullable::Value(CacheControl {
                        cache_type: Some(Nullable::Value("ephemeral".into())),
                        ..Default::default()
                    })),
                    ..ContentBlock::text("Be terse")
                }])),
                thinking: Some(Recognized::Known(ThinkingConfig::enabled(1024))),
                ..Default::default()
            },
        };
        let expected = request.clone();
        let result = EDENAI_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        assert_eq!(result, expected);
        assert_eq!(
            EDENAI_MESSAGES_CONFIG.request_body(&result).unwrap(),
            json!({
                "model": "openai/gpt-4.1-nano", "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 16,
                "system": [{"type": "text", "text": "Be terse", "cache_control": {"type": "ephemeral"}}],
                "thinking": {"type": "enabled", "budget_tokens": 1024}
            })
        );
    }

    #[rstest]
    #[case::environment(None, &[])]
    #[case::explicit(Some("explicit-key"), &[])]
    #[case::caller(None, &[("Authorization", "Bearer caller-token")])]
    #[tokio::test]
    async fn authenticated_headers_keep_bearer_and_anthropic_defaults(
        #[case] api_key: Option<&str>,
        #[case] caller: &[(&str, &str)],
    ) {
        use crate::base_llm::auth::resolve_auth;
        use litellm_auth::AuthServices;
        let lookup = |name: &str| (name == EDENAI_API_KEY_ENV).then(|| "environment-key".into());
        let environment = validated(caller, api_key, &lookup).unwrap();
        let authenticated = resolve_auth(&AuthServices::default(), environment, &lookup)
            .await
            .unwrap();
        let actual = litellm_http::request::with_default_headers(
            authenticated.headers,
            EDENAI_MESSAGES_CONFIG.default_headers(),
        );
        let expected_auth = if caller.is_empty() {
            (
                "authorization",
                format!("Bearer {}", api_key.unwrap_or("environment-key")),
            )
        } else {
            ("Authorization", "Bearer caller-token".into())
        };
        assert_eq!(
            actual,
            headers(&[
                (expected_auth.0, expected_auth.1.as_str()),
                ("anthropic-version", "2023-06-01"),
                ("content-type", "application/json")
            ])
        );
    }
    #[rstest]
    #[tokio::test]
    async fn native_stream_retains_start_text_and_stop_events() {
        use crate::base_llm::messages::streaming::{
            anthropic_sse_event_stream, encode_anthropic_sse,
        };
        use futures_util::{StreamExt, TryStreamExt, stream};
        use litellm_llms_types::{
            formats::messages::{
                Message, MessageContent, MessageRole, MessageType, MessagesOptionalParams,
                MessagesUsage,
                streaming::{
                    MessagesContentBlockDelta, MessagesStreamEvent, MessagesStreamMessage,
                },
            },
            serde_compat::Nullable,
        };
        let request = MessagesRequest {
            model: "openai/gpt-4.1-nano".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text("Say OK".into()),
                extra: Default::default(),
            }],
            params: MessagesOptionalParams {
                max_tokens: Some(16),
                stream: Some(true),
                ..Default::default()
            },
        };
        let result = EDENAI_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        assert_eq!(
            EDENAI_MESSAGES_CONFIG.request_body(&result).unwrap()["stream"],
            true
        );
        assert!(EDENAI_MESSAGES_CONFIG.stream_decoder().is_none());
        let events = vec![
            MessagesStreamEvent::MessageStart {
                message: Box::new(MessagesStreamMessage {
                    id: "msg_eden_1".into(),
                    message_type: MessageType::Message,
                    role: MessageRole::Assistant,
                    model: "openai/gpt-4.1-nano".into(),
                    content: Vec::new(),
                    stop_reason: None,
                    stop_sequence: None,
                    usage: MessagesUsage {
                        input_tokens: Some(Nullable::Value(0)),
                        output_tokens: Some(Nullable::Value(0)),
                        ..Default::default()
                    },
                    safeguard_results: None,
                    extra: Default::default(),
                }),
                extra: Default::default(),
            },
            MessagesStreamEvent::ContentBlockDelta {
                index: 0,
                delta: MessagesContentBlockDelta::TextDelta {
                    text: "OK".into(),
                    extra: Default::default(),
                },
                extra: Default::default(),
            },
            MessagesStreamEvent::MessageStop {
                usage: None,
                extra: Default::default(),
            },
        ];
        let frames = events
            .iter()
            .map(encode_anthropic_sse)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let expected_payloads = [
            json!({"type": "message_start", "message": {"id": "msg_eden_1", "type": "message", "role": "assistant", "model": "openai/gpt-4.1-nano", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 0, "output_tokens": 0}}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "OK"}}),
            json!({"type": "message_stop"}),
        ];
        for (frame, expected) in frames.iter().zip(expected_payloads) {
            let wire = std::str::from_utf8(frame).unwrap();
            let event_name = expected["type"].as_str().unwrap();
            assert!(wire.starts_with(&format!("event: {event_name}\n")));
            let data = wire
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .unwrap();
            assert_eq!(serde_json::from_str::<Value>(data).unwrap(), expected);
        }
        let decoded = anthropic_sse_event_stream(stream::iter(frames.into_iter().map(Ok)).boxed())
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(decoded, events);
    }
}
