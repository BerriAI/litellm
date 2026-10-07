use litellm_auth::{
    CredentialPlacement, CredentialPlanKind, CredentialRule, ExistingHeaderBehavior,
    ProviderAuthPolicy, SecretValue,
};
use litellm_llms_types::formats::messages::MessagesRequest;

use crate::{
    Error,
    anthropic::{
        beta_headers::BetaPolicy,
        common_utils::DEFAULT_ANTHROPIC_HEADERS,
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::{
                COMPATIBLE_HOST_REQUEST_POLICY, Compaction, RequestPolicy,
                transform_messages_request_with, update_headers_with_anthropic_beta,
            },
        },
    },
    base_llm::{
        auth::{Headers, ValidatedEnvironment},
        messages::{
            context::MessagesTransformContext,
            transformation::{BaseMessagesConfig, complete_messages_url},
        },
    },
    minimax::common_utils::{
        MINIMAX_API_BASE_ENV, MINIMAX_API_KEY_ENV, resolve_minimax_anthropic_api_base,
        resolve_minimax_api_key,
    },
};

const MINIMAX_AUTH_POLICY: ProviderAuthPolicy = ProviderAuthPolicy {
    rules: &[CredentialRule {
        kind: CredentialPlanKind::Static,
        placement: CredentialPlacement::Header("x-api-key"),
    }],
    accepted_existing_headers: &["x-api-key", "authorization"],
    existing_header_behavior: ExistingHeaderBehavior::Preserve,
    scope: None,
    audience: None,
};

const BETA_POLICY: BetaPolicy = BetaPolicy::Drop;

const REQUEST_POLICY: RequestPolicy = RequestPolicy {
    compaction: Compaction::Drop,
    ..COMPATIBLE_HOST_REQUEST_POLICY
};

pub struct MinimaxMessagesConfig;

pub const MINIMAX_MESSAGES_CONFIG: MinimaxMessagesConfig = MinimaxMessagesConfig;

impl BaseMessagesConfig for MinimaxMessagesConfig {
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
        Ok(complete_messages_url(&resolve_minimax_anthropic_api_base(
            api_base, env_lookup,
        )))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        transform_messages_request_with(request, context, REQUEST_POLICY)
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[MINIMAX_API_KEY_ENV, MINIMAX_API_BASE_ENV]
    }

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        Ok(ValidatedEnvironment::with_api_key(
            &MINIMAX_AUTH_POLICY,
            headers,
            resolve_minimax_api_key(api_key, env_lookup).map(SecretValue::new),
            litellm_auth::Error::MissingApiKey {
                provider: "MiniMax",
                environment_variable: MINIMAX_API_KEY_ENV,
            },
        )?)
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_ANTHROPIC_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        BETA_POLICY.apply(update_headers_with_anthropic_beta(headers, request))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use litellm_llms_types::formats::messages::MessagesResponse;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        base_llm::{
            auth::AuthScheme,
            messages::context::{MessagesModelCapabilities, SupportedEffortTiers, ThinkingBudgets},
        },
        minimax::common_utils::MINIMAX_ANTHROPIC_API_BASE,
    };

    const ANTHROPIC_ENV: &[(&str, &str)] = &[
        ("ANTHROPIC_API_KEY", "sk-ant-api"),
        ("ANTHROPIC_AUTH_TOKEN", "sk-ant-token"),
        ("ANTHROPIC_API_BASE", "https://api.anthropic.com"),
        ("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
    ];

    fn env_from(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name: &str| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn request(fields: Value) -> MessagesRequest {
        let Value::Object(fields) = fields else {
            panic!("request fields are an object")
        };
        serde_json::from_value(Value::Object(
            [
                ("model".to_string(), json!("MiniMax-M2")),
                (
                    "messages".to_string(),
                    json!([{"role": "user", "content": "hi"}]),
                ),
            ]
            .into_iter()
            .chain(fields)
            .collect(),
        ))
        .expect("valid request")
    }

    const ADAPTIVE_MODEL: MessagesModelCapabilities = MessagesModelCapabilities {
        supports_reasoning: true,
        supports_adaptive_thinking: true,
        thinking_always_on: false,
        supports_legacy_thinking: true,
        supports_output_config: true,
        supports_sampling_params: true,
        supports_speed: false,
        supports_mid_conversation_system: false,
        supports_cache_control_ttl: false,
        supports_native_structured_output: false,
        supports_tool_search: false,
        effort_ceiling: None,
        effort_tiers: SupportedEffortTiers {
            minimal: false,
            low: true,
            medium: true,
            high: true,
            xhigh: false,
            max: false,
        },
    };

    fn transform_for(capabilities: MessagesModelCapabilities, fields: Value) -> Value {
        let context = MessagesTransformContext::with_lookup(capabilities, false, &|_: &str| None);
        serde_json::to_value(
            MINIMAX_MESSAGES_CONFIG
                .transform_anthropic_messages_request(request(fields), &context)
                .expect("request transforms"),
        )
        .expect("serializable request")
    }

    fn transform(fields: Value) -> Value {
        transform_for(ADAPTIVE_MODEL, fields)
    }

    fn default_messages_url() -> String {
        format!("{MINIMAX_ANTHROPIC_API_BASE}/v1/messages")
    }

    #[rstest]
    #[case::china_region_base("https://api.minimaxi.com/anthropic")]
    #[case::china_region_trailing_slash("https://api.minimaxi.com/anthropic/")]
    #[case::version_suffix("https://api.minimaxi.com/anthropic/v1")]
    #[case::version_suffix_with_slash("https://api.minimaxi.com/anthropic/v1/")]
    #[case::complete_endpoint("https://api.minimaxi.com/anthropic/v1/messages")]
    #[case::complete_endpoint_with_slash("https://api.minimaxi.com/anthropic/v1/messages/")]
    fn every_base_shape_resolves_to_one_messages_path(#[case] api_base: &str) {
        assert_eq!(
            MINIMAX_MESSAGES_CONFIG
                .get_complete_url(Some(api_base), "MiniMax-M2", &|_| None)
                .unwrap(),
            "https://api.minimaxi.com/anthropic/v1/messages"
        );
    }

    #[rstest]
    #[case::param_beats_env(
        Some("https://param.example/anthropic"),
        "https://param.example/anthropic/v1/messages"
    )]
    #[case::env_when_param_absent(None, "https://env.example/anthropic/v1/messages")]
    #[case::env_when_param_blank(Some("  "), "https://env.example/anthropic/v1/messages")]
    fn the_base_comes_from_the_param_then_minimax_api_base(
        #[case] api_base: Option<&str>,
        #[case] expected: &str,
    ) {
        let env = env_from(&[(MINIMAX_API_BASE_ENV, "https://env.example/anthropic/v1")]);
        assert_eq!(
            MINIMAX_MESSAGES_CONFIG
                .get_complete_url(api_base, "MiniMax-M2", &env)
                .unwrap(),
            expected
        );
    }

    #[test]
    fn the_default_base_is_minimax_never_an_anthropic_base() {
        assert_eq!(
            MINIMAX_MESSAGES_CONFIG
                .get_complete_url(None, "MiniMax-M2", &env_from(ANTHROPIC_ENV))
                .unwrap(),
            default_messages_url()
        );
    }

    fn validated(
        forwarded: &[(&str, &str)],
        api_key: Option<&str>,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        MINIMAX_MESSAGES_CONFIG.validate_environment(headers(forwarded), api_key, "MiniMax-M2", env)
    }

    fn x_api_key(validated: ValidatedEnvironment) -> String {
        match validated.auth {
            AuthScheme::Credential {
                placement: CredentialPlacement::Header("x-api-key"),
                secret,
            } => secret.expose().to_string(),
            other => panic!("expected an x-api-key credential, got {other:?}"),
        }
    }

    #[rstest]
    #[case::param(Some("sk-param"), "sk-param")]
    #[case::env_when_param_absent(None, "sk-env")]
    #[case::env_when_param_blank(Some(" "), "sk-env")]
    fn the_minimax_key_goes_in_x_api_key(#[case] api_key: Option<&str>, #[case] expected: &str) {
        let env = env_from(&[(MINIMAX_API_KEY_ENV, "sk-env")]);
        assert_eq!(x_api_key(validated(&[], api_key, &env).unwrap()), expected);
    }

    #[rstest]
    #[case::x_api_key(&[("X-Api-Key", "caller")])]
    #[case::authorization(&[("Authorization", "Bearer caller")])]
    fn a_caller_credential_is_forwarded(#[case] forwarded: &[(&str, &str)]) {
        let validated = validated(forwarded, Some("sk-minimax"), &|_| None).unwrap();
        assert!(matches!(validated.auth, AuthScheme::Forwarded));
        assert_eq!(validated.headers, headers(forwarded));
    }

    #[test]
    fn a_missing_key_names_minimax_and_never_falls_back_to_anthropic_credentials() {
        let error = validated(&[], None, &env_from(ANTHROPIC_ENV)).unwrap_err();
        assert_eq!(
            error,
            Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "MiniMax",
                environment_variable: MINIMAX_API_KEY_ENV,
            })
        );
    }

    #[test]
    fn default_headers_are_the_anthropic_wire_headers() {
        assert_eq!(
            MINIMAX_MESSAGES_CONFIG.default_headers(),
            DEFAULT_ANTHROPIC_HEADERS
        );
    }

    #[rstest]
    #[case::disabled_thinking(json!({"max_tokens": 4096, "thinking": {"type": "disabled"}}))]
    #[case::legacy_thinking(json!({"max_tokens": 4096, "thinking": {"type": "enabled", "budget_tokens": 1024}}))]
    #[case::adaptive_thinking_with_effort(json!({
        "max_tokens": 4096,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "max"}
    }))]
    #[case::thinking_with_temperature(json!({
        "max_tokens": 4096,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "temperature": 0.2
    }))]
    fn thinking_and_output_config_reach_minimax_unchanged(#[case] body: Value) {
        let transformed = transform(body.clone());
        for key in ["thinking", "output_config", "temperature"] {
            assert_eq!(transformed.get(key), body.get(key), "{key}");
        }
    }

    #[test]
    fn reasoning_effort_is_still_mapped_to_native_thinking() {
        let transformed = transform_for(
            MessagesModelCapabilities {
                supports_reasoning: true,
                supports_legacy_thinking: true,
                ..Default::default()
            },
            json!({"max_tokens": 4096, "reasoning_effort": "low"}),
        );
        assert_eq!(transformed.get("reasoning_effort"), None);
        assert_eq!(
            transformed["thinking"],
            json!({"type": "enabled", "budget_tokens": ThinkingBudgets::default().low})
        );
    }

    #[rstest]
    #[case::only_billing_string(json!("x-anthropic-billing-header: cc_version=1"), None)]
    #[case::only_billing_blocks(
        json!([{"type": "text", "text": "x-anthropic-billing-header: cc_version=1"}]),
        None
    )]
    #[case::billing_beside_real_system(
        json!([
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
            {"type": "text", "text": "be terse"}
        ]),
        Some(json!([{"type": "text", "text": "be terse"}]))
    )]
    fn billing_metadata_is_stripped_from_system(
        #[case] system: Value,
        #[case] expected: Option<Value>,
    ) {
        let transformed = transform(json!({"max_tokens": 16, "system": system}));
        assert_eq!(transformed.get("system").cloned(), expected);
    }

    #[test]
    fn compaction_is_dropped_from_the_body() {
        let fields = json!({"max_tokens": 64, "compaction": {"enabled": true}});
        assert_eq!(
            serde_json::to_value(request(fields.clone())).unwrap()["compaction"],
            fields["compaction"]
        );
        assert_eq!(transform(fields).get("compaction"), None);
    }

    #[test]
    fn max_tokens_is_required() {
        assert_eq!(
            MINIMAX_MESSAGES_CONFIG.transform_anthropic_messages_request(
                request(json!({})),
                &MessagesTransformContext::default()
            ),
            Err(Error::MissingField("max_tokens"))
        );
    }

    #[rstest]
    #[case::forwarded_beta(json!({"max_tokens": 16}), &[("anthropic-beta", "web-search-2025-03-05")])]
    #[case::forwarded_beta_any_casing(json!({"max_tokens": 16}), &[("Anthropic-Beta", "context-1m-2025-08-07")])]
    #[case::compaction_param(json!({"max_tokens": 16, "compaction": {"enabled": true}}), &[])]
    #[case::structured_output(
        json!({"max_tokens": 16, "output_config": {"format": {"type": "json_schema"}}}),
        &[]
    )]
    #[case::compact_context_edit(
        json!({"max_tokens": 16, "context_management": {"edits": [{"type": "compact_20260112"}]}}),
        &[]
    )]
    fn no_anthropic_beta_reaches_minimax(
        #[case] fields: Value,
        #[case] forwarded: &[(&str, &str)],
    ) {
        let request = request(fields);
        let sent = MINIMAX_MESSAGES_CONFIG.request_headers(
            headers(&[("x-api-key", "k")])
                .into_iter()
                .chain(headers(forwarded))
                .collect(),
            &request,
        );
        assert_eq!(sent, headers(&[("x-api-key", "k")]));
    }

    #[test]
    fn transform_response_passes_through() {
        let response: MessagesResponse = serde_json::from_value(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "hmm", "signature": "s"}, {"type": "text", "text": "hello"}],
            "model": "MiniMax-M2",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 2}
        }))
        .expect("valid response");
        assert_eq!(
            MINIMAX_MESSAGES_CONFIG
                .transform_anthropic_messages_response("MiniMax-M2", response.clone())
                .expect("response transforms"),
            response
        );
        assert!(MINIMAX_MESSAGES_CONFIG.stream_decoder().is_none());
    }

    #[test]
    fn secret_names_cover_every_lookup_and_no_anthropic_variable_is_read() {
        let requested = RefCell::new(Vec::<String>::new());
        let record = |name: &str| -> Option<String> {
            requested.borrow_mut().push(name.to_string());
            None
        };
        let _ =
            MINIMAX_MESSAGES_CONFIG.validate_environment(Vec::new(), None, "MiniMax-M2", &record);
        let _ = MINIMAX_MESSAGES_CONFIG.get_complete_url(None, "MiniMax-M2", &record);
        let requested = requested.into_inner();
        assert_eq!(
            requested,
            MINIMAX_MESSAGES_CONFIG
                .secret_names()
                .iter()
                .map(|name| name.to_string())
                .collect::<Vec<_>>()
        );
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
        let result = MINIMAX_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        assert_eq!(result.params.thinking, params.thinking);
        assert_eq!(result.params.output_config, params.output_config);
        assert_eq!(result.params.temperature, params.temperature);
    }

    #[rstest]
    #[case::anthropic_key(&[("ANTHROPIC_API_KEY", "unrelated-value")])]
    #[case::workload_identity(&[("ANTHROPIC_FEDERATION_RULE_ID", "unrelated-value")])]
    #[case::organization(&[("ANTHROPIC_ORGANIZATION_ID", "unrelated-value")])]
    #[case::full_workload_identity(&[
        ("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_prod"),
        ("ANTHROPIC_ORGANIZATION_ID", "org-prod-uuid"),
        ("ANTHROPIC_IDENTITY_TOKEN_FILE", "/unread/identity-token"),
    ])]
    fn test_non_anthropic_provider_fails_closed_without_its_own_key(
        #[case] unrelated_secrets: &[(&str, &str)],
    ) {
        let lookup = |name: &str| {
            unrelated_secrets
                .iter()
                .find_map(|(key, value)| (*key == name).then(|| (*value).into()))
        };
        let result =
            MINIMAX_MESSAGES_CONFIG.validate_environment(Vec::new(), None, "native-model", &lookup);
        assert_eq!(
            result.unwrap_err(),
            Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "MiniMax",
                environment_variable: MINIMAX_API_KEY_ENV,
            })
        );
    }
}
