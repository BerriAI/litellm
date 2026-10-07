use litellm_auth::{
    CredentialPlacement, CredentialPlanKind, CredentialRule, ExistingHeaderBehavior,
    ProviderAuthPolicy, SecretValue,
};
use litellm_core_utils::settings::resolve_non_empty;
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
    tencent::common_utils::{DEFAULT_TENCENT_API_BASE, TENCENT_API_BASE_ENV, TENCENT_API_KEY_ENV},
};

pub const TENCENT_ANTHROPIC_API_BASE_ENV: &str = "TENCENT_ANTHROPIC_API_BASE";
const CHAT_COMPLETIONS_PATH_SUFFIX: &str = "/v1/chat/completions";
const BETA_POLICY: BetaPolicy = BetaPolicy::Drop;

const TENCENT_AUTH_POLICY: ProviderAuthPolicy = ProviderAuthPolicy {
    rules: &[CredentialRule {
        kind: CredentialPlanKind::Static,
        placement: CredentialPlacement::Header("x-api-key"),
    }],
    accepted_existing_headers: &["x-api-key", "authorization"],
    existing_header_behavior: ExistingHeaderBehavior::Preserve,
    scope: None,
    audience: None,
};

const REQUEST_POLICY: RequestPolicy = RequestPolicy {
    compaction: Compaction::Drop,
    ..COMPATIBLE_HOST_REQUEST_POLICY
};

pub struct TencentMessagesConfig;

pub const TENCENT_MESSAGES_CONFIG: TencentMessagesConfig = TencentMessagesConfig;

impl BaseMessagesConfig for TencentMessagesConfig {
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
        Ok(complete_tencent_messages_url(api_base, env_lookup))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        transform_messages_request_with(request, context, REQUEST_POLICY)
    }

    fn secret_names(&self) -> &'static [&'static str] {
        &[
            TENCENT_API_KEY_ENV,
            TENCENT_ANTHROPIC_API_BASE_ENV,
            TENCENT_API_BASE_ENV,
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
            &TENCENT_AUTH_POLICY,
            headers,
            resolve_non_empty(api_key, env_lookup, &[TENCENT_API_KEY_ENV]).map(SecretValue::new),
            litellm_auth::Error::MissingApiKey {
                provider: "Tencent",
                environment_variable: TENCENT_API_KEY_ENV,
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

/// The Messages endpoint under the Tencent base, which may be the chat completions base.
pub fn complete_tencent_messages_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let api_base = resolve_non_empty(
        api_base,
        env_lookup,
        &[TENCENT_ANTHROPIC_API_BASE_ENV, TENCENT_API_BASE_ENV],
    )
    .unwrap_or_else(|| DEFAULT_TENCENT_API_BASE.to_string());
    let api_base = api_base.trim_end_matches('/');
    complete_messages_url(
        api_base
            .strip_suffix(CHAT_COMPLETIONS_PATH_SUFFIX)
            .unwrap_or(api_base),
    )
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use litellm_llms_types::formats::messages::MessagesResponse;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        anthropic::common_utils::{ANTHROPIC_API_KEY_ENV, ANTHROPIC_AUTH_TOKEN_ENV},
        base_llm::{auth::AuthScheme, messages::context::MessagesModelCapabilities},
    };

    type Env = &'static [(&'static str, &'static str)];

    fn env(vars: Env) -> impl Fn(&str) -> Option<String> {
        move |name| {
            vars.iter()
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
            panic!("fields are an object")
        };
        serde_json::from_value(Value::Object(
            [
                ("model".to_string(), json!("hunyuan-2.0")),
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

    fn transform(fields: Value) -> Result<Value, Error> {
        let context = MessagesTransformContext::with_lookup(
            MessagesModelCapabilities {
                supports_reasoning: true,
                supports_sampling_params: true,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        TENCENT_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request(fields), &context)
            .map(|transformed| serde_json::to_value(transformed).expect("serializable"))
    }

    #[rstest]
    #[case::bare_host("https://tokenhub.example", "https://tokenhub.example/v1/messages")]
    #[case::trailing_slash("https://tokenhub.example/", "https://tokenhub.example/v1/messages")]
    #[case::version_base("https://tokenhub.example/v1", "https://tokenhub.example/v1/messages")]
    #[case::version_base_with_slash(
        "https://tokenhub.example/v1/",
        "https://tokenhub.example/v1/messages"
    )]
    #[case::chat_completions_base(
        "https://tokenhub.example/v1/chat/completions",
        "https://tokenhub.example/v1/messages"
    )]
    #[case::chat_completions_base_with_slash(
        "https://tokenhub.example/v1/chat/completions/",
        "https://tokenhub.example/v1/messages"
    )]
    #[case::complete_endpoint(
        "https://tokenhub.example/v1/messages",
        "https://tokenhub.example/v1/messages"
    )]
    fn one_base_serves_chat_and_messages(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(
            TENCENT_MESSAGES_CONFIG
                .get_complete_url(Some(base), "hunyuan", &|_| None)
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::param_wins(
        Some("https://param.example"),
        &[(TENCENT_ANTHROPIC_API_BASE_ENV, "https://anthropic.example"), (TENCENT_API_BASE_ENV, "https://shared.example")],
        "https://param.example/v1/messages"
    )]
    #[case::anthropic_base_env_before_shared(
        Some("  "),
        &[(TENCENT_ANTHROPIC_API_BASE_ENV, "https://anthropic.example"), (TENCENT_API_BASE_ENV, "https://shared.example")],
        "https://anthropic.example/v1/messages"
    )]
    #[case::shared_base_env(
        None,
        &[(TENCENT_API_BASE_ENV, "https://shared.example/v1")],
        "https://shared.example/v1/messages"
    )]
    fn the_base_resolves_from_param_then_env(
        #[case] api_base: Option<&str>,
        #[case] vars: Env,
        #[case] expected: &str,
    ) {
        assert_eq!(
            complete_tencent_messages_url(api_base, &env(vars)),
            expected
        );
    }

    #[test]
    fn without_a_base_the_default_host_is_used() {
        assert_eq!(
            complete_tencent_messages_url(None, &|_| None),
            complete_messages_url(DEFAULT_TENCENT_API_BASE)
        );
    }

    fn validated(
        forwarded: &[(&str, &str)],
        api_key: Option<&str>,
        vars: Env,
    ) -> Result<ValidatedEnvironment, Error> {
        TENCENT_MESSAGES_CONFIG.validate_environment(
            headers(forwarded),
            api_key,
            "hunyuan",
            &env(vars),
        )
    }

    #[rstest]
    #[case::param(Some("sk-param"), &[(TENCENT_API_KEY_ENV, "sk-env")], "sk-param")]
    #[case::env(Some(" "), &[(TENCENT_API_KEY_ENV, "sk-env")], "sk-env")]
    #[case::oauth_shaped_key_is_still_a_plain_key(Some("sk-ant-oat01-x"), &[], "sk-ant-oat01-x")]
    fn the_tencent_key_goes_in_x_api_key(
        #[case] api_key: Option<&str>,
        #[case] vars: Env,
        #[case] expected: &str,
    ) {
        assert!(matches!(
            validated(&[], api_key, vars).unwrap().auth,
            AuthScheme::Credential {
                placement: CredentialPlacement::Header("x-api-key"),
                ref secret
            } if secret.expose() == expected
        ));
    }

    #[test]
    fn a_missing_key_never_falls_back_to_anthropic_credentials() {
        let err = validated(
            &[],
            None,
            &[
                (ANTHROPIC_API_KEY_ENV, "sk-ant"),
                (ANTHROPIC_AUTH_TOKEN_ENV, "tok-ant"),
            ],
        )
        .expect_err("missing key");
        assert_eq!(
            err,
            Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Tencent",
                environment_variable: TENCENT_API_KEY_ENV,
            })
        );
    }

    #[rstest]
    #[case::x_api_key(&[("X-Api-Key", "caller")])]
    #[case::bearer(&[("Authorization", "Bearer caller")])]
    fn a_caller_credential_header_wins_over_the_key(#[case] forwarded: &[(&str, &str)]) {
        let validated = validated(forwarded, Some("sk-tencent"), &[]).unwrap();
        assert!(matches!(validated.auth, AuthScheme::Forwarded));
        assert_eq!(validated.headers, headers(forwarded));
    }

    #[test]
    fn default_headers_are_the_anthropic_ones() {
        assert_eq!(
            TENCENT_MESSAGES_CONFIG.default_headers(),
            DEFAULT_ANTHROPIC_HEADERS
        );
    }

    #[test]
    fn thinking_and_temperature_pass_through_as_sent() {
        let fields = json!({
            "max_tokens": 64,
            "thinking": {"type": "disabled"},
            "temperature": 0.3
        });
        let transformed = transform(fields.clone()).unwrap();
        assert_eq!(transformed["thinking"], fields["thinking"]);
        assert_eq!(transformed["temperature"], fields["temperature"]);
    }

    #[test]
    fn reasoning_effort_is_still_translated() {
        let transformed =
            transform(json!({"max_tokens": 4096, "reasoning_effort": "low"})).unwrap();
        assert_eq!(transformed.get("reasoning_effort"), None);
        assert_eq!(transformed["thinking"]["type"], json!("enabled"));
    }

    #[test]
    fn max_tokens_is_required() {
        assert_eq!(transform(json!({})), Err(Error::MissingField("max_tokens")));
    }

    #[rstest]
    #[case::only_billing(json!("x-anthropic-billing-header: cc_version=1"), Value::Null)]
    #[case::billing_among_blocks(
        json!([
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
            {"type": "text", "text": "be terse"}
        ]),
        json!([{"type": "text", "text": "be terse"}])
    )]
    fn billing_metadata_is_stripped(#[case] system: Value, #[case] expected: Value) {
        let transformed = transform(json!({"max_tokens": 64, "system": system})).unwrap();
        assert_eq!(transformed["system"], expected);
    }

    #[test]
    fn compaction_is_not_sent() {
        let fields = json!({"max_tokens": 64, "compaction": {"type": "auto"}});
        assert_eq!(
            serde_json::to_value(request(fields.clone())).unwrap()["compaction"],
            fields["compaction"]
        );
        assert_eq!(transform(fields).unwrap().get("compaction"), None);
    }

    #[test]
    fn openai_context_management_is_mapped_to_anthropic_edits() {
        let transformed = transform(json!({
            "max_tokens": 64,
            "context_management": [{"type": "compaction", "compact_threshold": 1000}]
        }))
        .unwrap();
        assert_eq!(
            transformed["context_management"]["edits"][0]["type"],
            json!("compact_20260112")
        );
    }

    #[test]
    fn advisor_history_is_stripped_without_the_advisor_tool() {
        let transformed = transform(json!({
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": [
                    {"type": "server_tool_use", "id": "srvtoolu_1", "name": "advisor", "input": {}},
                    {"type": "advisor_tool_result", "tool_use_id": "srvtoolu_1", "content": {"type": "advisor_result", "text": "a"}},
                    {"type": "text", "text": "done"}
                ]}
            ]
        }))
        .unwrap();
        assert_eq!(
            transformed["messages"][1]["content"],
            json!([{"type": "text", "text": "done"}])
        );
    }

    #[rstest]
    #[case::caller_beta(json!({"max_tokens": 16}), &[("anthropic-beta", "web-search-2025-03-05")])]
    #[case::feature_beta(
        json!({"max_tokens": 16, "speed": "fast", "output_config": {"format": {"type": "json_schema"}}}),
        &[]
    )]
    fn no_beta_reaches_tencent(#[case] fields: Value, #[case] forwarded: &[(&str, &str)]) {
        assert_eq!(
            TENCENT_MESSAGES_CONFIG.request_headers(
                headers(&[("x-api-key", "k")])
                    .into_iter()
                    .chain(headers(forwarded))
                    .collect(),
                &request(fields),
            ),
            headers(&[("x-api-key", "k")])
        );
    }

    #[test]
    fn transform_response_passes_through() {
        let response: MessagesResponse = serde_json::from_value(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": "hunyuan-2.0",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 2}
        }))
        .expect("valid response");
        assert_eq!(
            TENCENT_MESSAGES_CONFIG
                .transform_anthropic_messages_response("hunyuan-2.0", response.clone())
                .unwrap(),
            response
        );
        assert!(TENCENT_MESSAGES_CONFIG.stream_decoder().is_none());
    }

    #[test]
    fn secret_names_cover_every_credential_and_base_lookup() {
        let requested = RefCell::new(Vec::<String>::new());
        let record = |name: &str| -> Option<String> {
            requested.borrow_mut().push(name.to_string());
            None
        };
        let _ = TENCENT_MESSAGES_CONFIG.validate_environment(Vec::new(), None, "hunyuan", &record);
        let _ = TENCENT_MESSAGES_CONFIG.get_complete_url(None, "hunyuan", &record);
        let requested = requested.into_inner();
        assert!(!requested.is_empty());
        let undeclared: Vec<&String> = requested
            .iter()
            .filter(|name| {
                !TENCENT_MESSAGES_CONFIG
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
        let result = TENCENT_MESSAGES_CONFIG
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
            TENCENT_MESSAGES_CONFIG.validate_environment(Vec::new(), None, "native-model", &lookup);
        assert_eq!(
            result.unwrap_err(),
            Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Tencent",
                environment_variable: TENCENT_API_KEY_ENV,
            })
        );
    }
}
