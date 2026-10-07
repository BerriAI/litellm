use std::{
    collections::BTreeMap,
    sync::{Arc, LazyLock},
};

use litellm_auth::{CredentialPlacement, ResolvedCredential, SecretValue, TokenProviderHandle};
use litellm_auth_gcp::{SECRET_NAMES, VertexAuth, VertexConfig};
use litellm_core_utils::settings::resolve_non_empty;
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest, MessagesTool, OutputConfig},
    providers::anthropic::{AnthropicBeta, BetaProvider, BetaSet},
    recognized::Recognized,
};
use serde_json::{Map, Value};
use url::Url;

use crate::{
    Error, ErrorDetail,
    anthropic::{
        beta_headers::BetaPolicy,
        common_utils::{merge_beta_headers, supports_effort_param},
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::{
                PARTNER_HOST_REQUEST_POLICY, transform_messages_request_with,
                update_headers_with_anthropic_beta,
            },
        },
    },
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        messages::{
            context::{MessagesModelCapabilities, MessagesTransformContext},
            normalization::normalize_system_role_messages,
            transformation::{BaseMessagesConfig, messages_request_body},
        },
    },
    vertex_ai::common_utils::{VERTEX_AI_API_KEY_ENV, VERTEXAI_API_KEY_ENV, VertexTarget},
};

pub const VERTEX_ANTHROPIC_VERSION: &str = "vertex-2023-10-16";
const ANTHROPIC_VERSION_FIELD: &str = "anthropic_version";
const EFFORT_FIELD: &str = "effort";
const WEB_SEARCH_TOOL_PREFIX: &str = "web_search";
const API_VERSIONS: [&str; 2] = ["v1", "v1beta1"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Endpoint {
    RawPredict,
    StreamRawPredict,
}

impl Endpoint {
    fn as_str(self) -> &'static str {
        match self {
            Self::RawPredict => "rawPredict",
            Self::StreamRawPredict => "streamRawPredict",
        }
    }
}

pub struct VertexAiMessagesConfig {
    auth: fn() -> &'static VertexAuth,
}

pub const VERTEX_AI_MESSAGES_CONFIG: VertexAiMessagesConfig = VertexAiMessagesConfig {
    auth: shared_vertex_auth,
};

fn shared_vertex_auth() -> &'static VertexAuth {
    static AUTH: LazyLock<VertexAuth> = LazyLock::new(VertexAuth::default);
    &AUTH
}

impl BaseMessagesConfig for VertexAiMessagesConfig {
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
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        vertex_claude_url(api_base, model, env_lookup, Endpoint::RawPredict)
    }

    fn complete_stream_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        vertex_claude_url(api_base, model, env_lookup, Endpoint::StreamRawPredict)
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        let request = transform_messages_request_with(
            normalize_system_role_messages(
                request,
                context
                    .thinking
                    .capabilities
                    .supports_mid_conversation_system,
            ),
            context,
            PARTNER_HOST_REQUEST_POLICY,
        )?;
        let params = request.params;
        Ok(MessagesRequest {
            params: MessagesOptionalParams {
                output_config: params.output_config.and_then(|output_config| {
                    sanitize_output_config(output_config, &context.thinking.capabilities)
                }),
                extra: params
                    .extra
                    .into_iter()
                    .chain([(
                        ANTHROPIC_VERSION_FIELD.to_string(),
                        Value::String(VERTEX_ANTHROPIC_VERSION.to_string()),
                    )])
                    .collect(),
                ..params
            },
            ..request
        })
    }

    fn request_body(&self, request: &MessagesRequest) -> Result<Value, Error> {
        let Value::Object(fields) = messages_request_body(request)? else {
            unreachable!("MessagesRequest serializes as an object")
        };
        Ok(Value::Object(
            fields
                .into_iter()
                .filter(|(key, _)| key != "model")
                .collect(),
        ))
    }

    fn secret_names(&self) -> &'static [&'static str] {
        SECRET_NAMES
    }

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let headers = litellm_http::request::without_headers(headers, &["authorization"]);
        let auth = match resolve_non_empty(
            api_key,
            env_lookup,
            &[VERTEX_AI_API_KEY_ENV, VERTEXAI_API_KEY_ENV],
        ) {
            Some(token) => AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret: SecretValue::new(token),
            },
            None => AuthScheme::Token {
                provider: access_token_provider((self.auth)(), env_lookup),
            },
        };
        Ok(ValidatedEnvironment { headers, auth })
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        BetaPolicy::Filter(BetaProvider::VertexAi).apply(merge_beta_headers(
            update_headers_with_anthropic_beta(headers, request),
            vertex_feature_betas(&request.params),
        ))
    }
}

/// The token source reads the credential settings when the request is sent, so the
/// settings visible now are copied for it.
fn access_token_provider(
    auth: &'static VertexAuth,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> TokenProviderHandle {
    let settings: Arc<BTreeMap<&'static str, String>> = Arc::new(
        SECRET_NAMES
            .iter()
            .filter_map(|name| env_lookup(name).map(|value| (*name, value)))
            .collect(),
    );
    TokenProviderHandle::from_callback(move || {
        let settings = Arc::clone(&settings);
        async move {
            let token = auth
                .access_token(&VertexConfig::default(), &|name: &str| {
                    settings.get(name).cloned()
                })
                .await?;
            Ok(ResolvedCredential::AccessToken {
                token: SecretValue::new(token),
                expires_on: None,
            })
        }
    })
}

/// Python fixes the URL while validating the environment because its project can come from
/// the credentials. Here the project and location resolve from the same settings on every
/// call, so the URL and the environment always agree.
fn vertex_claude_url(
    api_base: Option<&str>,
    model: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
    endpoint: Endpoint,
) -> Result<String, Error> {
    let target = VertexTarget::resolve(&VertexConfig::default(), env_lookup)?;
    let prediction = format!("{model}:{}", endpoint.as_str());
    let model_path = [
        "projects",
        &target.project,
        "locations",
        target.location.as_str(),
        "publishers",
        "anthropic",
        "models",
        &prediction,
    ];
    let custom_base = api_base.map(str::trim).filter(|base| !base.is_empty());
    let Some(custom_base) = custom_base else {
        return with_path(target.location.base_url(), &["v1"], &model_path).map(Into::into);
    };
    let base = Url::parse(custom_base)
        .map_err(|error| Error::InvalidRequest(ErrorDetail::invalid("api_base", error)))?;
    let segments: Vec<&str> = base
        .path_segments()
        .ok_or_else(invalid_api_base)?
        .filter(|segment| !segment.is_empty())
        .collect();
    let version: Option<&[&str]> = match segments.as_slice() {
        [] => Some(&["v1"]),
        [version] if API_VERSIONS.contains(version) => Some(&[]),
        _ => None,
    };
    let url = match version {
        Some(version) => with_path(base, version, &model_path)?,
        None => with_endpoint_suffix(base, endpoint),
    };
    Ok(match endpoint {
        Endpoint::RawPredict => url,
        Endpoint::StreamRawPredict => with_sse_query(url),
    }
    .into())
}

fn invalid_api_base() -> Error {
    Error::InvalidRequest(ErrorDetail::Message(
        "api_base must be a URL that can carry a path".into(),
    ))
}

fn with_path(mut url: Url, version: &[&str], model_path: &[&str]) -> Result<Url, Error> {
    url.path_segments_mut()
        .map_err(|()| invalid_api_base())?
        .pop_if_empty()
        .extend(version.iter().chain(model_path));
    Ok(url)
}

fn with_endpoint_suffix(mut url: Url, endpoint: Endpoint) -> Url {
    let path = format!("{}:{}", url.path(), endpoint.as_str());
    url.set_path(&path);
    url
}

fn with_sse_query(mut url: Url) -> Url {
    url.query_pairs_mut().append_pair("alt", "sse");
    url
}

fn vertex_feature_betas(params: &MessagesOptionalParams) -> BetaSet {
    let has_safeguards = params
        .safeguards
        .as_ref()
        .is_some_and(|value| !matches!(value, Recognized::Unrecognized(Value::Null)));
    [
        uses_web_search(params.tools.as_deref()).then_some(AnthropicBeta::WebSearch20250305),
        has_safeguards.then_some(AnthropicBeta::DangerousToolUse20260903),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn uses_web_search(tools: Option<&[Recognized<MessagesTool>]>) -> bool {
    tools.into_iter().flatten().any(|tool| match tool {
        Recognized::Known(MessagesTool::Builtin(tool)) => tool.is_web_search(),
        Recognized::Known(_) => false,
        Recognized::Unrecognized(tool) => tool
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|tool_type| tool_type.starts_with(WEB_SEARCH_TOOL_PREFIX)),
    })
}

fn sanitize_output_config(
    output_config: Recognized<OutputConfig>,
    capabilities: &MessagesModelCapabilities,
) -> Option<Recognized<OutputConfig>> {
    let keeps_effort = supports_effort_param(capabilities);
    match output_config {
        Recognized::Known(config) => {
            let config = OutputConfig {
                effort: config.effort.filter(|_| keeps_effort),
                ..config
            };
            (!config.is_empty()).then_some(Recognized::Known(config))
        }
        Recognized::Unrecognized(Value::Object(fields)) => {
            let fields: Map<String, Value> = fields
                .into_iter()
                .filter(|(key, _)| keeps_effort || key != EFFORT_FIELD)
                .collect();
            (!fields.is_empty()).then_some(Recognized::Unrecognized(Value::Object(fields)))
        }
        Recognized::Unrecognized(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use litellm_auth::AuthServices;
    use litellm_auth_gcp::{
        CredentialSource, VertexAuthFuture, VertexProviderLoader, VertexTokenSource,
    };
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::base_llm::auth::resolve_auth;

    struct EchoSource(String);

    impl VertexTokenSource for EchoSource {
        fn project_id(&self) -> VertexAuthFuture<'_, String> {
            Box::pin(async { Ok("credential-project".into()) })
        }

        fn token(&self) -> VertexAuthFuture<'_, String> {
            Box::pin(async { Ok(self.0.clone()) })
        }
    }

    struct EchoLoader;

    impl VertexProviderLoader for EchoLoader {
        fn load(
            &self,
            source: CredentialSource,
        ) -> VertexAuthFuture<'_, Arc<dyn VertexTokenSource>> {
            let token = match source {
                CredentialSource::Trusted(credentials) => {
                    format!("minted-from-{}", credentials.expose())
                }
                CredentialSource::ApplicationCredentials(path) => format!("minted-from-{path}"),
                CredentialSource::Inline(_) => "minted-from-request".to_string(),
                CredentialSource::Adc => "minted-from-adc".to_string(),
            };
            Box::pin(async move { Ok(Arc::new(EchoSource(token)) as Arc<dyn VertexTokenSource>) })
        }
    }

    fn echo_auth() -> &'static VertexAuth {
        static AUTH: LazyLock<VertexAuth> = LazyLock::new(|| VertexAuth::new(Arc::new(EchoLoader)));
        &AUTH
    }

    const CONFIG: VertexAiMessagesConfig = VertexAiMessagesConfig { auth: echo_auth };

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
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

    fn request_from(value: Value) -> MessagesRequest {
        serde_json::from_value(value).expect("valid request")
    }

    fn transformed(value: Value, capabilities: MessagesModelCapabilities) -> Value {
        let context = MessagesTransformContext::with_lookup(capabilities, false, &|_: &str| None);
        serde_json::to_value(
            CONFIG
                .transform_anthropic_messages_request(request_from(value), &context)
                .expect("request transforms"),
        )
        .expect("serializable request")
    }

    const PROJECT_ENV: &[(&str, &str)] = &[("VERTEXAI_PROJECT", "proj-1")];

    #[rstest]
    #[case::regional_default(
        None,
        &[("VERTEXAI_PROJECT", "proj-1")],
        "https://us-central1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-central1/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict"
    )]
    #[case::global(
        None,
        &[("VERTEXAI_PROJECT", "proj-1"), ("VERTEXAI_LOCATION", "global")],
        "https://aiplatform.googleapis.com/v1/projects/proj-1/locations/global/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict"
    )]
    #[case::multi_region_geography(
        None,
        &[("VERTEXAI_PROJECT", "proj-1"), ("VERTEX_LOCATION", "eu")],
        "https://aiplatform.eu.rep.googleapis.com/v1/projects/proj-1/locations/eu/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict"
    )]
    #[case::custom_host(
        Some("https://gateway.example/"),
        &[("VERTEXAI_PROJECT", "proj-1")],
        "https://gateway.example/v1/projects/proj-1/locations/us-central1/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict"
    )]
    #[case::custom_host_with_beta_version(
        Some("https://gateway.example/v1beta1"),
        &[("VERTEXAI_PROJECT", "proj-1")],
        "https://gateway.example/v1beta1/projects/proj-1/locations/us-central1/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict"
    )]
    #[case::custom_full_endpoint(
        Some("https://gateway.example/vertex/claude"),
        &[("VERTEXAI_PROJECT", "proj-1")],
        "https://gateway.example/vertex/claude:rawPredict"
    )]
    fn the_url_targets_the_claude_publisher(
        #[case] api_base: Option<&str>,
        #[case] settings: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        assert_eq!(
            CONFIG
                .get_complete_url(api_base, "claude-sonnet-4-5@20250929", &env(settings))
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::default_host(
        None,
        "https://us-central1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-central1/publishers/anthropic/models/claude:streamRawPredict"
    )]
    #[case::custom_host(
        Some("https://gateway.example"),
        "https://gateway.example/v1/projects/proj-1/locations/us-central1/publishers/anthropic/models/claude:streamRawPredict?alt=sse"
    )]
    fn streaming_uses_the_stream_endpoint(#[case] api_base: Option<&str>, #[case] expected: &str) {
        assert_eq!(
            CONFIG
                .complete_stream_url(api_base, "claude", &env(PROJECT_ENV))
                .unwrap(),
            expected
        );
    }

    #[test]
    fn an_injected_location_never_reaches_the_hostname() {
        let error = CONFIG
            .get_complete_url(
                None,
                "claude",
                &env(&[
                    ("VERTEXAI_PROJECT", "p"),
                    ("VERTEXAI_LOCATION", "attacker.example/x"),
                ]),
            )
            .unwrap_err();
        assert!(matches!(error, Error::InvalidRequest(_)));
    }

    #[test]
    fn project_and_model_cannot_escape_their_path_segments() {
        let url = CONFIG
            .get_complete_url(
                None,
                "claude/../../evil",
                &env(&[("VERTEXAI_PROJECT", "p/../other?x=1")]),
            )
            .unwrap();
        let parsed = Url::parse(&url).unwrap();
        assert_eq!(
            parsed.host_str(),
            Some("us-central1-aiplatform.googleapis.com")
        );
        assert_eq!(parsed.query(), None);
        assert_eq!(parsed.path_segments().unwrap().count(), 9);
    }

    #[test]
    fn a_missing_project_is_a_configuration_error() {
        assert!(matches!(
            CONFIG.get_complete_url(None, "claude", &|_| None),
            Err(Error::Auth(litellm_auth::Error::InvalidConfiguration(_)))
        ));
    }

    async fn sent_headers(
        forwarded: &[(&str, &str)],
        api_key: Option<&str>,
        settings: &'static [(&'static str, &'static str)],
    ) -> Headers {
        let validated = CONFIG
            .validate_environment(headers(forwarded), api_key, "claude", &env(settings))
            .unwrap();
        resolve_auth(&AuthServices::default(), validated, &|_| None)
            .await
            .unwrap()
            .headers
    }

    #[rstest]
    #[case::stale_forwarded_token_does_not_shadow_explicit_token(
        &[("Authorization", "Bearer expired")],
        Some("fresh-static"),
        &[],
        &[("authorization", "Bearer fresh-static")]
    )]
    #[case::stale_forwarded_token_does_not_skip_refresh(
        &[("Authorization", "Bearer expired")],
        None,
        &[("VERTEXAI_CREDENTIALS", "refresh-fixture")],
        &[("authorization", "Bearer minted-from-refresh-fixture")]
    )]
    #[case::api_key_is_a_static_bearer(&[], Some("static"), &[("VERTEX_AI_API_KEY", "env-key")], &[("authorization", "Bearer static")])]
    #[case::environment_static_token(&[], None, &[("VERTEXAI_API_KEY", "env-key")], &[("authorization", "Bearer env-key")])]
    #[case::minted_from_configured_credentials(
        &[],
        None,
        &[("VERTEXAI_CREDENTIALS", "sa-json")],
        &[("authorization", "Bearer minted-from-sa-json")]
    )]
    #[case::minted_from_application_credentials(
        &[],
        None,
        &[("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")],
        &[("authorization", "Bearer minted-from-/sa.json")]
    )]
    #[case::minted_from_adc(&[], Some("  "), &[], &[("authorization", "Bearer minted-from-adc")])]
    #[tokio::test]
    async fn the_bearer_comes_from_the_first_available_google_credential(
        #[case] forwarded: &[(&str, &str)],
        #[case] api_key: Option<&str>,
        #[case] settings: &'static [(&'static str, &'static str)],
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(
            sent_headers(forwarded, api_key, settings).await,
            headers(expected)
        );
    }

    #[test]
    fn default_headers_carry_no_anthropic_version_header() {
        assert_eq!(
            CONFIG.default_headers(),
            &[("content-type", "application/json")]
        );
    }

    #[rstest::rstest]
    fn the_body_carries_the_vertex_version_and_strips_cache_scope() {
        let body = transformed(
            json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "anthropic_version": "caller-version",
                "cache_control": {"type": "ephemeral", "scope": "global"},
                "system": [
                    {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
                    {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "global"}}
                ],
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "scope": "global"}}
                    ]},
                    {"role": "system", "content": "folded"}
                ]
            }),
            MessagesModelCapabilities::default(),
        );
        assert_eq!(body["anthropic_version"], json!(VERTEX_ANTHROPIC_VERSION));
        assert_eq!(body["cache_control"], json!({"type": "ephemeral"}));
        assert_eq!(
            body["system"],
            json!([
                {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
            ])
        );
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
            ]}, {"role":"user", "content":[
                {"type":"text", "text":crate::base_llm::messages::normalization::CONVERTED_SYSTEM_NOTE},
                {"type":"text", "text":"folded"}
            ]}])
        );
    }

    #[test]
    fn thinking_follows_claude_rules() {
        let body = transformed(
            json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "temperature": 0.5,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [{"role": "user", "content": "hi"}]
            }),
            MessagesModelCapabilities::default(),
        );
        assert_eq!(body.get("temperature"), None);
        assert_eq!(body["thinking"]["type"], json!("enabled"));
    }

    const EFFORT_CAPABLE: MessagesModelCapabilities = MessagesModelCapabilities {
        supports_reasoning: false,
        supports_adaptive_thinking: false,
        thinking_always_on: false,
        supports_legacy_thinking: false,
        supports_output_config: true,
        supports_sampling_params: true,
        supports_speed: false,
        supports_mid_conversation_system: false,
        supports_cache_control_ttl: false,
        supports_native_structured_output: false,
        supports_tool_search: false,
        effort_ceiling: None,
        effort_tiers: crate::base_llm::messages::context::SupportedEffortTiers {
            minimal: false,
            low: false,
            medium: false,
            high: false,
            xhigh: false,
            max: false,
        },
    };

    #[rstest]
    #[case::effort_kept_when_accepted(
        EFFORT_CAPABLE,
        json!({"effort": "high"}),
        Some(json!({"effort": "high"}))
    )]
    #[case::effort_dropped_and_empty_config_removed(
        MessagesModelCapabilities::default(),
        json!({"effort": "high"}),
        None
    )]
    #[case::format_survives_without_effort(
        MessagesModelCapabilities::default(),
        json!({"effort": "high", "format": {"type": "json_schema", "schema": {"type": "object"}}}),
        Some(json!({"format": {"type": "json_schema", "schema": {"type": "object"}}}))
    )]
    #[case::non_object_dropped(EFFORT_CAPABLE, json!("high"), None)]
    fn output_config_effort_follows_the_model(
        #[case] capabilities: MessagesModelCapabilities,
        #[case] output_config: Value,
        #[case] expected: Option<Value>,
    ) {
        let body = transformed(
            json!({
                "model": "claude-haiku",
                "max_tokens": 16,
                "output_config": output_config,
                "messages": [{"role": "user", "content": "hi"}]
            }),
            capabilities,
        );
        assert_eq!(body.get("output_config").cloned(), expected);
    }

    fn beta_header_for(fields: Value, forwarded: &[(&str, &str)]) -> Option<String> {
        let Value::Object(fields) = fields else {
            panic!("case fields are an object")
        };
        let request = request_from(Value::Object(
            [
                ("model".to_string(), json!("claude")),
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
        CONFIG
            .request_headers(headers(forwarded), &request)
            .into_iter()
            .find(|(name, _)| name == "anthropic-beta")
            .map(|(_, value)| value)
    }

    fn vertex_name(beta: AnthropicBeta) -> String {
        beta.on(BetaProvider::VertexAi)
            .expect("the beta is accepted on Vertex")
            .to_string()
    }

    #[test]
    fn tool_search_sends_the_vertex_tool_search_beta() {
        assert_eq!(
            beta_header_for(
                json!({"tools": [{"type": "tool_search_tool_regex_20251119", "name": "tool_search"}]}),
                &[]
            ),
            Some(vertex_name(AnthropicBeta::AdvancedToolUse20251120))
        );
        assert_ne!(
            vertex_name(AnthropicBeta::AdvancedToolUse20251120),
            AnthropicBeta::AdvancedToolUse20251120.to_string()
        );
    }

    #[rstest]
    #[case::web_search_tool(
        json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
        AnthropicBeta::WebSearch20250305
    )]
    #[case::web_search_20260209(
        json!({"tools": [{"type": "web_search_20260209", "name": "web_search"}]}),
        AnthropicBeta::WebSearch20250305
    )]
    #[case::web_search_20260318(
        json!({"tools": [{"type": "web_search_20260318", "name": "web_search"}]}),
        AnthropicBeta::WebSearch20250305
    )]
    #[case::future_web_search_tool(
        json!({"tools": [{"type": "web_search_20991231", "name": "web_search"}]}),
        AnthropicBeta::WebSearch20250305
    )]
    #[case::safeguards(
        json!({"safeguards": [{"type": "dangerous_tool_use"}]}),
        AnthropicBeta::DangerousToolUse20260903
    )]
    fn vertex_specific_features_add_their_beta(#[case] fields: Value, #[case] beta: AnthropicBeta) {
        assert_eq!(beta_header_for(fields, &[]), Some(vertex_name(beta)));
    }

    #[rstest]
    #[case::web_fetch(json!({"tools": [{"type": "web_fetch_20260318", "name": "web_search"}]}))]
    #[case::custom_tool(json!({"tools": [{"name": "web_search", "input_schema": {"type": "object"}}]}))]
    #[case::explicit_custom_tool(json!({"tools": [{"type": "custom", "name": "web_search"}]}))]
    #[case::case_sensitive(json!({"tools": [{"type": "Web_Search_20991231"}]}))]
    fn unrelated_tools_do_not_add_the_web_search_beta(#[case] fields: Value) {
        assert_eq!(beta_header_for(fields, &[]), None);
    }

    #[test]
    fn betas_vertex_rejects_are_removed() {
        let rejected = AnthropicBeta::KNOWN
            .into_iter()
            .find(|beta| beta.on(BetaProvider::VertexAi).is_none())
            .expect("Vertex rejects at least one beta");
        assert_eq!(
            beta_header_for(json!({}), &[("anthropic-beta", rejected.as_str())]),
            None
        );
        assert_eq!(beta_header_for(json!({}), &[]), None);
    }

    #[test]
    fn secret_names_cover_every_lookup() {
        let requested = RefCell::new(Vec::<String>::new());
        let record = |name: &str| -> Option<String> {
            requested.borrow_mut().push(name.to_string());
            None
        };
        let _ = CONFIG.validate_environment(Vec::new(), None, "claude", &record);
        let _ = CONFIG.get_complete_url(None, "claude", &record);
        let _ = CONFIG.complete_stream_url(None, "claude", &record);
        let requested = requested.into_inner();
        assert!(!requested.is_empty());
        let undeclared: Vec<&String> = requested
            .iter()
            .filter(|name| !CONFIG.secret_names().contains(&name.as_str()))
            .collect();
        assert_eq!(undeclared, Vec::<&String>::new());
    }
    #[rstest]
    #[case::supported_mid_conversation(true, false, true)]
    #[case::supported_leading_run(true, true, true)]
    #[case::unsupported_mid_conversation(false, false, false)]
    #[case::unsupported_leading_run(false, true, false)]
    fn test_messages_preserve_mid_conversation_system_by_capability(
        #[case] supports_mid_conversation_system: bool,
        #[case] leading_system: bool,
        #[case] expected_system_role: bool,
    ) {
        use litellm_llms_types::formats::messages::{
            ContentBlock, Message, MessageContent, MessageRole, MessagesOptionalParams,
            SystemPrompt,
        };
        let message = |role, text: &str| Message {
            role,
            content: MessageContent::Text(text.into()),
            extra: Default::default(),
        };
        let messages = leading_system
            .then(|| {
                [
                    message(MessageRole::System, "leading"),
                    message(MessageRole::System, "second leading"),
                ]
            })
            .into_iter()
            .flatten()
            .chain([
                message(MessageRole::User, "first"),
                message(MessageRole::Assistant, "answer"),
                message(MessageRole::System, "later"),
                message(MessageRole::User, "last"),
            ])
            .collect();
        let request = MessagesRequest {
            model: "test-model".into(),
            messages,
            params: MessagesOptionalParams {
                max_tokens: Some(4096),
                ..Default::default()
            },
        };
        let context = MessagesTransformContext::with_lookup(
            MessagesModelCapabilities {
                supports_mid_conversation_system,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        let result = VERTEX_AI_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &context)
            .unwrap();
        assert_eq!(result.messages.len(), 4);
        assert_eq!(result.messages[0], message(MessageRole::User, "first"));
        assert_eq!(
            result.messages[1],
            message(MessageRole::Assistant, "answer")
        );
        assert_eq!(result.messages[3], message(MessageRole::User, "last"));
        assert_eq!(
            result.messages[2].role,
            if expected_system_role {
                MessageRole::System
            } else {
                MessageRole::User
            }
        );
        assert_eq!(
            result.messages[2].content,
            if expected_system_role {
                MessageContent::Text("later".into())
            } else {
                MessageContent::Blocks(vec![
                    ContentBlock::text(
                        "Operator note (not from the user): the following was originally a mid-conversation system-role reminder.",
                    ),
                    ContentBlock::text("later"),
                ])
            }
        );
        assert_eq!(
            result.params.system,
            leading_system.then(|| SystemPrompt::Blocks(vec![
                ContentBlock::text("leading"),
                ContentBlock::text("second leading")
            ]))
        );
    }

    #[rstest]
    #[case::nonstream(false)]
    #[case::stream(true)]
    fn vertex_wire_body_uses_the_url_model_and_preserves_stream(#[case] stream: bool) {
        use litellm_llms_types::formats::messages::{Message, MessageContent, MessageRole};
        let request = MessagesRequest {
            model: "claude-model".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text("hello".into()),
                extra: Default::default(),
            }],
            params: MessagesOptionalParams {
                max_tokens: Some(16),
                stream: Some(stream),
                ..Default::default()
            },
        };
        let result = VERTEX_AI_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        let body = VERTEX_AI_MESSAGES_CONFIG.request_body(&result).unwrap();
        assert_eq!(
            body,
            json!({ "messages": [{"role": "user", "content": "hello"}], "max_tokens": 16, "stream": stream, "anthropic_version": VERTEX_ANTHROPIC_VERSION })
        );
        assert_eq!(result.model, "claude-model");
    }
    #[rstest]
    #[case::caller_omits_beta(true, None)]
    #[case::caller_sends_only_other_beta(true, Some("interleaved-thinking-2025-05-14"))]
    #[case::caller_sends_beta(true, Some("dangerous-tool-use-2026-09-03"))]
    #[case::caller_sends_mixed_betas(
        true,
        Some("dangerous-tool-use-2026-09-03,interleaved-thinking-2025-05-14")
    )]
    #[case::no_safeguards(false, None)]
    fn test_messages_forwards_safeguards_with_one_dangerous_tool_use_beta(
        #[case] enabled: bool,
        #[case] beta_header: Option<&str>,
    ) {
        use litellm_llms_types::formats::messages::{
            Message, MessageContent, MessageRole, Safeguard,
        };
        let safeguards = Recognized::Known(vec![Recognized::Known(Safeguard {
            safeguard_type: "dangerous_tool_use".into(),
            classifier_context: Some(Recognized::Known(
                [
                    ("v".into(), json!(1)),
                    ("permission_mode".into(), json!("auto")),
                ]
                .into_iter()
                .collect(),
            )),
            extra: Map::new(),
        })]);
        let request = MessagesRequest {
            model: "test-model".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text("hello".into()),
                extra: Map::new(),
            }],
            params: MessagesOptionalParams {
                max_tokens: Some(64),
                safeguards: enabled.then_some(safeguards.clone()),
                ..Default::default()
            },
        };
        let result = VERTEX_AI_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        assert_eq!(result.params.safeguards, enabled.then_some(safeguards));
        let headers = beta_header
            .map(|beta| ("anthropic-beta".into(), beta.into()))
            .into_iter()
            .collect();
        let headers = VERTEX_AI_MESSAGES_CONFIG.request_headers(headers, &result);
        let betas = crate::anthropic::common_utils::existing_betas(&headers);
        assert_eq!(
            betas
                .iter()
                .filter(|beta| beta.as_str() == "dangerous-tool-use-2026-09-03")
                .count(),
            usize::from(enabled)
        );
        let response: litellm_llms_types::formats::messages::MessagesResponse = serde_json::from_value(json!({
            "id":"msg_1", "type":"message", "role":"assistant", "model":"model",
            "content":[{"type":"text","text":"ok"}], "stop_reason":"end_turn", "stop_sequence":null,
            "usage":{"input_tokens":1,"output_tokens":2},
            "safeguard_results":[{"type":"dangerous_tool_use", "status":{"type":"available", "tool_uses":{"toolu_01":{"type":"evaluated", "outcome":"not_flagged"}}}}]
        })).unwrap();
        let actual = VERTEX_AI_MESSAGES_CONFIG
            .transform_anthropic_messages_response("model", response.clone())
            .unwrap();
        assert_eq!(actual, response);
    }
    #[rstest]
    #[case::adaptive_entry(true)]
    #[case::entry_override_disables_adaptive(false)]
    fn test_messages_thinking_shape_follows_injected_provider_entry_flag(#[case] adaptive: bool) {
        use litellm_llms_types::formats::{
            chat_completions::ReasoningEffort,
            messages::{
                EffortLevel, Message, MessageContent, MessageRole, OutputConfig, ThinkingConfig,
                ThinkingDisplay,
            },
        };
        let request = MessagesRequest {
            model: "same-model".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text("hello".into()),
                extra: Default::default(),
            }],
            params: MessagesOptionalParams {
                max_tokens: Some(4096),
                reasoning_effort: Some(Recognized::Known(ReasoningEffort::Medium)),
                ..Default::default()
            },
        };
        let context = MessagesTransformContext::with_lookup(
            crate::base_llm::messages::context::MessagesModelCapabilities {
                supports_adaptive_thinking: adaptive,
                supports_output_config: adaptive,
                supports_reasoning: true,
                supports_legacy_thinking: true,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        let result = VERTEX_AI_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &context)
            .unwrap();
        assert_eq!(
            result.params.thinking,
            Some(Recognized::Known(if adaptive {
                ThinkingConfig::adaptive(Some(ThinkingDisplay::Summarized))
            } else {
                ThinkingConfig::enabled(2048)
            }))
        );
        assert_eq!(
            result.params.output_config,
            adaptive.then(|| Recognized::Known(OutputConfig {
                effort: Some(Recognized::Known(EffortLevel::Medium)),
                ..Default::default()
            }))
        );
    }
    #[rstest]
    #[case::drop(true)]
    #[case::reject(false)]
    fn test_vertex_fast_mode_follows_drop_params(#[case] drop_params: bool) {
        use litellm_llms_types::formats::messages::{Message, MessageContent, MessageRole, Speed};
        let request = MessagesRequest {
            model: "test-model".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text("Hello".into()),
                extra: Map::new(),
            }],
            params: MessagesOptionalParams {
                max_tokens: Some(1024),
                speed: Some(Recognized::Known(Speed::Fast)),
                ..Default::default()
            },
        };
        let context = MessagesTransformContext::with_lookup(
            MessagesModelCapabilities::default(),
            drop_params,
            &|_: &str| None,
        );
        let result =
            VERTEX_AI_MESSAGES_CONFIG.transform_anthropic_messages_request(request, &context);
        if drop_params {
            assert!(result.unwrap().params.speed.is_none());
        } else {
            assert!(matches!(result, Err(Error::InvalidRequest(_))));
        }
    }
    #[rstest]
    #[case::no_edits(vec![], None)]
    #[case::compact(vec![litellm_llms_types::formats::messages::ContextEdit::Compact { trigger: None, extra: Map::new() }], Some("compact-2026-01-12"))]
    #[case::clear_tools(vec![litellm_llms_types::formats::messages::ContextEdit::ClearToolUses { extra: Map::new() }], Some("context-management-2025-06-27"))]
    #[case::both(vec![litellm_llms_types::formats::messages::ContextEdit::Compact { trigger: None, extra: Map::new() }, litellm_llms_types::formats::messages::ContextEdit::ClearToolUses { extra: Map::new() }], Some("compact-2026-01-12,context-management-2025-06-27"))]
    fn test_vertex_context_edits_add_exact_feature_headers(
        #[case] edits: Vec<litellm_llms_types::formats::messages::ContextEdit>,
        #[case] expected: Option<&str>,
    ) {
        use litellm_llms_types::formats::messages::{
            ContextManagement, Message, MessageContent, MessageRole,
        };
        let request = MessagesRequest {
            model: "model".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text("hello".into()),
                extra: Map::new(),
            }],
            params: MessagesOptionalParams {
                max_tokens: Some(16),
                context_management: Some(Recognized::Known(ContextManagement {
                    edits: Some(edits.into_iter().map(Recognized::Known).collect()),
                    extra: Map::new(),
                })),
                ..Default::default()
            },
        };
        let actual = CONFIG.request_headers(vec![], &request);
        let expected: Headers = expected
            .map(|beta| ("anthropic-beta".into(), beta.into()))
            .into_iter()
            .collect();
        assert_eq!(actual, expected);
    }

    #[rstest]
    #[case::absent(None)]
    #[case::empty(Some(OutputConfig::default()))]
    #[case::effort(Some(OutputConfig { effort: Some(Recognized::Known(litellm_llms_types::formats::messages::EffortLevel::Low)), ..Default::default() }))]
    fn test_vertex_system_output_config_adds_per_turn_header(
        #[case] output_config: Option<OutputConfig>,
    ) {
        use litellm_llms_types::formats::messages::{
            ContentBlock, Message, MessageContent, MessageRole,
        };
        let expected = output_config
            .is_some()
            .then(|| {
                (
                    "anthropic-beta".into(),
                    "per-turn-control-2026-07-01".into(),
                )
            })
            .into_iter()
            .collect::<Headers>();
        let request = MessagesRequest {
            model: "model".into(),
            messages: vec![
                Message {
                    role: MessageRole::User,
                    content: MessageContent::Text("hello".into()),
                    extra: Map::new(),
                },
                Message {
                    role: MessageRole::System,
                    content: MessageContent::Blocks(vec![ContentBlock::text("environment")]),
                    extra: output_config
                        .map(|config| {
                            (
                                "output_config".into(),
                                serde_json::to_value(config).unwrap(),
                            )
                        })
                        .into_iter()
                        .collect(),
                },
            ],
            params: MessagesOptionalParams {
                max_tokens: Some(16),
                ..Default::default()
            },
        };
        let context = MessagesTransformContext::with_lookup(
            MessagesModelCapabilities {
                supports_mid_conversation_system: true,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        let transformed = CONFIG
            .transform_anthropic_messages_request(request.clone(), &context)
            .unwrap();
        assert_eq!(transformed.messages, request.messages);
        assert_eq!(CONFIG.request_headers(vec![], &transformed), expected);
    }
}
