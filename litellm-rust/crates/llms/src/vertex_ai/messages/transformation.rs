use litellm_auth_gcp::{
    SECRET_NAMES, VertexConfig, get_vertex_ai_location, get_vertex_ai_project,
    get_vertex_ai_project_from_credentials,
};
use litellm_http::request::has_header;
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest, MessagesTool, OutputConfig},
    providers::anthropic::{AnthropicBeta, BetaProvider, BetaSet},
    recognized::Recognized,
};
use litellm_router_types::LitellmParams;
use serde_json::{Value, json};

use crate::{
    Error,
    anthropic::{
        common_utils::{merge_beta_headers, supports_effort_param},
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::{provider_feature_betas, transform_messages_request},
        },
    },
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        messages::{
            context::MessagesTransformContext,
            normalization::{normalize_system_role_messages, strip_cache_control_scope},
            transformation::BaseMessagesConfig,
        },
    },
    vertex_ai::common_utils::{DEFAULT_VERTEX_LOCATION, get_vertex_base_url},
};

pub const VERTEX_ANTHROPIC_VERSION: &str = "vertex-2023-10-16";
const RAW_PREDICT: &str = "rawPredict";
const STREAM_RAW_PREDICT: &str = "streamRawPredict";
const WEB_SEARCH_TOOL_PREFIX: &str = "web_search";

/// Claude on Vertex AI, Python's `VertexAIPartnerModelsAnthropicMessagesConfig`: the Anthropic
/// payload addressed to a project and location, authenticated with a Google access token.
pub struct VertexAiPartnerModelsAnthropicMessagesConfig;

pub const VERTEX_ANTHROPIC_MESSAGES_CONFIG: VertexAiPartnerModelsAnthropicMessagesConfig =
    VertexAiPartnerModelsAnthropicMessagesConfig;

impl BaseMessagesConfig for VertexAiPartnerModelsAnthropicMessagesConfig {
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
        litellm_params: &LitellmParams,
        stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let config = VertexConfig::from_params(&litellm_params.vertex);
        let project = get_vertex_ai_project(&config, env_lookup)
            .or_else(|| get_vertex_ai_project_from_credentials(&config, env_lookup))
            .ok_or_else(|| {
                Error::Auth(litellm_auth::Error::InvalidConfiguration(
                    "Vertex AI project is required: set vertex_project, VERTEXAI_PROJECT, or credentials that name one"
                        .into(),
                ))
            })?;
        let location = get_vertex_ai_location(&config, env_lookup)
            .unwrap_or_else(|| DEFAULT_VERTEX_LOCATION.to_string());
        complete_vertex_anthropic_url(api_base, &project, &location, model, stream)
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        let request = transform_messages_request(request, context)?;
        let request = strip_cache_control_scope(normalize_system_role_messages(
            request,
            context
                .thinking
                .capabilities
                .supports_mid_conversation_system,
        ));
        let accepts_effort = supports_effort_param(&context.thinking.capabilities);
        Ok(MessagesRequest {
            params: MessagesOptionalParams {
                output_config: request
                    .params
                    .output_config
                    .and_then(|config| sanitize_output_config(config, accepts_effort)),
                extra: request
                    .params
                    .extra
                    .into_iter()
                    .chain([(
                        "anthropic_version".to_string(),
                        json!(VERTEX_ANTHROPIC_VERSION),
                    )])
                    .collect(),
                ..request.params
            },
            ..request
        })
    }

    fn secret_names(&self) -> &'static [&'static str] {
        SECRET_NAMES
    }

    /// A forwarded bearer is sent as is; otherwise a Google access token for the configured
    /// credentials (`vertex_credentials`, `VERTEXAI_CREDENTIALS`, `GOOGLE_APPLICATION_CREDENTIALS`,
    /// then application default credentials) is acquired when the request is sent.
    fn validate_environment(
        &self,
        headers: Headers,
        _api_key: Option<&str>,
        _model: &str,
        litellm_params: &LitellmParams,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if has_header(&headers, "authorization") {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::GcpAccessToken {
                config: Box::new(VertexConfig::from_params(&litellm_params.vertex)),
            },
        })
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        merge_beta_headers(headers, vertex_feature_betas(request))
    }

    /// Vertex addresses the model in the URL and rejects it in the body.
    fn wire_body(&self, mut body: Value) -> Value {
        if let Value::Object(fields) = &mut body {
            fields.remove("model");
        }
        body
    }
}

/// Python's `create_vertex_url` plus `_check_custom_proxy` for the Claude partner: the
/// publisher path under the location host, or under a custom base that has no path of its
/// own, else the custom base with the predict verb appended. A custom base also gets
/// `alt=sse` when streaming.
pub fn complete_vertex_anthropic_url(
    api_base: Option<&str>,
    project: &str,
    location: &str,
    model: &str,
    stream: bool,
) -> Result<String, Error> {
    let verb = if stream {
        STREAM_RAW_PREDICT
    } else {
        RAW_PREDICT
    };
    let path = format!(
        "/v1/projects/{project}/locations/{location}/publishers/anthropic/models/{model}:{verb}"
    );
    let custom = api_base.map(str::trim).filter(|base| !base.is_empty());
    let Some(custom) = custom else {
        return Ok(format!("{}{path}", get_vertex_base_url(location)?));
    };
    let custom = custom.trim_end_matches('/');
    let authority_and_path = custom.split_once("://").map_or(custom, |(_, rest)| rest);
    let base_path = authority_and_path
        .find('/')
        .map_or("", |index| &authority_and_path[index..]);
    let url = match base_path {
        "" => format!("{custom}{path}"),
        "/v1" | "/v1beta1" => format!("{custom}{}", path.trim_start_matches("/v1")),
        _ => format!("{custom}:{verb}"),
    };
    if !stream {
        return Ok(url);
    }
    Ok(match url.contains('?') {
        true => format!("{url}&alt=sse"),
        false => format!("{url}?alt=sse"),
    })
}

/// Python's `sanitize_vertex_anthropic_output_params`: `effort` only reaches models whose
/// capabilities accept it, a malformed `output_config` is dropped, and an emptied one is omitted.
fn sanitize_output_config(
    config: Recognized<OutputConfig>,
    accepts_effort: bool,
) -> Option<Recognized<OutputConfig>> {
    let Recognized::Known(config) = config else {
        return None;
    };
    let sanitized = OutputConfig {
        effort: config.effort.filter(|_| accepts_effort),
        ..config
    };
    (!sanitized.is_empty()).then_some(Recognized::Known(sanitized))
}

fn has_web_search_tool(tools: Option<&[Recognized<MessagesTool>]>) -> bool {
    tools.unwrap_or_default().iter().any(|tool| match tool {
        Recognized::Unrecognized(value) => value
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind.starts_with(WEB_SEARCH_TOOL_PREFIX)),
        Recognized::Known(_) => false,
    })
}

/// Python's Vertex `validate_anthropic_messages_environment` beta set: the shared feature
/// betas under Vertex's policy, plus the web search and safeguards betas Vertex needs spelled
/// out.
fn vertex_feature_betas(request: &MessagesRequest) -> BetaSet {
    let params = &request.params;
    provider_feature_betas(request, BetaProvider::VertexAi).union(
        [
            has_web_search_tool(params.tools.as_deref())
                .then_some(AnthropicBeta::WebSearch20250305),
            params
                .extra
                .get("safeguards")
                .is_some_and(|value| !value.is_null())
                .then_some(AnthropicBeta::DangerousToolUse20260903),
        ]
        .into_iter()
        .flatten()
        .collect(),
    )
}

#[cfg(test)]
mod tests {
    use litellm_auth_gcp::VertexParams;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::base_llm::messages::context::{MessagesModelCapabilities, ThinkingContext};

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn params(project: Option<&str>, location: Option<&str>) -> LitellmParams {
        LitellmParams {
            vertex: VertexParams {
                vertex_project: project.map(Into::into),
                vertex_location: location.map(Into::into),
                ..VertexParams::default()
            },
            ..LitellmParams::default()
        }
    }

    fn url(
        api_base: Option<&str>,
        litellm_params: &LitellmParams,
        stream: bool,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        VERTEX_ANTHROPIC_MESSAGES_CONFIG.get_complete_url(
            api_base,
            "claude-sonnet-4-5@20250929",
            litellm_params,
            stream,
            env,
        )
    }

    const PATH: &str = "/v1/projects/proj/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929";

    #[rstest]
    #[case::regional(
        None,
        false,
        "https://us-east5-aiplatform.googleapis.com",
        ":rawPredict"
    )]
    #[case::streaming(
        None,
        true,
        "https://us-east5-aiplatform.googleapis.com",
        ":streamRawPredict"
    )]
    #[case::custom_host_gets_the_default_path(
        Some("https://proxy.example/"),
        false,
        "https://proxy.example",
        ":rawPredict"
    )]
    #[case::custom_host_streaming_adds_alt_sse(
        Some("https://proxy.example"),
        true,
        "https://proxy.example",
        ":streamRawPredict?alt=sse"
    )]
    #[case::custom_versioned_base_is_grafted(
        Some("https://proxy.example/v1"),
        false,
        "https://proxy.example",
        ":rawPredict"
    )]
    fn the_url_is_the_publisher_path_under_the_host(
        #[case] api_base: Option<&str>,
        #[case] stream: bool,
        #[case] host: &str,
        #[case] suffix: &str,
    ) {
        assert_eq!(
            url(
                api_base,
                &params(Some("proj"), Some("us-east5")),
                stream,
                &no_env
            )
            .unwrap(),
            format!("{host}{PATH}{suffix}")
        );
    }

    #[rstest]
    #[case::not_streaming(false, "https://proxy.example/custom/path:rawPredict")]
    #[case::streaming(true, "https://proxy.example/custom/path:streamRawPredict?alt=sse")]
    fn a_custom_base_with_its_own_path_only_gets_the_verb(
        #[case] stream: bool,
        #[case] expected: &str,
    ) {
        assert_eq!(
            url(
                Some("https://proxy.example/custom/path"),
                &params(Some("proj"), Some("us-east5")),
                stream,
                &no_env
            )
            .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::global(
        "global",
        "https://aiplatform.googleapis.com/v1/projects/proj/locations/global/"
    )]
    #[case::geography(
        "eu",
        "https://aiplatform.eu.rep.googleapis.com/v1/projects/proj/locations/eu/"
    )]
    fn the_host_follows_the_location_kind(#[case] location: &str, #[case] prefix: &str) {
        assert!(
            url(None, &params(Some("proj"), Some(location)), false, &no_env)
                .unwrap()
                .starts_with(prefix)
        );
    }

    #[test]
    fn a_location_that_could_name_another_host_is_rejected() {
        assert!(matches!(
            url(
                None,
                &params(Some("proj"), Some("attacker.example/")),
                false,
                &no_env
            ),
            Err(Error::Auth(_))
        ));
    }

    #[rstest]
    #[case::param_wins(
        params(Some("from-param"), None),
        &[("VERTEXAI_PROJECT", "from-env"), ("VERTEXAI_LOCATION", "europe-west4")],
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/from-param/locations/europe-west4/"
    )]
    #[case::legacy_spelling(
        LitellmParams {
            vertex: VertexParams {
                vertex_ai_project: Some("legacy".into()),
                vertex_ai_location: Some("us-east5".into()),
                ..VertexParams::default()
            },
            ..LitellmParams::default()
        },
        &[],
        "https://us-east5-aiplatform.googleapis.com/v1/projects/legacy/locations/us-east5/"
    )]
    #[case::env_then_default_location(
        LitellmParams::default(),
        &[("VERTEXAI_PROJECT", "from-env")],
        "https://us-central1-aiplatform.googleapis.com/v1/projects/from-env/locations/us-central1/"
    )]
    #[case::project_from_the_credentials(
        LitellmParams {
            vertex: VertexParams {
                vertex_credentials: Some(r#"{"type": "service_account", "project_id": "from-creds"}"#.into()),
                ..VertexParams::default()
            },
            ..LitellmParams::default()
        },
        &[("VERTEX_LOCATION", "us-east5")],
        "https://us-east5-aiplatform.googleapis.com/v1/projects/from-creds/locations/us-east5/"
    )]
    fn project_and_location_follow_the_python_precedence(
        #[case] litellm_params: LitellmParams,
        #[case] env: &[(&str, &str)],
        #[case] prefix: &str,
    ) {
        let lookup = |name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        };
        assert!(
            url(None, &litellm_params, false, &lookup)
                .unwrap()
                .starts_with(prefix)
        );
    }

    #[test]
    fn without_a_project_anywhere_the_call_fails_before_sending() {
        assert!(matches!(
            url(None, &LitellmParams::default(), false, &no_env),
            Err(Error::Auth(_))
        ));
    }

    fn validated(forwarded: &[(&str, &str)]) -> ValidatedEnvironment {
        VERTEX_ANTHROPIC_MESSAGES_CONFIG
            .validate_environment(
                forwarded
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.to_string()))
                    .collect(),
                Some("ignored-api-key"),
                "claude-sonnet-4-5",
                &params(Some("proj"), None),
                &no_env,
            )
            .unwrap()
    }

    #[test]
    fn the_credential_is_a_google_access_token_for_the_configured_project() {
        assert!(matches!(
            validated(&[]).auth,
            AuthScheme::GcpAccessToken { config } if config.project_id() == Some("proj")
        ));
    }

    #[test]
    fn a_forwarded_bearer_is_sent_as_is() {
        assert!(matches!(
            validated(&[("Authorization", "Bearer caller")]).auth,
            AuthScheme::Forwarded
        ));
    }

    #[test]
    fn default_headers_carry_no_anthropic_version() {
        assert_eq!(
            VERTEX_ANTHROPIC_MESSAGES_CONFIG.default_headers(),
            &[("content-type", "application/json")]
        );
    }

    fn transformed(value: Value, context: &MessagesTransformContext) -> Value {
        let request: MessagesRequest = serde_json::from_value(value).unwrap();
        serde_json::to_value(
            VERTEX_ANTHROPIC_MESSAGES_CONFIG
                .transform_anthropic_messages_request(request, context)
                .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn the_body_carries_the_vertex_version_and_the_wire_drops_the_model() {
        let body = transformed(
            json!({
                "model": "claude-sonnet-4-5@20250929",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}]
            }),
            &MessagesTransformContext::default(),
        );
        assert_eq!(body["anthropic_version"], json!("vertex-2023-10-16"));
        assert_eq!(body["model"], json!("claude-sonnet-4-5@20250929"));
        let wire = VERTEX_ANTHROPIC_MESSAGES_CONFIG.wire_body(body);
        assert_eq!(wire.get("model"), None);
        assert_eq!(wire["anthropic_version"], json!("vertex-2023-10-16"));
    }

    #[rstest]
    fn leading_system_roles_are_hoisted_billing_blocks_drop_and_scope_is_stripped() {
        let body = transformed(
            json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "system": [
                    {"type": "text", "text": "x-anthropic-billing-header: cc_version=1"},
                    {"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral", "scope": "global"}}
                ],
                "messages": [
                    {"role": "system", "content": [
                        {"type": "text", "text": "x-anthropic-billing-header: cc_version=2"},
                        {"type": "text", "text": "leading system"}
                    ]},
                    {"role": "user", "content": [
                        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "global"}}
                    ]}
                ]
            }),
            &MessagesTransformContext::default(),
        );
        assert_eq!(
            body["system"],
            json!([
                {"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "leading system"}
            ])
        );
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
            ]}])
        );
    }

    #[rstest]
    #[case::converted_in_place_without_the_capability(false, "user")]
    #[case::kept_in_place_with_the_capability(true, "system")]
    fn a_later_system_turn_never_moves_into_the_prompt_prefix(
        #[case] supports_mid_conversation_system: bool,
        #[case] role: &str,
    ) {
        let body = transformed(
            json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "system": "be terse",
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                    {"role": "system", "content": "reminder"},
                    {"role": "user", "content": "c"}
                ]
            }),
            &MessagesTransformContext {
                thinking: ThinkingContext {
                    capabilities: MessagesModelCapabilities {
                        supports_mid_conversation_system,
                        ..MessagesModelCapabilities::default()
                    },
                    ..ThinkingContext::default()
                },
                drop_params: false,
            },
        );
        assert_eq!(body["system"], json!("be terse"));
        assert_eq!(body["messages"][2]["role"], json!(role));
        assert_eq!(body["messages"].as_array().unwrap().len(), 4);
    }

    fn with_output_config() -> MessagesTransformContext {
        MessagesTransformContext {
            thinking: ThinkingContext {
                capabilities: MessagesModelCapabilities {
                    supports_output_config: true,
                    ..MessagesModelCapabilities::default()
                },
                ..ThinkingContext::default()
            },
            drop_params: false,
        }
    }

    #[rstest]
    #[case::effort_kept_when_the_model_accepts_it(
        with_output_config(),
        json!({"effort": "high", "format": {"type": "json_schema"}}),
        Some(json!({"effort": "high", "format": {"type": "json_schema"}}))
    )]
    #[case::effort_dropped_otherwise(
        MessagesTransformContext::default(),
        json!({"effort": "high", "format": {"type": "json_schema"}}),
        Some(json!({"format": {"type": "json_schema"}}))
    )]
    #[case::emptied_config_is_omitted(
        MessagesTransformContext::default(),
        json!({"effort": "high"}),
        None
    )]
    #[case::malformed_config_is_dropped(
        with_output_config(),
        json!("not an object"),
        None
    )]
    fn output_config_is_sanitized_for_vertex(
        #[case] context: MessagesTransformContext,
        #[case] output_config: Value,
        #[case] expected: Option<Value>,
    ) {
        let body = transformed(
            json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "output_config": output_config,
                "messages": [{"role": "user", "content": "hi"}]
            }),
            &context,
        );
        assert_eq!(body.get("output_config").cloned(), expected);
    }

    fn betas(fields: Value) -> Vec<String> {
        let request: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap();
        let request: MessagesRequest = serde_json::from_value({
            let mut base = serde_json::to_value(request).unwrap();
            base.as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            base
        })
        .unwrap();
        VERTEX_ANTHROPIC_MESSAGES_CONFIG
            .request_headers(Vec::new(), &request)
            .into_iter()
            .filter(|(name, _)| name == "anthropic-beta")
            .flat_map(|(_, value)| value.split(',').map(str::to_string).collect::<Vec<_>>())
            .collect()
    }

    #[rstest]
    #[case::web_search_tool(
        json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
        &["web-search-2025-03-05"]
    )]
    #[case::tool_search_uses_the_vertex_header(
        json!({"tools": [{"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}]}),
        &["tool-search-tool-2025-10-19"]
    )]
    #[case::safeguards(json!({"safeguards": {"mode": "strict"}}), &["dangerous-tool-use-2026-09-03"])]
    #[case::context_management_edits(
        json!({"context_management": {"edits": [
            {"type": "compact_20260112"},
            {"type": "clear_tool_uses_20250919"}
        ]}}),
        &["compact-2026-01-12", "context-management-2025-06-27"]
    )]
    #[case::advisor_tool_is_not_a_vertex_beta(
        json!({"tools": [{"type": "advisor_20260301", "name": "advisor"}]}),
        &[]
    )]
    #[case::fast_mode_is_not_a_vertex_beta(json!({"speed": "fast"}), &[])]
    fn beta_headers_follow_the_vertex_policy(#[case] fields: Value, #[case] expected: &[&str]) {
        assert_eq!(betas(fields), expected);
    }
}
