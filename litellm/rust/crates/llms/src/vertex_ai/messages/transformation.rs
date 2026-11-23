//! Claude on Vertex AI's `/v1/messages`: the Anthropic request reshaped for Vertex (version in
//! the body, model in the URL, Vertex's betas and labels), sent with a Google access token to the
//! project and region the call resolves.

use crate::vertex_ai::common_utils::vertex_region;
use litellm_auth::VertexParams;
use litellm_auth_gcp::{
    VertexConfig, get_vertex_ai_project, get_vertex_ai_project_from_credentials, secret_names,
    vertex::vertex_location,
};
use litellm_host::interceptors::RetriedRequest;
use litellm_http::request::{header_value, without_headers};
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest, MessagesTool, OutputConfig},
    providers::anthropic::{AnthropicBeta, BetaProvider, BetaSet, VERSION_HEADER},
    recognized::Recognized,
};
use litellm_router_types::LitellmParams;
use serde_json::{Value, json};

use crate::{
    Error,
    anthropic::{
        beta_gated_fields::strip_beta_gated_request,
        beta_headers::BetaPolicy,
        common_utils::{existing_betas, merge_beta_headers, supports_effort_param},
        messages::transformation::{
            ANTHROPIC_MESSAGES_CONFIG, drop_sampling_params_removed_by_version,
            provider_feature_betas,
        },
    },
    base_llm::{
        auth::{AuthScheme, Headers, LoggedIn, ValidatedEnvironment},
        messages::{
            context::{MessagesShapeOptions, MessagesTransformContext},
            normalization::{normalize_system_role_messages, strip_cache_control_scope},
            tools::ToolCustomField,
            transformation::{MessagesAuth, MessagesDialect, MessagesEndpoint, MessagesReply},
        },
    },
    vertex_ai::{
        common_utils::{get_vertex_base_url, graft_default_vertex_path, join_api_base},
        request_labels::with_partner_labels_header,
    },
};
use litellm_llms_types::providers::vertex_ai::{
    ANTHROPIC_VERSION, RAW_PREDICT, STREAM_RAW_PREDICT,
};
use litellm_python_compat::urllib::urlsplit;

const WEB_SEARCH_TOOL_PREFIX: &str = "web_search";
const CONTENT_TYPE_HEADER: &str = "content-type";
const JSON_CONTENT_TYPE: &str = "application/json";
const VERSIONS_SERVED_AS_VERTEX_DEFAULT: [&str; 3] = ["", "2023-01-01", "2023-06-01"];
const VERSIONS_VERTEX_ACCEPTS_IN_BODY: [&str; 2] = [ANTHROPIC_VERSION, "2026-10-21"];

/// Claude on Vertex AI, Python's `VertexAIPartnerModelsAnthropicMessagesConfig`: the Anthropic
/// payload addressed to a project and location, authenticated with a Google access token.
pub struct VertexAiMessagesConfig;

pub const VERTEX_AI_MESSAGES_CONFIG: VertexAiMessagesConfig = VertexAiMessagesConfig;

impl MessagesEndpoint for VertexAiMessagesConfig {
    fn select_endpoint(
        &self,
        litellm_params: LitellmParams,
        model: &str,
        logged_in: LoggedIn,
        supported_regions: &[String],
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> LitellmParams {
        vertex_connection_params(
            litellm_params,
            model,
            logged_in.project,
            supported_regions,
            env_lookup,
        )
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
            .ok_or_else(|| Error::Auth(VertexParams::PROJECT.missing("Vertex AI")))?;
        let location = vertex_region(vertex_location(&litellm_params.vertex, env_lookup), &[]);
        complete_vertex_anthropic_url(api_base, &project, &location, model, stream)
    }
}

impl MessagesAuth for VertexAiMessagesConfig {
    fn secret_names(&self) -> &'static [&'static str] {
        secret_names()
    }

    /// Python's `_finalize_vertex_messages_environment`: the call always logs in to Google as
    /// `VertexBase.load_auth` does, from `vertex_credentials`, `VERTEXAI_CREDENTIALS`, then
    /// application default credentials, and the token is set under the exact `Authorization` key,
    /// replacing a caller's header of that spelling only. The exact `content-type` key is set to
    /// JSON the same way, and the proxy's request labels replace the caller's
    /// `X-Vertex-AI-Labels` once label fields are configured.
    fn validate_environment(
        &self,
        headers: Headers,
        _api_key: Option<&str>,
        _model: &str,
        _optional_params: &MessagesOptionalParams,
        litellm_params: &LitellmParams,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        Ok(ValidatedEnvironment {
            headers: with_partner_labels_header(
                with_exact_header(headers, CONTENT_TYPE_HEADER, JSON_CONTENT_TYPE),
                &litellm_params.extra,
            ),
            auth: AuthScheme::GcpAccessToken {
                params: Box::new(litellm_params.vertex.clone()),
            },
        })
    }
}

impl MessagesDialect for VertexAiMessagesConfig {
    fn shape_request(
        &self,
        request: MessagesRequest,
        options: MessagesShapeOptions,
    ) -> Result<MessagesRequest, Error> {
        ANTHROPIC_MESSAGES_CONFIG.shape_request(request, options)
    }

    fn tool_custom_field(&self, model: &str) -> ToolCustomField {
        ANTHROPIC_MESSAGES_CONFIG.tool_custom_field(model)
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        let request = ANTHROPIC_MESSAGES_CONFIG.transform_anthropic_messages_request(
            request,
            &context.warning_dropped_disabled_thinking(),
        )?;
        Ok(sanitize_vertex_anthropic_output_params(
            request,
            supports_effort_param(&context.thinking.capabilities),
        ))
    }

    fn request_body(
        &self,
        request: MessagesRequest,
        headers: &Headers,
        _url: &str,
        drop_params: bool,
    ) -> Result<MessagesRequest, Error> {
        let request = with_vertex_anthropic_version(request, headers, drop_params)?;
        fold_output_format_into_output_config(request)
    }

    /// Python's Vertex beta set merged into the caller's betas, then the handler's
    /// `update_headers_with_filtered_beta` over the `vertex_ai` row.
    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        let headers = without_headers(headers, &[VERSION_HEADER]);
        BetaPolicy::Filter(BetaProvider::VertexAi)
            .apply(merge_beta_headers(headers, vertex_feature_betas(request)))
    }

    /// Python's transform strips the thinking and system fields the final, filtered header does
    /// not unlock, then normalizes system roles and drops `cache_control.scope`, all after the
    /// betas were chosen from the request as sent.
    fn finalize_request(
        &self,
        request: MessagesRequest,
        headers: Headers,
        context: &MessagesTransformContext,
    ) -> Result<(MessagesRequest, Headers), Error> {
        let request = strip_beta_gated_request(
            request,
            &existing_betas(&headers),
            context.drops_unsupported_params,
            BetaProvider::VertexAi,
        );
        let request = normalize_system_role_messages(
            request,
            context
                .thinking
                .capabilities
                .supports_mid_conversation_system,
            context.drops_unsupported_params,
            BetaProvider::VertexAi,
        );
        Ok((strip_cache_control_scope(request), headers))
    }

    /// Vertex addresses the model in the URL and rejects it in the body.
    fn wire_body(&self, body: Value) -> Value {
        without_model(body)
    }
}

impl MessagesReply for VertexAiMessagesConfig {
    fn retry_after_http_error(
        &self,
        status: u16,
        error_text: &str,
        request_body: &Value,
        context: &MessagesTransformContext,
    ) -> Option<RetriedRequest> {
        ANTHROPIC_MESSAGES_CONFIG.retry_after_http_error(status, error_text, request_body, context)
    }
}

/// The connection params Python's `get_complete_vertex_url` reads: the project the login
/// resolved, and the region a model the model map lists `supported_regions` for is routed to.
pub fn vertex_connection_params(
    params: LitellmParams,
    model: &str,
    logged_in_project: Option<String>,
    supported_regions: &[String],
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> LitellmParams {
    let vertex_location = match supported_regions.is_empty() {
        true => params.vertex.vertex_location.clone(),
        false => {
            let configured = vertex_location(&params.vertex, env_lookup);
            let routed = vertex_region(configured.clone(), supported_regions);
            if let Some(configured) = configured.filter(|configured| *configured != routed) {
                warn_unsupported_region(model, &configured, supported_regions, &routed);
            }
            Some(routed)
        }
    };
    LitellmParams {
        vertex: VertexParams {
            vertex_project: logged_in_project.or(params.vertex.vertex_project.clone()),
            vertex_location,
            ..params.vertex.clone()
        },
        ..params
    }
}

/// `get_vertex_region`'s warning when it overrides a configured location the model does not
/// support.
fn warn_unsupported_region(model: &str, configured: &str, supported: &[String], routed: &str) {
    tracing::warn!(
        "Vertex AI model {model} is not served in {configured}, so the call goes to {routed} (served in {}).",
        supported.join(", ")
    );
}

/// Python's `get_complete_vertex_url` for the Claude partner: the publisher path joined to the
/// location host, or `_check_custom_proxy` over a custom base. A base without a path gets the
/// publisher path, a base that is only an API version gets it grafted after that version, and
/// any other base gets the predict verb appended. A custom base also gets `alt=sse` in its
/// query when streaming.
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
    let route = format!(
        "v1/projects/{project}/locations/{location}/publishers/anthropic/models/{model}:{verb}"
    );
    let Some(custom) = api_base.filter(|base| !base.is_empty()) else {
        return Ok(join_api_base(&get_vertex_base_url(location)?, &route));
    };
    let default_url = join_api_base(custom, &route);
    let base_path = urlsplit(custom).path;
    // The `/v1` / `/v1beta1` branch deliberately keeps Python's doubled version segment
    // (`/v1/v1/projects/...`) so existing gateway setups keep working.
    let url = if base_path.is_empty() || base_path == "/" {
        join_api_base(custom, urlsplit(&default_url).path)
    } else if matches!(base_path.trim_end_matches('/'), "/v1" | "/v1beta1")
        && urlsplit(&default_url).path.contains("/projects/")
    {
        graft_default_vertex_path(custom, &default_url)
    } else {
        format!("{}:{verb}", custom.trim_end_matches('/'))
    };
    if !stream {
        return Ok(url);
    }
    let parsed = urlsplit(&url);
    let query = match parsed.query.is_empty() {
        true => "alt=sse".to_string(),
        false => format!("{}&alt=sse", parsed.query),
    };
    Ok(parsed.with(parsed.path, &query))
}

/// Python's `sanitize_vertex_anthropic_output_params`: `effort` only reaches models whose
/// capabilities accept it, a malformed `output_config` is dropped, and an emptied one is omitted.
fn sanitize_vertex_anthropic_output_params(
    request: MessagesRequest,
    accepts_effort: bool,
) -> MessagesRequest {
    let output_config = match request.params.output_config {
        Some(Recognized::Known(config)) => Some(OutputConfig {
            effort: config.effort.filter(|_| accepts_effort),
            ..config
        })
        .filter(|config| !config.is_empty())
        .map(Recognized::Known),
        Some(Recognized::Unrecognized(_)) | None => None,
    };
    MessagesRequest {
        params: MessagesOptionalParams {
            output_config,
            ..request.params
        },
        ..request
    }
}

/// Python's `headers[name] = value`: only a key spelled exactly `name` is replaced.
fn with_exact_header(headers: Headers, name: &str, value: &str) -> Headers {
    headers
        .into_iter()
        .filter(|(present, _)| present != name)
        .chain([(name.to_string(), value.to_string())])
        .collect()
}

/// Python's `_fold_output_format_into_output_config`, run after the version check as Python's
/// transform does: an object `output_format` becomes `output_config.format`, winning over one
/// already there and replacing a malformed `output_config`. Any other value is refused, since
/// the shared transform already dropped it under `drop_params`.
fn fold_output_format_into_output_config(
    mut request: MessagesRequest,
) -> Result<MessagesRequest, Error> {
    let output_format = match request.params.output_format.take().map(OutputFormat::from) {
        None => return Ok(request),
        Some(OutputFormat::Object(format)) => format,
        Some(OutputFormat::Other { python_type }) => {
            return Err(Error::InvalidRequest(
                crate::ErrorDetail::NonObjectVertexOutputFormat {
                    model: request.model,
                    actual: python_type,
                },
            ));
        }
    };
    let config = match request.params.output_config {
        Some(Recognized::Known(config)) => config,
        Some(Recognized::Unrecognized(_)) | None => OutputConfig::default(),
    };
    Ok(MessagesRequest {
        params: MessagesOptionalParams {
            output_config: Some(Recognized::Known(OutputConfig {
                format: Some(Value::Object(output_format)),
                ..config
            })),
            ..request.params
        },
        ..request
    })
}

/// A caller's `output_format`: an object Vertex folds into `output_config.format`, or a value of
/// another type, which Vertex refuses by its Python type name.
#[derive(Clone, Debug, PartialEq)]
enum OutputFormat {
    Object(serde_json::Map<String, Value>),
    Other { python_type: &'static str },
}

impl From<Value> for OutputFormat {
    fn from(value: Value) -> Self {
        match value {
            Value::Object(format) => Self::Object(format),
            other => Self::Other {
                python_type: litellm_python_compat::json::type_name(&other),
            },
        }
    }
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
/// betas under Vertex's policy, less structured outputs, which Python never derives for Vertex,
/// plus the web search beta Vertex needs spelled out.
fn vertex_feature_betas(request: &MessagesRequest) -> BetaSet {
    let params = &request.params;
    provider_feature_betas(request, BetaProvider::VertexAi)
        .into_iter()
        .filter(|beta| *beta != AnthropicBeta::StructuredOutputs20251113)
        .chain(
            has_web_search_tool(params.tools.as_deref())
                .then_some(AnthropicBeta::WebSearch20250305),
        )
        .collect()
}

/// The version Vertex reads from the body, carried there as `anthropic_version`.
fn with_vertex_anthropic_version(
    request: MessagesRequest,
    headers: &Headers,
    drop_params: bool,
) -> Result<MessagesRequest, Error> {
    let version = vertex_version(
        &request.model,
        header_value(headers, VERSION_HEADER),
        drop_params,
    )?;
    let request = drop_sampling_params_removed_by_version(request, version, drop_params)?;
    Ok(MessagesRequest {
        params: MessagesOptionalParams {
            extra: request
                .params
                .extra
                .into_iter()
                .chain([("anthropic_version".to_string(), json!(version))])
                .collect(),
            ..request.params
        },
        ..request
    })
}

fn without_model(mut body: Value) -> Value {
    if let Value::Object(fields) = &mut body {
        fields.remove("model");
    }
    body
}

fn vertex_version<'a>(
    model: &str,
    requested: Option<&'a str>,
    drop_params: bool,
) -> Result<&'a str, Error> {
    match requested.map(str::trim).unwrap_or_default() {
        version if VERSIONS_SERVED_AS_VERTEX_DEFAULT.contains(&version) => Ok(ANTHROPIC_VERSION),
        version if VERSIONS_VERTEX_ACCEPTS_IN_BODY.contains(&version) => Ok(version),
        _ if drop_params => Ok(ANTHROPIC_VERSION),
        version => Err(Error::InvalidRequest(
            crate::ErrorDetail::UnsupportedVertexAnthropicVersion {
                model: model.into(),
                value: version.into(),
            },
        )),
    }
}

#[cfg(test)]
mod tests {
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
        VERTEX_AI_MESSAGES_CONFIG.get_complete_url(
            api_base,
            "claude-sonnet-4-5@20250929",
            litellm_params,
            stream,
            env,
        )
    }

    const PATH: &str =
        "/projects/proj/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929";
    const VERTEX_URL: &str = "https://us-east5-aiplatform.googleapis.com/v1/projects/proj/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict";

    #[rstest]
    #[case::regional(
        None,
        false,
        "https://us-east5-aiplatform.googleapis.com/v1{P}:rawPredict"
    )]
    #[case::regional_stream(
        None,
        true,
        "https://us-east5-aiplatform.googleapis.com/v1{P}:streamRawPredict"
    )]
    #[case::empty_base_is_no_base(
        Some(""),
        true,
        "https://us-east5-aiplatform.googleapis.com/v1{P}:streamRawPredict"
    )]
    #[case::host(Some("https://h/"), false, "https://h/v1{P}:rawPredict")]
    #[case::host_stream(Some("https://h"), true, "https://h/v1{P}:streamRawPredict?alt=sse")]
    #[case::version_is_doubled(Some("https://h/v1"), false, "https://h/v1/v1{P}:rawPredict")]
    #[case::version_with_slash(
        Some("https://h:8080/v1/"),
        true,
        "https://h:8080/v1/v1{P}:streamRawPredict?alt=sse"
    )]
    #[case::beta_version(Some("https://h/v1beta1"), false, "https://h/v1beta1/v1{P}:rawPredict")]
    #[case::version_with_query(
        Some("https://h/v1?k=1"),
        true,
        "https://h/v1?k=1:streamRawPredict&alt=sse"
    )]
    #[case::host_with_query(Some("https://h?k=1"), false, "https://h?k=1/")]
    #[case::host_slash_query(Some("https://h/?k=1"), true, "https://h?k=1/&alt=sse")]
    #[case::own_path(
        Some("https://h/custom/path/"),
        false,
        "https://h/custom/path:rawPredict"
    )]
    #[case::own_path_stream(
        Some("https://h/custom/path"),
        true,
        "https://h/custom/path:streamRawPredict?alt=sse"
    )]
    #[case::own_path_query(Some("https://h/a?b"), true, "https://h/a?b:streamRawPredict&alt=sse")]
    #[case::fragment_stream(
        Some("https://h/a?b#c"),
        true,
        "https://h/a?b&alt=sse#c:streamRawPredict"
    )]
    // Approved difference: Python's urlparse reads `;x` as path params; the builder keeps it in
    // the path, so the base is no longer a bare version and gets the verb.
    #[case::version_params(
        Some("https://h/v1;x"),
        true,
        "https://h/v1;x:streamRawPredict?alt=sse"
    )]
    #[case::upper_case_version(Some("https://h/V1"), false, "https://h/V1:rawPredict")]
    fn the_url_matches_python(
        #[case] api_base: Option<&str>,
        #[case] stream: bool,
        #[case] expected: &str,
    ) {
        assert_eq!(
            url(
                api_base,
                &params(Some("proj"), Some("us-east5")),
                stream,
                &no_env
            )
            .unwrap(),
            expected.replace("{P}", PATH)
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

    #[rstest]
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

    #[rstest]
    #[case::login_project_and_listed_region(
        Some("logged-in"),
        &["us-east5"],
        &[("VERTEXAI_LOCATION", "europe-west4")],
        "https://us-east5-aiplatform.googleapis.com/v1/projects/logged-in/locations/us-east5/"
    )]
    #[case::supported_location_kept(
        Some("logged-in"),
        &["us-east5", "europe-west4"],
        &[("VERTEXAI_LOCATION", "europe-west4")],
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/logged-in/locations/europe-west4/"
    )]
    #[case::no_regions_keep_the_location(
        None,
        &[],
        &[("VERTEXAI_LOCATION", "europe-west4")],
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/proj/locations/europe-west4/"
    )]
    fn the_login_project_and_supported_regions_address_the_call(
        #[case] logged_in: Option<&str>,
        #[case] supported: &[&str],
        #[case] env: &[(&str, &str)],
        #[case] prefix: &str,
    ) {
        let lookup = |name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        };
        let supported: Vec<String> = supported.iter().map(|region| region.to_string()).collect();
        let connection = vertex_connection_params(
            params(Some("proj"), None),
            "claude-sonnet-5-5@default",
            logged_in.map(str::to_string),
            &supported,
            &lookup,
        );
        assert!(
            url(None, &connection, false, &lookup)
                .unwrap()
                .starts_with(prefix)
        );
    }

    #[rstest]
    fn without_a_project_anywhere_the_call_fails_before_sending() {
        let error = url(None, &LitellmParams::default(), false, &no_env).unwrap_err();

        assert!(matches!(
            error,
            Error::Auth(litellm_auth::Error::MissingParam {
                provider: "Vertex AI",
                spec: &VertexParams::PROJECT,
            })
        ));
        assert_eq!(
            error.to_string(),
            "Missing Vertex AI project - pass vertex_project or vertex_ai_project, set litellm.vertex_project, or set VERTEXAI_PROJECT"
        );
    }

    fn validated(forwarded: &[(&str, &str)]) -> ValidatedEnvironment {
        VERTEX_AI_MESSAGES_CONFIG
            .validate_environment(
                forwarded
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.to_string()))
                    .collect(),
                Some("ignored-api-key"),
                "claude-sonnet-4-5",
                &MessagesOptionalParams::default(),
                &params(Some("proj"), None),
                &no_env,
            )
            .unwrap()
    }

    #[rstest]
    #[case::configured(
        json!(["user_api_key_team_id"]),
        vec![("X-Vertex-AI-Labels".to_string(), base64::Engine::encode(&base64::engine::general_purpose::STANDARD, json!({"user_api_key_team_id": "t1"}).to_string()))]
    )]
    #[case::unset(json!(null), vec![("x-vertex-ai-labels".to_string(), "forged".to_string())])]
    fn configured_label_fields_replace_the_callers_labels_header(
        #[case] fields: Value,
        #[case] expected: Headers,
    ) {
        let litellm_params = LitellmParams {
            extra: [
                ("vertex_request_label_fields".to_string(), fields),
                (
                    "litellm_metadata".to_string(),
                    json!({"user_api_key_team_id": "t1"}),
                ),
            ]
            .into_iter()
            .collect(),
            ..params(Some("proj"), None)
        };
        let headers = VERTEX_AI_MESSAGES_CONFIG
            .validate_environment(
                vec![("x-vertex-ai-labels".to_string(), "forged".to_string())],
                None,
                "claude-sonnet-4-5",
                &MessagesOptionalParams::default(),
                &litellm_params,
                &no_env,
            )
            .unwrap()
            .headers;
        let labels: Headers = headers
            .into_iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("x-vertex-ai-labels"))
            .collect();
        assert_eq!(labels, expected);
    }

    #[rstest]
    #[case::no_caller_credential(&[])]
    #[case::caller_bearer(&[("Authorization", "Bearer caller")])]
    #[case::caller_lowercase_bearer(&[("authorization", "Bearer caller")])]
    fn the_credential_is_always_a_google_access_token_for_the_configured_project(
        #[case] forwarded: &[(&str, &str)],
    ) {
        assert!(matches!(
            validated(forwarded).auth,
            AuthScheme::GcpAccessToken { params } if params.project().as_deref() == Some("proj")
        ));
    }

    #[rstest]
    #[case::caller_json(&[("content-type", "application/json")], &[("content-type", "application/json")])]
    #[case::caller_text_is_replaced(
        &[("content-type", "text/plain"), ("x-trace", "1")],
        &[("x-trace", "1"), ("content-type", "application/json")]
    )]
    #[case::another_casing_stays_beside_it(
        &[("Content-Type", "text/plain")],
        &[("Content-Type", "text/plain"), ("content-type", "application/json")]
    )]
    #[case::none_sent(&[], &[("content-type", "application/json")])]
    fn the_exact_content_type_key_is_json(
        #[case] forwarded: &[(&str, &str)],
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(
            validated(forwarded).headers,
            expected
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect::<Headers>()
        );
    }

    #[rstest]
    fn default_headers_carry_no_anthropic_version() {
        assert_eq!(
            VERTEX_AI_MESSAGES_CONFIG.default_headers(),
            &[("content-type", "application/json")]
        );
    }

    #[rstest]
    #[case::sdk_spelling("anthropic-version")]
    #[case::any_case("Anthropic-Version")]
    fn a_forwarded_anthropic_version_never_reaches_vertex(#[case] name: &str) {
        let request: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap();
        assert_eq!(
            VERTEX_AI_MESSAGES_CONFIG.request_headers(
                vec![
                    (name.to_string(), "2026-10-21".to_string()),
                    ("x-request-id".to_string(), "req-1".to_string()),
                ],
                &request,
            ),
            vec![("x-request-id".to_string(), "req-1".to_string())]
        );
    }

    fn finalized(
        value: Value,
        headers: &[(&str, &str)],
        context: &MessagesTransformContext,
    ) -> Value {
        let request: MessagesRequest = serde_json::from_value(value).unwrap();
        let transformed = VERTEX_AI_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, context)
            .unwrap();
        let headers = headers
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        let (finalized, _) = VERTEX_AI_MESSAGES_CONFIG
            .finalize_request(transformed, headers, context)
            .unwrap();
        serde_json::to_value(finalized).unwrap()
    }

    fn transformed(value: Value, context: &MessagesTransformContext) -> Value {
        finalized(value, &[], context)
    }

    #[rstest]
    #[case::sdk_2(&[("anthropic-version", "2026-10-21")], false, "2026-10-21")]
    #[case::any_case(&[("Anthropic-Version", "2026-10-21")], false, "2026-10-21")]
    #[case::vertex_name(&[("anthropic-version", "vertex-2023-10-16")], false, "vertex-2023-10-16")]
    #[case::sdk_1(&[("anthropic-version", "2023-06-01")], false, "vertex-2023-10-16")]
    #[case::first_version(&[("anthropic-version", "2023-01-01")], false, "vertex-2023-10-16")]
    #[case::blank(&[("anthropic-version", " ")], false, "vertex-2023-10-16")]
    #[case::no_header(&[], false, "vertex-2023-10-16")]
    #[case::unknown_under_drop_params(&[("anthropic-version", "latest")], true, "vertex-2023-10-16")]
    fn the_requested_version_goes_in_the_body(
        #[case] forwarded: &[(&str, &str)],
        #[case] drop_params: bool,
        #[case] body_version: &str,
    ) {
        let request: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap();
        let headers: Headers = forwarded
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        let body = serde_json::to_value(
            VERTEX_AI_MESSAGES_CONFIG
                .request_body(request, &headers, VERTEX_URL, drop_params)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["anthropic_version"], json!(body_version));
    }

    #[rstest]
    #[case::unreleased("2025-01-01")]
    #[case::alias("latest")]
    fn an_unsupported_version_is_refused_without_drop_params(#[case] requested: &str) {
        let request: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap();
        let headers: Headers = vec![("anthropic-version".to_string(), requested.to_string())];
        let Error::InvalidRequest(detail) = VERTEX_AI_MESSAGES_CONFIG
            .request_body(request, &headers, VERTEX_URL, false)
            .unwrap_err()
        else {
            panic!("expected an invalid request");
        };
        assert_eq!(
            detail.to_string(),
            format!(
                "claude-sonnet-4-5 does not support anthropic-version={requested}. \
                 To drop unsupported params, set `litellm.drop_params = True`."
            )
        );
    }

    fn sampling_request() -> MessagesRequest {
        serde_json::from_value(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "temperature": 1.0,
            "top_p": 0.9,
            "top_k": 40,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap()
    }

    fn sampling_params(body: &MessagesRequest) -> (Option<f64>, Option<f64>, Option<i64>) {
        (
            body.params
                .temperature
                .as_ref()
                .and_then(Recognized::known)
                .copied(),
            body.params
                .top_p
                .as_ref()
                .and_then(Recognized::known)
                .copied(),
            body.params
                .top_k
                .as_ref()
                .and_then(Recognized::known)
                .copied(),
        )
    }

    #[rstest]
    #[case::sdk_2_under_drop_params("2026-10-21", true, (None, None, None))]
    #[case::sdk_1("2023-06-01", false, (Some(1.0), Some(0.9), Some(40)))]
    #[case::sdk_1_under_drop_params("2023-06-01", true, (Some(1.0), Some(0.9), Some(40)))]
    #[case::unknown_version_served_as_the_vertex_default("2027-01-01", true, (Some(1.0), Some(0.9), Some(40)))]
    fn sampling_params_follow_the_version_vertex_applies(
        #[case] requested: &str,
        #[case] drop_params: bool,
        #[case] expected: (Option<f64>, Option<f64>, Option<i64>),
    ) {
        let headers: Headers = vec![("anthropic-version".to_string(), requested.to_string())];
        let body = VERTEX_AI_MESSAGES_CONFIG
            .request_body(sampling_request(), &headers, VERTEX_URL, drop_params)
            .unwrap();
        assert_eq!(sampling_params(&body), expected);
    }

    #[rstest]
    fn sampling_params_are_refused_under_2026_10_21_without_drop_params() {
        let headers: Headers = vec![("anthropic-version".to_string(), "2026-10-21".to_string())];
        let Error::InvalidRequest(detail) = VERTEX_AI_MESSAGES_CONFIG
            .request_body(sampling_request(), &headers, VERTEX_URL, false)
            .unwrap_err()
        else {
            panic!("expected an invalid request");
        };
        assert_eq!(
            detail.to_string(),
            "claude-sonnet-4-5 does not support temperature=1.0. anthropic-version 2026-10-21 \
             removed temperature, top_p and top_k. To drop unsupported params, set \
             `litellm.drop_params = True`."
        );
    }

    fn body_with(fields: Value, headers: &[(&str, &str)]) -> Result<Value, Error> {
        let mut request = json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        });
        request
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let headers: Headers = headers
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        VERTEX_AI_MESSAGES_CONFIG
            .request_body(
                serde_json::from_value(request).unwrap(),
                &headers,
                VERTEX_URL,
                false,
            )
            .map(|body| serde_json::to_value(body).unwrap())
    }

    #[rstest]
    #[case::moved_into_output_config(
        json!({"output_format": {"type": "json_schema"}}),
        Some(json!({"format": {"type": "json_schema"}}))
    )]
    #[case::wins_over_an_existing_format(
        json!({"output_format": {"type": "new"}, "output_config": {"effort": "high", "format": {"type": "old"}}}),
        Some(json!({"effort": "high", "format": {"type": "new"}}))
    )]
    #[case::replaces_a_malformed_config(
        json!({"output_format": {"type": "json_schema"}, "output_config": "not an object"}),
        Some(json!({"format": {"type": "json_schema"}}))
    )]
    #[case::null_is_no_format(json!({"output_format": null, "output_config": {"effort": "low"}}), Some(json!({"effort": "low"})))]
    #[case::absent(json!({}), None)]
    fn output_format_folds_into_output_config(
        #[case] fields: Value,
        #[case] expected: Option<Value>,
    ) {
        let body = body_with(fields, &[]).unwrap();
        assert_eq!(body.get("output_format"), None);
        assert_eq!(body.get("output_config").cloned(), expected);
    }

    #[rstest]
    #[case::string(json!("text"), "str")]
    #[case::list(json!([1]), "list")]
    #[case::integer(json!(1), "int")]
    #[case::float(json!(1.5), "float")]
    #[case::boolean(json!(true), "bool")]
    fn a_non_object_output_format_is_refused_with_pythons_text(
        #[case] output_format: Value,
        #[case] actual: &str,
    ) {
        let Err(Error::InvalidRequest(detail)) =
            body_with(json!({"output_format": output_format}), &[])
        else {
            panic!("expected an invalid request");
        };
        assert_eq!(
            detail.to_string(),
            format!(
                "claude-sonnet-4-5 on vertex_ai needs `output_format` to be a JSON object, got {actual}. \
                 To drop it, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n"
            )
        );
    }

    #[rstest]
    fn an_unsupported_version_is_refused_before_the_output_format() {
        assert!(matches!(
            body_with(
                json!({"output_format": "text"}),
                &[("anthropic-version", "latest")]
            ),
            Err(Error::InvalidRequest(
                crate::ErrorDetail::UnsupportedVertexAnthropicVersion { .. }
            ))
        ));
    }

    #[rstest]
    fn the_body_carries_the_vertex_version_and_the_wire_drops_the_model() {
        let request: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-5@20250929",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let body = serde_json::to_value(
            VERTEX_AI_MESSAGES_CONFIG
                .request_body(
                    VERTEX_AI_MESSAGES_CONFIG
                        .transform_anthropic_messages_request(
                            request,
                            &MessagesTransformContext::default(),
                        )
                        .unwrap(),
                    &Vec::new(),
                    VERTEX_URL,
                    false,
                )
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["model"], json!("claude-sonnet-4-5@20250929"));
        let wire = VERTEX_AI_MESSAGES_CONFIG.wire_body(body);
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
                drops_unsupported_params: false,
                ..MessagesTransformContext::default()
            },
        );
        assert_eq!(
            body["system"],
            json!([{"type": "text", "text": "be terse"}])
        );
        assert_eq!(body["messages"][2]["role"], json!(role));
        assert_eq!(body["messages"].as_array().unwrap().len(), 4);
    }

    fn dropping_context(drops_unsupported_params: bool) -> MessagesTransformContext {
        MessagesTransformContext {
            drops_unsupported_params,
            ..MessagesTransformContext::default()
        }
    }

    #[rstest]
    #[case::string("be terse", Some(json!([{"type": "text", "text": "be terse"}])))]
    #[case::empty_string("", None)]
    fn the_system_string_is_sent_as_blocks_as_python_does(
        #[case] system: &str,
        #[case] expected: Option<Value>,
    ) {
        let body = transformed(
            json!({"model": "claude-sonnet-4-5", "max_tokens": 16, "system": system, "messages": [{"role": "user", "content": "a"}]}),
            &MessagesTransformContext::default(),
        );
        assert_eq!(body.get("system").cloned(), expected);
    }

    #[rstest]
    #[case::dropped_under_drop_params(true, 2)]
    #[case::kept_without_drop_params(false, 3)]
    fn a_converted_system_turn_loses_its_tool_changes_under_drop_params(
        #[case] drops_unsupported_params: bool,
        #[case] converted_blocks: usize,
    ) {
        let body = transformed(
            json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "messages": [
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                    {"role": "system", "content": [
                        {"type": "text", "text": "reminder"},
                        {"type": "tool_addition", "tool": {"type": "tool_reference", "tool_name": "t"}}
                    ]},
                    {"role": "user", "content": "c"}
                ]
            }),
            &dropping_context(drops_unsupported_params),
        );
        assert_eq!(body["messages"][2]["role"], json!("user"));
        assert_eq!(
            body["messages"][2]["content"].as_array().unwrap().len(),
            converted_blocks
        );
    }

    fn reminder(system: Value) -> Value {
        json!({
            "model": "claude-opus-5-5",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                system,
                {"role": "user", "content": "go on"}
            ]
        })
    }

    fn mid_conversation_system(drops_unsupported_params: bool) -> MessagesTransformContext {
        MessagesTransformContext {
            thinking: ThinkingContext {
                capabilities: MessagesModelCapabilities {
                    supports_mid_conversation_system: true,
                    ..MessagesModelCapabilities::default()
                },
                ..ThinkingContext::default()
            },
            drops_unsupported_params,
            ..MessagesTransformContext::default()
        }
    }

    #[rstest]
    #[case::clear_at_without_its_beta(
        json!({"role": "system", "content": "note", "clear_at": {"type": "turns", "value": 2}}),
        &[],
        true,
        json!({"role": "system", "content": "note"})
    )]
    #[case::clear_at_with_its_beta(
        json!({"role": "system", "content": "note", "clear_at": {"type": "turns", "value": 2}}),
        &[("anthropic-beta", "mid-conversation-system-clear-at-2026-08-21")],
        true,
        json!({"role": "system", "content": "note", "clear_at": {"type": "turns", "value": 2}})
    )]
    #[case::clear_at_kept_without_drop_params(
        json!({"role": "system", "content": "note", "clear_at": {"type": "turns", "value": 2}}),
        &[],
        false,
        json!({"role": "system", "content": "note", "clear_at": {"type": "turns", "value": 2}})
    )]
    #[case::tool_addition_without_its_beta(
        json!({"role": "system", "content": [{"type": "text", "text": "note"}, {"type": "tool_addition", "tool": {"name": "t"}}]}),
        &[("anthropic-beta", "context-1m-2025-08-07")],
        true,
        json!({"role": "system", "content": [{"type": "text", "text": "note"}]})
    )]
    fn fields_the_final_betas_do_not_unlock_are_stripped_under_drop_params(
        #[case] system: Value,
        #[case] headers: &[(&str, &str)],
        #[case] drops_unsupported_params: bool,
        #[case] expected: Value,
    ) {
        assert_eq!(
            finalized(
                reminder(system),
                headers,
                &mid_conversation_system(drops_unsupported_params)
            )["messages"][2],
            expected
        );
    }

    #[rstest]
    fn a_leading_system_turn_is_stripped_before_it_is_hoisted() {
        let body = finalized(
            json!({
                "model": "claude-opus-5-5",
                "max_tokens": 64,
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "be terse"}, {"type": "tool_addition", "tool": {"name": "t"}}]},
                    {"role": "user", "content": "hi"}
                ]
            }),
            &[],
            &mid_conversation_system(true),
        );
        assert_eq!(
            body["system"],
            json!([{"type": "text", "text": "be terse"}])
        );
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
            drops_unsupported_params: false,
            ..MessagesTransformContext::default()
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

    fn betas(sent: &[(&str, &str)], fields: Value) -> Vec<String> {
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
        VERTEX_AI_MESSAGES_CONFIG
            .request_headers(
                sent.iter()
                    .map(|(name, value)| (name.to_string(), value.to_string()))
                    .collect(),
                &request,
            )
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
    #[case::compaction(json!({"compaction": {"type": "summarize"}}), &["compact-2026-09-04"])]
    #[case::thinking_display_updates(
        json!({"thinking": {"type": "adaptive", "display": "updates"}}),
        &["thinking-display-updates-2026-08-18"]
    )]
    #[case::tool_addition(
        json!({"messages": [
            {"role": "user", "content": "Use the ping tool"},
            {"role": "system", "content": [{"type": "tool_addition", "tool": {"type": "tool_reference", "name": "ping"}}]}
        ]}),
        &["mid-conversation-tool-changes-2026-07-01"]
    )]
    #[case::advisor_tool_is_not_a_vertex_beta(
        json!({"tools": [{"type": "advisor_20260301", "name": "advisor"}]}),
        &[]
    )]
    #[case::fast_mode_is_not_a_vertex_beta(json!({"speed": "fast"}), &[])]
    #[case::structured_output_needs_no_vertex_beta(
        json!({"output_format": {"type": "json_schema", "schema": {"type": "object"}}}),
        &[]
    )]
    #[case::output_config_format_needs_no_vertex_beta(
        json!({"output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}}}),
        &[]
    )]
    fn beta_headers_follow_the_vertex_policy(#[case] fields: Value, #[case] expected: &[&str]) {
        assert_eq!(betas(&[], fields), expected);
    }

    #[rstest]
    #[case::supported_kept("context-1m-2025-08-07", &["context-1m-2025-08-07"])]
    #[case::renamed_for_vertex("advanced-tool-use-2025-11-20", &["tool-search-tool-2025-10-19"])]
    #[case::header_only_row_kept("prompt-caching-2024-07-31", &["prompt-caching-2024-07-31"])]
    #[case::structured_outputs_kept("structured-outputs-2025-11-13", &["structured-outputs-2025-11-13"])]
    #[case::unsupported_dropped("files-api-2025-04-14,mcp-client-2025-11-20", &[])]
    #[case::unknown_dropped("not-a-beta-2099-01-01", &[])]
    #[case::mixed(
        " files-api-2025-04-14 , context-1m-2025-08-07,unknown ",
        &["context-1m-2025-08-07"]
    )]
    fn caller_betas_are_filtered_through_the_vertex_row(
        #[case] sent: &str,
        #[case] expected: &[&str],
    ) {
        assert_eq!(betas(&[("anthropic-beta", sent)], json!({})), expected);
    }

    #[rstest]
    fn caller_and_feature_betas_are_filtered_together() {
        assert_eq!(
            betas(
                &[(
                    "anthropic-beta",
                    "files-api-2025-04-14,context-1m-2025-08-07"
                )],
                json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]})
            ),
            &["context-1m-2025-08-07", "web-search-2025-03-05"]
        );
    }

    #[rstest]
    fn a_header_left_without_betas_is_removed() {
        let request: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap();
        assert_eq!(
            VERTEX_AI_MESSAGES_CONFIG.request_headers(
                vec![
                    (
                        "anthropic-beta".to_string(),
                        "files-api-2025-04-14".to_string()
                    ),
                    ("x-trace".to_string(), "1".to_string()),
                ],
                &request,
            ),
            vec![("x-trace".to_string(), "1".to_string())]
        );
    }

    #[derive(Clone, Default)]
    struct Warnings(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

    impl litellm_tracing::Sink for Warnings {
        fn enabled(&self, metadata: &litellm_tracing::Metadata<'_>) -> bool {
            *metadata.level() == litellm_tracing::Level::WARN
        }

        fn emit(&self, record: &litellm_tracing::Record) {
            self.0.lock().unwrap().push(record.message.clone());
        }
    }

    #[rstest]
    #[case::vertex_ai_warns(
        true,
        vec!["Dropping `thinking={'type': 'disabled'}` for model=claude-sonnet-5-5@default: thinking is always on for this model and cannot be disabled (the alternative is a provider 400). The model will still think adaptively, its response can contain thinking blocks, and those thinking tokens are billed as output tokens.".to_owned()]
    )]
    #[case::anthropic_stays_silent_as_on_main(false, vec![])]
    fn a_disabled_thinking_drop_warns_as_python_on_vertex_ai_only(
        #[case] vertex: bool,
        #[case] expected: Vec<String>,
    ) {
        let request: MessagesRequest = serde_json::from_value(json!({
            "model": "claude-sonnet-5-5@default",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "disabled"},
        }))
        .unwrap();
        let context = MessagesTransformContext {
            thinking: ThinkingContext {
                capabilities: MessagesModelCapabilities {
                    thinking_always_on: true,
                    ..MessagesModelCapabilities::default()
                },
                ..ThinkingContext::default()
            },
            ..MessagesTransformContext::default()
        };
        let warnings = Warnings::default();

        let transformed = litellm_tracing::Logger::new(warnings.clone()).scope(|| match vertex {
            true => {
                VERTEX_AI_MESSAGES_CONFIG.transform_anthropic_messages_request(request, &context)
            }
            false => crate::anthropic::messages::transformation::ANTHROPIC_MESSAGES_CONFIG
                .transform_anthropic_messages_request(request, &context),
        });

        assert_eq!(transformed.unwrap().params.thinking, None);
        assert_eq!(*warnings.0.lock().unwrap(), expected);
    }

    #[rstest]
    #[case::unsupported(Some("us-east5"), vec!["Vertex AI model claude-fz-regional-5-5@default is not served in us-east5, so the call goes to europe-west1 (served in europe-west1, asia-southeast1).".to_owned()])]
    #[case::supported(Some("asia-southeast1"), vec![])]
    #[case::unconfigured(None, vec![])]
    fn an_overridden_region_warns_as_get_vertex_region_does(
        #[case] location: Option<&str>,
        #[case] expected: Vec<String>,
    ) {
        let supported = ["europe-west1".to_owned(), "asia-southeast1".to_owned()];
        let warnings = Warnings::default();
        litellm_tracing::Logger::new(warnings.clone()).scope(|| {
            vertex_connection_params(
                params(Some("proj"), location),
                "claude-fz-regional-5-5@default",
                None,
                &supported,
                &no_env,
            )
        });
        assert_eq!(*warnings.0.lock().unwrap(), expected);
    }

    /// The URL Python's `get_complete_vertex_url` builds for each base, location, model and
    /// stream flag in `testdata/python_urls.json`, written by LiteLLM's Python.
    #[rstest]
    fn the_url_matches_python_s_for_every_normal_base() {
        #[derive(serde::Deserialize)]
        struct Row {
            api_base: Option<String>,
            location: String,
            model: String,
            stream: bool,
            url: String,
        }
        let rows: Vec<Row> =
            serde_json::from_str(include_str!("testdata/python_urls.json")).unwrap();
        assert!(!rows.is_empty());
        for row in rows {
            assert_eq!(
                complete_vertex_anthropic_url(
                    row.api_base.as_deref(),
                    "p",
                    &row.location,
                    &row.model,
                    row.stream
                )
                .unwrap(),
                row.url,
                "{:?} {} {} stream={}",
                row.api_base,
                row.location,
                row.model,
                row.stream
            );
        }
    }
}
