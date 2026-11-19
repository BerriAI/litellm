//! Claude Platform on AWS (`bedrock/claude_platform/`): plain Anthropic Messages sent to
//! `aws-external-anthropic.<region>.api.aws`, billed to the workspace named in
//! `anthropic-workspace-id`, authenticated by an API key, the Bedrock bearer token or SigV4.
use std::sync::LazyLock;

use litellm_auth::{AwsParams, ParamKind, ParamSpec};
use litellm_auth_aws::{AwsCredentialSource, CallerSigningHeaders, resolve_aws_region};
use litellm_env::resolve_non_empty;
use litellm_host::interceptors::RetriedRequest;
use litellm_http::request::{header_value, with_header, without_headers};
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest},
    providers::anthropic::{BetaProvider, DEFAULT_HEADERS, MESSAGES_PATH},
};
use litellm_router_types::LitellmParams;
use serde::de::Error as _;
use serde_json::Value;

use crate::{
    Error, ErrorDetail,
    anthropic::messages::{
        handler::shape_anthropic_messages_request,
        transformation::{
            retry_anthropic_messages_after_http_error, transform_messages_request,
            update_headers_with_safeguards_beta,
        },
    },
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        messages::{
            context::{MessagesShapeOptions, MessagesTransformContext},
            transformation::BaseMessagesConfig,
        },
    },
    bedrock::{common_utils::is_valid_region, messages::mantle_transformation::bearer_environment},
};

const SERVICE: &str = "aws-external-anthropic";
const ROUTE: &str = "claude_platform/";
const PROVIDER: &str = "Claude Platform on AWS";
const API_KEY_ENV: &str = "ANTHROPIC_AWS_API_KEY";
const BEARER_ENV: &str = "AWS_BEARER_TOKEN_BEDROCK";
const BASE_ENVS: &[&str] = &["ANTHROPIC_AWS_BASE_URL", "ANTHROPIC_AWS_API_BASE"];
const WORKSPACE_ENVS: &[&str] = &["ANTHROPIC_AWS_WORKSPACE_ID", "ANTHROPIC_WORKSPACE_ID"];
const REGION_ENVS: &[&str] = &["AWS_REGION_NAME", "AWS_REGION", "AWS_DEFAULT_REGION"];
const WORKSPACE_HEADER: &str = "anthropic-workspace-id";
const DEFAULT_UNSUPPORTED: &[&str] = &["context_management"];
/// The region botocore signs in when nothing names one.
const DEFAULT_SIGNING_REGION: &str = "us-west-2";

/// The workspace Python's config requires before it builds the call, raising `AuthenticationError`.
static WORKSPACE: ParamSpec = ParamSpec {
    setting: "workspace_id",
    kind: ParamKind::Credential,
    wire: &[
        "workspace_id",
        "aws_workspace_id",
        "anthropic-workspace-id",
        "anthropic_workspace_id",
    ],
    module_global: None,
    env: WORKSPACE_ENVS,
};

/// The region Python's config requires to build the default endpoint, raising `AuthenticationError`
/// once its logger knows the resolved call.
pub static REGION: ParamSpec = ParamSpec {
    setting: "region",
    kind: ParamKind::Credential,
    wire: &["aws_region_name"],
    module_global: None,
    env: REGION_ENVS,
};

/// The `bedrock/claude_platform/` Messages config, Python's `BedrockClaudePlatformMessagesConfig`.
pub struct BedrockClaudePlatformMessagesConfig;

pub const BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG: BedrockClaudePlatformMessagesConfig =
    BedrockClaudePlatformMessagesConfig;

impl BaseMessagesConfig for BedrockClaudePlatformMessagesConfig {
    fn tool_custom_field(&self, model: &str) -> crate::base_llm::messages::tools::ToolCustomField {
        crate::base_llm::messages::tools::ToolCustomField::Rewrite {
            hoist_defer_loading: crate::anthropic::messages::transformation::is_claude_model(model),
        }
    }

    fn retry_after_http_error(
        &self,
        status: u16,
        error_text: &str,
        request_body: &Value,
        context: &MessagesTransformContext,
    ) -> Option<RetriedRequest> {
        retry_anthropic_messages_after_http_error(status, error_text, request_body, context)
    }

    fn shape_request(
        &self,
        request: MessagesRequest,
        options: MessagesShapeOptions,
    ) -> Result<MessagesRequest, Error> {
        shape_anthropic_messages_request(request, options)
    }

    /// Python's order: an API key goes out as `x-api-key` unless the caller forwarded one, and
    /// the call is sent unsigned. Without a key the Bedrock bearer token, then SigV4, authenticate it.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        _model: &str,
        _optional_params: &MessagesOptionalParams,
        params: &LitellmParams,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let workspace = resolve_non_empty(
            params.claude_platform.workspace().as_deref(),
            env,
            WORKSPACE_ENVS,
        )
        .ok_or_else(|| Error::Auth(WORKSPACE.missing(PROVIDER)))?;
        let headers = with_header(headers, WORKSPACE_HEADER, workspace);
        let key = resolve_non_empty(api_key.filter(|key| !key.is_empty()), env, &[API_KEY_ENV]);
        if let Some(key) = key {
            let headers = match header_value(&headers, "x-api-key") {
                Some(_) => headers,
                None => with_header(headers, "x-api-key", key),
            };
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        if let Some(token) = resolve_non_empty(None, env, &[BEARER_ENV]) {
            return Ok(bearer_environment(headers, &token));
        }
        let base = resolve_non_empty(params.api_base.as_deref(), env, BASE_ENVS);
        Ok(ValidatedEnvironment {
            headers: without_headers(headers, &["authorization"]),
            auth: AuthScheme::AwsSigV4 {
                region: signing_region(base.as_deref(), params, env)?,
                service: SERVICE,
                credentials: Box::new(AwsCredentialSource::from_params(&params.aws, env)),
                caller_headers: CallerSigningHeaders::Botocore,
            },
        })
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        params: &LitellmParams,
        _stream: bool,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let base = resolve_non_empty(
            api_base
                .filter(|base| !base.is_empty())
                .or(params.api_base.as_deref()),
            env,
            BASE_ENVS,
        );
        let base = match base {
            Some(base) => base,
            None => format!(
                "https://{SERVICE}.{}.api.aws",
                configured_region(params, env)?
            ),
        };
        Ok(messages_url(&base))
    }

    /// Python's transform sends the model without its route and warns when it drops a
    /// disabled thinking on an always-thinking model.
    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        let model = request
            .model
            .strip_prefix(ROUTE)
            .unwrap_or(&request.model)
            .to_string();
        transform_messages_request(
            MessagesRequest { model, ..request },
            &context.warning_dropped_disabled_thinking(),
        )
    }

    /// Python's `filter_claude_platform_request_body`: `context_management` by default, or the
    /// list a deployment names in `claude_platform_unsupported_params`.
    fn without_rejected_params(
        &self,
        request: MessagesRequest,
        params: &LitellmParams,
    ) -> Result<MessagesRequest, Error> {
        if let Some(type_name) = params.claude_platform.ignored_override_type() {
            litellm_tracing::warn!(
                "bedrock/claude_platform: ignoring claude_platform_unsupported_params of type {type_name}; expected a list of param names. Using the default unsupported-param set."
            );
        }
        let unsupported: Vec<&str> = match params.claude_platform.unsupported_params() {
            Some(names) => names.iter().map(String::as_str).collect(),
            None => DEFAULT_UNSUPPORTED.to_vec(),
        };
        without_params(request, &unsupported)
    }

    fn secret_names(&self) -> &'static [&'static str] {
        static NAMES: LazyLock<Box<[&'static str]>> = LazyLock::new(|| {
            [API_KEY_ENV, BEARER_ENV]
                .into_iter()
                .chain(BASE_ENVS.iter().copied())
                .chain(WORKSPACE_ENVS.iter().copied())
                .chain(REGION_ENVS.iter().copied())
                .chain(AwsParams::secret_names().iter().copied())
                .collect()
        });
        &NAMES
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        update_headers_with_safeguards_beta(headers, request, BetaProvider::Anthropic)
    }

    /// The call is billed on the `bedrock/claude_platform/` rows, as Python bills it.
    fn logged_model(&self, model: &str) -> String {
        match model.starts_with(ROUTE) {
            true => model.to_string(),
            false => format!("{ROUTE}{model}"),
        }
    }
}

/// Python's `BedrockModelInfo._explicit_claude_platform_route`, which the route's model keeps
/// with or without its `bedrock/` provider prefix.
pub fn is_claude_platform_route(model: &str) -> bool {
    model.starts_with(ROUTE) || model.contains(&format!("/{ROUTE}"))
}

/// Whether the call was refused for want of the region its default endpoint is built in.
pub fn is_missing_region(error: &litellm_auth::Error) -> bool {
    matches!(error, litellm_auth::Error::MissingParam { spec, .. } if std::ptr::eq(*spec, &REGION))
}

/// The model Python's handler logs a refused call under: the route's model without `claude_platform/`.
pub fn logged_model(model: &str) -> &str {
    let routed = model.strip_prefix("bedrock/").unwrap_or(model);
    routed.strip_prefix(ROUTE).unwrap_or(routed)
}

/// Python's `ensure_api_route(base, "v1/messages")`.
fn messages_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    match base.ends_with(MESSAGES_PATH) {
        true => base.to_string(),
        false => format!("{base}{MESSAGES_PATH}"),
    }
}

/// Python's `_get_required_aws_region_name`, which only the default endpoint needs.
fn configured_region(
    params: &LitellmParams,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let region = resolve_non_empty(params.aws.aws_region_name.as_deref(), env, REGION_ENVS)
        .ok_or_else(|| Error::Auth(REGION.missing(PROVIDER)))?;
    checked_region(region)
}

/// The region SigV4 signs in: a public `aws-external-anthropic.<region>.api.aws` base names its
/// own, as the Mantle routes sign in their host's region; otherwise the configured one.
fn signing_region(
    base: Option<&str>,
    params: &LitellmParams,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let host_region = base.and_then(|base| {
        let url = url::Url::parse(base).ok()?;
        url.host_str()?
            .strip_prefix("aws-external-anthropic.")?
            .strip_suffix(".api.aws")
            .map(str::to_string)
    });
    match host_region {
        Some(region) => checked_region(region),
        None => checked_region(
            resolve_aws_region(None, &params.aws, env)
                .or_else(|| resolve_non_empty(None, env, &["AWS_DEFAULT_REGION"]))
                .unwrap_or_else(|| DEFAULT_SIGNING_REGION.to_string()),
        ),
    }
}

fn checked_region(region: String) -> Result<String, Error> {
    match is_valid_region(&region) {
        true => Ok(region),
        false => Err(Error::InvalidRequest(ErrorDetail::InvalidAwsRegion {
            region,
        })),
    }
}

/// The request without the params the endpoint rejects, which Python drops before its transform.
fn without_params(request: MessagesRequest, names: &[&str]) -> Result<MessagesRequest, Error> {
    let Value::Object(fields) = serde_json::to_value(request).map_err(invalid_request)? else {
        return Err(Error::InvalidRequest(ErrorDetail::invalid(
            "Messages request",
            serde_json::Error::custom("not an object"),
        )));
    };
    let (dropped, kept): (Vec<_>, Vec<_>) = fields
        .into_iter()
        .partition(|(name, _)| names.contains(&name.as_str()));
    if !dropped.is_empty() {
        litellm_tracing::warn!(
            "bedrock/claude_platform: dropping unsupported Messages API param(s) {} from the request body; the Claude Platform on AWS endpoint rejects unknown fields. The request will proceed without them.",
            python_tuple(dropped.iter().map(|(name, _)| name.as_str()))
        );
    }
    serde_json::from_value(Value::Object(kept.into_iter().collect())).map_err(invalid_request)
}

/// How Python prints the tuple of dropped param names in its warning.
fn python_tuple<'a>(names: impl Iterator<Item = &'a str>) -> String {
    let quoted: Vec<String> = names.map(|name| format!("'{name}'")).collect();
    match quoted.as_slice() {
        [only] => format!("({only},)"),
        _ => format!("({})", quoted.join(", ")),
    }
}

fn invalid_request(error: serde_json::Error) -> Error {
    Error::InvalidRequest(ErrorDetail::invalid("Messages request", error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::json;

    fn params(fields: Value) -> LitellmParams {
        let model = json!({"model": "bedrock/claude_platform/claude-test"});
        serde_json::from_value(Value::Object(
            model
                .as_object()
                .unwrap()
                .clone()
                .into_iter()
                .chain(fields.as_object().unwrap().clone())
                .collect(),
        ))
        .unwrap()
    }

    const BEARER: (&str, &str) = ("AWS_BEARER_TOKEN_BEDROCK", "bearer");

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    fn validated(
        headers: &[(&str, &str)],
        api_key: Option<&str>,
        fields: Value,
        pairs: &'static [(&'static str, &'static str)],
    ) -> Result<ValidatedEnvironment, Error> {
        BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG.validate_environment(
            headers
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
            api_key,
            "claude_platform/claude-test",
            &MessagesOptionalParams::default(),
            &params(fields),
            &env(pairs),
        )
    }

    #[rstest]
    #[case::workspace_id(json!({"workspace_id": "a", "aws_workspace_id": "b"}), &[], "a")]
    #[case::aws_workspace_id(json!({"aws_workspace_id": "b", "anthropic_workspace_id": "d"}), &[], "b")]
    #[case::header_spelling(json!({"anthropic-workspace-id": "c", "anthropic_workspace_id": "d"}), &[], "c")]
    #[case::anthropic_workspace_id(json!({"anthropic_workspace_id": "d"}), &[], "d")]
    #[case::empty_falls_through(json!({"workspace_id": "", "aws_workspace_id": "b"}), &[], "b")]
    #[case::number(json!({"workspace_id": 123}), &[], "123")]
    #[case::env(json!({}), &[("ANTHROPIC_AWS_WORKSPACE_ID", "env")], "env")]
    #[case::second_env(json!({}), &[("ANTHROPIC_WORKSPACE_ID", "env2")], "env2")]
    #[case::param_over_env(json!({"workspace_id": "a"}), &[("ANTHROPIC_AWS_WORKSPACE_ID", "env")], "a")]
    fn the_workspace_replaces_any_caller_header(
        #[case] fields: Value,
        #[case] pairs: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        let environment =
            validated(&[(WORKSPACE_HEADER, "caller")], Some("key"), fields, pairs).unwrap();
        assert_eq!(
            header_value(&environment.headers, WORKSPACE_HEADER),
            Some(expected)
        );
    }

    #[rstest]
    #[case::nothing(&[])]
    #[case::caller_header_alone(&[(WORKSPACE_HEADER, "caller")])]
    fn a_call_without_a_configured_workspace_is_a_missing_credential(
        #[case] headers: &[(&str, &str)],
    ) {
        let Err(Error::Auth(error)) = validated(headers, Some("key"), json!({}), &[]) else {
            panic!("the workspace is required");
        };
        assert!(error.is_missing_credential());
        assert!(!is_missing_region(&error));
    }

    #[rstest]
    #[case::explicit_key(&[], Some("key"), &[BEARER], Some("key"))]
    #[case::env_key(&[], None, &[("ANTHROPIC_AWS_API_KEY", "env-key"), BEARER], Some("env-key"))]
    #[case::empty_key_reads_env(&[], Some(""), &[("ANTHROPIC_AWS_API_KEY", "env-key"), BEARER], Some("env-key"))]
    #[case::forwarded_key_wins(&[("x-api-key", "caller")], Some("key"), &[BEARER], Some("caller"))]
    fn an_api_key_goes_out_unsigned_as_x_api_key(
        #[case] headers: &[(&str, &str)],
        #[case] api_key: Option<&str>,
        #[case] pairs: &'static [(&'static str, &'static str)],
        #[case] expected: Option<&str>,
    ) {
        let environment = validated(headers, api_key, json!({"workspace_id": "w"}), pairs).unwrap();
        assert!(matches!(environment.auth, AuthScheme::Forwarded));
        assert_eq!(header_value(&environment.headers, "x-api-key"), expected);
        assert_eq!(header_value(&environment.headers, "authorization"), None);
    }

    #[rstest]
    fn a_forwarded_authorization_stays_beside_the_api_key() {
        let environment = validated(
            &[("authorization", "Bearer caller")],
            Some("key"),
            json!({"workspace_id": "w"}),
            &[],
        )
        .unwrap();
        assert_eq!(
            header_value(&environment.headers, "authorization"),
            Some("Bearer caller")
        );
    }

    #[rstest]
    fn without_a_key_the_bedrock_bearer_token_authenticates_the_call() {
        let environment = validated(
            &[("x-api-key", "caller")],
            None,
            json!({"workspace_id": "w"}),
            &[("AWS_BEARER_TOKEN_BEDROCK", "bearer")],
        )
        .unwrap();
        assert!(matches!(environment.auth, AuthScheme::Forwarded));
        assert_eq!(
            header_value(&environment.headers, "authorization"),
            Some("Bearer bearer")
        );
    }

    #[rstest]
    #[case::configured_region(json!({"aws_region_name": "eu-west-1"}), "eu-west-1")]
    #[case::botocore_default(json!({"api_base": "https://gw.example"}), "us-west-2")]
    #[case::public_host_names_its_region(
        json!({"aws_region_name": "eu-west-1", "api_base": "https://aws-external-anthropic.us-west-2.api.aws"}),
        "us-west-2"
    )]
    fn without_a_key_or_bearer_sigv4_signs_and_drops_a_forwarded_authorization(
        #[case] extra: Value,
        #[case] expected_region: &str,
    ) {
        let fields = json!({"workspace_id": "w"})
            .as_object()
            .unwrap()
            .clone()
            .into_iter()
            .chain(extra.as_object().unwrap().clone())
            .collect();
        let environment = validated(
            &[("authorization", "Bearer caller"), ("x-api-key", "caller")],
            None,
            Value::Object(fields),
            &[],
        )
        .unwrap();
        let AuthScheme::AwsSigV4 {
            region, service, ..
        } = &environment.auth
        else {
            panic!("expected SigV4");
        };
        assert_eq!(region, expected_region);
        assert_eq!(*service, SERVICE);
        assert_eq!(header_value(&environment.headers, "authorization"), None);
        assert_eq!(
            header_value(&environment.headers, "x-api-key"),
            Some("caller")
        );
    }

    fn url(
        api_base: Option<&str>,
        fields: Value,
        pairs: &'static [(&'static str, &'static str)],
    ) -> Result<String, Error> {
        BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG.get_complete_url(
            api_base,
            "claude_platform/claude-test",
            &params(fields),
            false,
            &env(pairs),
        )
    }

    #[rstest]
    #[case::bare("https://gw.example", "https://gw.example/v1/messages")]
    #[case::trailing_slash("https://gw.example/", "https://gw.example/v1/messages")]
    #[case::full_route("https://gw.example/v1/messages", "https://gw.example/v1/messages")]
    #[case::path_prefix("https://gw.example/proxy", "https://gw.example/proxy/v1/messages")]
    fn a_base_gets_the_messages_route_once(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(url(Some(base), json!({}), &[]).unwrap(), expected);
    }

    #[rstest]
    #[case::env_base(None, json!({}), &[("ANTHROPIC_AWS_BASE_URL", "https://env.example")], "https://env.example/v1/messages")]
    #[case::second_env_base(None, json!({}), &[("ANTHROPIC_AWS_API_BASE", "https://env2.example")], "https://env2.example/v1/messages")]
    #[case::call_base_over_env(Some("https://call.example"), json!({}), &[("ANTHROPIC_AWS_BASE_URL", "https://env.example")], "https://call.example/v1/messages")]
    #[case::default_endpoint(None, json!({"aws_region_name": "us-west-2"}), &[], "https://aws-external-anthropic.us-west-2.api.aws/v1/messages")]
    #[case::default_endpoint_env_region(None, json!({}), &[("AWS_DEFAULT_REGION", "eu-west-1")], "https://aws-external-anthropic.eu-west-1.api.aws/v1/messages")]
    fn the_base_is_the_call_s_then_the_env_s_then_the_regional_endpoint(
        #[case] api_base: Option<&str>,
        #[case] fields: Value,
        #[case] pairs: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        assert_eq!(url(api_base, fields, pairs).unwrap(), expected);
    }

    #[rstest]
    fn the_default_endpoint_without_a_region_is_a_missing_region() {
        let Err(Error::Auth(error)) = url(None, json!({}), &[]) else {
            panic!("the region is required");
        };
        assert!(is_missing_region(&error));
        assert!(error.is_missing_credential());
    }

    #[rstest]
    fn an_injected_region_never_reaches_the_host() {
        let result = url(
            None,
            json!({"aws_region_name": "us-east-1.attacker.test/x?"}),
            &[],
        );
        assert!(matches!(
            result,
            Err(Error::InvalidRequest(ErrorDetail::InvalidAwsRegion { .. }))
        ));
    }

    fn request(fields: Value) -> MessagesRequest {
        serde_json::from_value(
            json!({"model": "claude_platform/claude-test", "max_tokens": 16, "messages": []})
                .as_object()
                .unwrap()
                .clone()
                .into_iter()
                .chain(fields.as_object().unwrap().clone())
                .collect::<serde_json::Map<String, Value>>()
                .into(),
        )
        .unwrap()
    }

    #[rstest]
    #[case::default_drops_context_management(json!({}), false, true)]
    #[case::override_keeps_it(json!({"claude_platform_unsupported_params": []}), true, true)]
    #[case::override_drops_temperature(json!({"claude_platform_unsupported_params": ["temperature"]}), true, false)]
    #[case::non_list_override_is_ignored(json!({"claude_platform_unsupported_params": "temperature"}), false, true)]
    fn the_endpoint_s_rejected_params_are_dropped(
        #[case] fields: Value,
        #[case] keeps_context_management: bool,
        #[case] keeps_temperature: bool,
    ) {
        let filtered = BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG
            .without_rejected_params(
                request(json!({"temperature": 0.5, "context_management": {"edits": []}})),
                &params(fields),
            )
            .unwrap();
        let body = serde_json::to_value(filtered).unwrap();
        assert_eq!(
            body.get("context_management").is_some(),
            keeps_context_management
        );
        assert_eq!(body.get("temperature").is_some(), keeps_temperature);
    }

    #[rstest]
    fn the_route_is_billed_but_never_sent() {
        let transformed = BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                request(json!({})),
                &MessagesTransformContext::default(),
            )
            .unwrap();
        assert_eq!(transformed.model, "claude-test");
        assert_eq!(
            BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG.logged_model(&transformed.model),
            "claude_platform/claude-test"
        );
    }

    #[rstest]
    #[case::one(&["context_management"], "('context_management',)")]
    #[case::two(&["temperature", "top_p"], "('temperature', 'top_p')")]
    fn dropped_params_are_named_as_python_prints_its_tuple(
        #[case] names: &[&str],
        #[case] expected: &str,
    ) {
        assert_eq!(python_tuple(names.iter().copied()), expected);
    }

    #[rstest]
    #[case::provider_prefixed("bedrock/claude_platform/claude-test", true, "claude-test")]
    #[case::routed("claude_platform/claude-test", true, "claude-test")]
    #[case::invoke("bedrock/us.anthropic.claude-test", false, "us.anthropic.claude-test")]
    #[case::mantle("bedrock/mantle/claude-test", false, "mantle/claude-test")]
    fn the_route_is_recognised_with_or_without_its_provider(
        #[case] model: &str,
        #[case] platform: bool,
        #[case] logged: &str,
    ) {
        assert_eq!(is_claude_platform_route(model), platform);
        assert_eq!(logged_model(model), logged);
    }
}
