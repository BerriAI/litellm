//! Bedrock Mantle's Anthropic Messages endpoint, Python's `BedrockMantleAnthropicMessagesConfig` on
//! the `bedrock_mantle/` provider and `AmazonMantleMessagesConfig` on the `bedrock/mantle/` route:
//! the Bedrock Invoke request transform with Mantle's overrides, the betas moved into a header, the
//! model id and `stream` back in the body, the Mantle URL and zero-filled usage. The two routes
//! differ only in their URL, region, bearer token, Invoke settings and the model id they send.

use std::sync::LazyLock;

use litellm_auth::AwsParams;
use litellm_auth_aws::{
    AwsCredentialSource, CallerSigningHeaders,
    constants::{
        AWS_BEARER_TOKEN_BEDROCK, AWS_DEFAULT_REGION, AWS_REGION, AWS_REGION_NAME, BEDROCK_SERVICE,
    },
    resolve_bedrock_region,
};
use litellm_host::interceptors::RetriedRequest;
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest, MessagesResponse},
    providers::anthropic::BetaProvider,
    recognized::Recognized,
};
use litellm_router_types::LitellmParams;
use serde::{Deserialize, Serialize, de::IgnoredAny};
use serde_json::{Map, Value};
use url::Url;

use crate::{
    Error, ErrorDetail, MantleRegionCheck,
    anthropic::messages::transformation::{
        ANTHROPIC_MESSAGES_CONFIG, drop_sampling_params_removed_by_requested_version,
    },
    base_llm::{
        auth::AuthScheme,
        messages::{
            context::{MessagesModelCapabilities, MessagesShapeOptions, MessagesTransformContext},
            transformation::{
                Headers, MessagesAuth, MessagesDialect, MessagesEndpoint, MessagesReply,
                ValidatedEnvironment,
            },
        },
    },
    bedrock::{
        common_utils::is_valid_region,
        messages::invoke::{
            request::{
                BEDROCK_INVOKE_REQUEST_SETTINGS, InvokeRequestSettings,
                clamp_adaptive_reasoning_effort_for_bedrock, transform_anthropic_messages_request,
                with_logged_tools,
            },
            transformation::{
                bearer_token, invoke_headers, native_output_fallback_litellm_params,
                region_from_model, should_retry_anthropic_messages_without_native_output,
            },
        },
    },
};

const BEDROCK_MANTLE: &str = "bedrock_mantle";
const BEDROCK: &str = "bedrock";
const MANTLE_MESSAGES_PATH: &str = "/anthropic/v1/messages";
const MANTLE_API_BASE_ENV: &str = "BEDROCK_MANTLE_API_BASE";
const MANTLE_OPENAI_BASE_SUFFIXES: [&str; 2] = ["/openai/v1", "/v1"];
const DEFAULT_ANTHROPIC_API_VERSION: &str = "2023-06-01";
const MANTLE_ALLOWED_TOP_LEVEL_FIELDS: [&str; 1] = ["service_tier"];
const USER_PROFILE_ID_HEADER: &str = "anthropic-user-profile-id";
const USER_PROFILE_ID_FIELD: &str = "user_profile_id";
const BEDROCK_MANTLE_DEFAULT_REGION: &str = "us-east-1";
/// Python's `AmazonBedrockGlobalConfig().get_all_regions()`, which `BEDROCK_REGIONS` freezes.
const BEDROCK_REGIONS: [&str; 23] = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "us-gov-east-1",
    "us-gov-west-1",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-central-1",
    "eu-central-2",
    "eu-south-1",
    "eu-south-2",
    "eu-north-1",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-south-1",
    "ap-south-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ca-central-1",
    "sa-east-1",
];
const REGION_ENVS: [&str; 3] = ["BEDROCK_MANTLE_REGION", "AWS_REGION_NAME", "AWS_REGION"];
const KEY_ENVS: [&str; 2] = ["BEDROCK_MANTLE_API_KEY", AWS_BEARER_TOKEN_BEDROCK];
const BEDROCK_MANTLE_SECRET_NAMES: [&str; 6] = [
    "BEDROCK_MANTLE_API_KEY",
    AWS_BEARER_TOKEN_BEDROCK,
    MANTLE_API_BASE_ENV,
    "BEDROCK_MANTLE_REGION",
    AWS_REGION_NAME,
    AWS_REGION,
];
const BEDROCK_ROUTE_SECRET_NAMES: [&str; 5] = [
    AWS_BEARER_TOKEN_BEDROCK,
    MANTLE_API_BASE_ENV,
    AWS_REGION_NAME,
    AWS_REGION,
    AWS_DEFAULT_REGION,
];
const BASE_SUFFIXES_TO_STRIP: [&str; 6] = [
    MANTLE_MESSAGES_PATH,
    "/v1/messages",
    "/messages",
    "/anthropic/v1",
    "/openai/v1",
    "/v1",
];
const CONTEXT_MANAGEMENT_EDITS: [(&str, &str); 3] = [
    ("compact_20260112", "compact-2026-01-12"),
    ("clear_tool_uses_20250919", "context-management-2025-06-27"),
    ("clear_thinking_20251015", "context-management-2025-06-27"),
];

const BEDROCK_ROUTE_REQUEST_SETTINGS: InvokeRequestSettings = InvokeRequestSettings {
    beta_headers_provider: BetaProvider::BedrockMantle,
    filters_caller_betas: false,
    drops_dangerous_tool_use_beta_without_safeguards: false,
    extra_allowed_top_level_fields: &MANTLE_ALLOWED_TOP_LEVEL_FIELDS,
    ..BEDROCK_INVOKE_REQUEST_SETTINGS
};

const BEDROCK_MANTLE_REQUEST_SETTINGS: InvokeRequestSettings = InvokeRequestSettings {
    custom_llm_provider: BetaProvider::BedrockMantle,
    context_management_edits: &CONTEXT_MANAGEMENT_EDITS,
    ..BEDROCK_ROUTE_REQUEST_SETTINGS
};

/// Which Mantle route a config serves, which picks its URL, region, bearer token, secrets and the
/// model id it sends.
#[derive(Clone, Copy)]
enum MantleRoute {
    /// `bedrock_mantle/`: Python's `build_mantle_native_messages_url` with a us-east-1 default,
    /// `BEDROCK_MANTLE_API_KEY` before `AWS_BEARER_TOKEN_BEDROCK`, SigV4 in the URL's region, and
    /// the model without its region prefix and first `mantle/`.
    BedrockMantle,
    /// `bedrock/mantle/`: Bedrock's region order with its us-west-2 default, Python's
    /// `build_mantle_messages_url`, only `AWS_BEARER_TOKEN_BEDROCK`, and the model without its
    /// first `mantle/`.
    Bedrock,
}

/// Bedrock Mantle's Messages config, one value per Mantle route.
pub struct BedrockMantleMessagesConfig {
    route: MantleRoute,
    request_settings: InvokeRequestSettings,
}

/// The `bedrock_mantle/` provider, for Claude models.
pub const BEDROCK_MANTLE_MESSAGES_CONFIG: BedrockMantleMessagesConfig =
    BedrockMantleMessagesConfig {
        route: MantleRoute::BedrockMantle,
        request_settings: BEDROCK_MANTLE_REQUEST_SETTINGS,
    };

/// The `bedrock/mantle/` route.
pub const BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG: BedrockMantleMessagesConfig =
    BedrockMantleMessagesConfig {
        route: MantleRoute::Bedrock,
        request_settings: BEDROCK_ROUTE_REQUEST_SETTINGS,
    };

/// The provider and model Python's Mantle Messages configs probe capabilities under: the shared
/// Anthropic transform reads the route's provider, the Bedrock Invoke helpers read `bedrock`, and
/// both get the model id the route sends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MantleCapabilityLookup {
    pub provider: &'static str,
    pub model: String,
}

/// The body Mantle receives: the Invoke body without the version and betas, which travel as
/// headers, with `model` and `stream` last and `stream` only when it is `true`.
#[derive(Deserialize, Serialize)]
struct MantleWireBody {
    #[serde(flatten)]
    fields: Map<String, Value>,
    #[serde(rename = "anthropic_version", default, skip_serializing)]
    _version: Option<IgnoredAny>,
    #[serde(rename = "anthropic_beta", default, skip_serializing)]
    _betas: Option<IgnoredAny>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    stream: Option<StreamFlag>,
}

/// A `stream` value: only a boolean `true` streams.
#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum StreamFlag {
    Flag(bool),
    #[serde(skip_serializing)]
    Other(IgnoredAny),
}

/// The usage a Mantle reply bills: the two token counts Python zero-fills, every other field as sent.
#[derive(Debug, Deserialize, Serialize)]
struct BilledUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(flatten)]
    other: Map<String, Value>,
}

/// The model id `bedrock/mantle/` sends: the model without its first `mantle/`.
pub fn amazon_mantle_model_id(model: &str) -> String {
    model.replacen("mantle/", "", 1)
}

/// The model Python's Mantle transform sends: `get_llm_provider` drops the region prefix, then
/// `AmazonMantleMessagesConfig` drops the first `mantle/`.
pub fn mantle_model_id(model: &str) -> String {
    amazon_mantle_model_id(split_mantle_region_prefix(model).1)
}

/// The lookup for `bedrock_mantle/` and `bedrock/mantle/`, `None` on every other route.
pub fn mantle_capability_lookup(provider: &str, model: &str) -> Option<MantleCapabilityLookup> {
    match provider {
        BEDROCK_MANTLE => Some(MantleCapabilityLookup {
            provider: BEDROCK_MANTLE,
            model: mantle_model_id(model.strip_prefix("bedrock_mantle/").unwrap_or(model)),
        }),
        BEDROCK
            if !has_route_prefix(model, "claude_platform/")
                && !has_route_prefix(model, "converse/")
                && has_route_prefix(model, "mantle/") =>
        {
            Some(MantleCapabilityLookup {
                provider: BEDROCK,
                model: amazon_mantle_model_id(model.strip_prefix("bedrock/").unwrap_or(model)),
            })
        }
        _ => None,
    }
}

/// The model Python's handler logs a `bedrock_mantle/` or `bedrock/mantle/` call under before the
/// Mantle config transforms the request: `get_llm_provider`'s model, which drops a `bedrock_mantle/`
/// region prefix but keeps a `mantle/` segment. `None` on every other route.
pub fn mantle_logged_model(provider: &str, model: &str) -> Option<String> {
    let lookup = mantle_capability_lookup(provider, model)?;
    let routed = model
        .strip_prefix(lookup.provider)
        .and_then(|rest| rest.strip_prefix('/'))
        .unwrap_or(model);
    Some(match lookup.provider {
        BEDROCK_MANTLE => BEDROCK_MANTLE_MESSAGES_CONFIG.logged_model(routed),
        _ => BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG.logged_model(routed),
    })
}

impl MantleRoute {
    fn model_id(self, model: &str) -> String {
        match self {
            MantleRoute::BedrockMantle => mantle_model_id(model),
            MantleRoute::Bedrock => amazon_mantle_model_id(model),
        }
    }
}

/// Python's `BedrockModelInfo._model_has_route_prefix`.
fn has_route_prefix(model: &str, prefix: &str) -> bool {
    model.starts_with(prefix) || model.contains(&format!("/{prefix}"))
}

/// Python's `split_mantle_region_prefix`, which `get_llm_provider` applies before the config sees the model.
fn split_mantle_region_prefix(model: &str) -> (Option<&str>, &str) {
    match model.split_once('/') {
        Some((head, tail)) if BEDROCK_REGIONS.contains(&head) => (Some(head), tail),
        _ => (None, model),
    }
}

fn env_value(env_lookup: &dyn Fn(&str) -> Option<String>, names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| env_lookup(name).filter(|value| !value.trim().is_empty()))
}

fn configured_region(litellm_params: &LitellmParams) -> Option<&str> {
    litellm_params
        .aws
        .aws_region_name
        .as_deref()
        .filter(|region| !region.is_empty())
}

fn validated_mantle_region(region: String, check: MantleRegionCheck) -> Result<String, Error> {
    match is_valid_region(&region) {
        true => Ok(region),
        false => Err(Error::InvalidRequest(ErrorDetail::InvalidMantleRegion {
            region,
            check,
        })),
    }
}

/// A Mantle base parsed as a URL, refused when it is not one.
fn parse_mantle_base(base: &str) -> Result<Url, Error> {
    Url::parse(base).map_err(|error| Error::InvalidRequest(ErrorDetail::invalid("api_base", error)))
}

/// `base` with the first of `suffixes` its path ends in taken off and the Messages route put on.
/// The query string stays after the path.
fn with_mantle_messages_path(mut base: Url, suffixes: &[&str]) -> Url {
    let path = base.path().trim_end_matches('/');
    let stripped = suffixes
        .iter()
        .find_map(|suffix| path.strip_suffix(suffix))
        .unwrap_or(path);
    let path = format!("{stripped}{MANTLE_MESSAGES_PATH}");
    base.set_path(&path);
    base
}

/// The region of a public Mantle host, `bedrock-mantle.<region>.api.aws` with no port. The url
/// crate lowercases the host, so the region signs as it is sent.
fn mantle_host_region(url: &Url) -> Option<&str> {
    if !matches!(url.scheme(), "http" | "https") || url.port().is_some() {
        return None;
    }
    let region = url
        .host_str()?
        .strip_prefix("bedrock-mantle.")?
        .strip_suffix(".api.aws")?;
    (!region.is_empty() && !region.contains('.')).then_some(region)
}

/// Python's `resolve_mantle_region`.
fn resolve_mantle_region(
    base: &Url,
    litellm_params: &LitellmParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    if let Some(region) = configured_region(litellm_params) {
        return validated_mantle_region(region.to_string(), MantleRegionCheck::ProviderResolution);
    }
    Ok(mantle_host_region(base)
        .map(str::to_string)
        .or_else(|| env_value(env_lookup, &REGION_ENVS))
        .unwrap_or_else(|| BEDROCK_MANTLE_DEFAULT_REGION.to_string()))
}

/// The base Python's `get_llm_provider` hands the `bedrock_mantle/` config: the caller's, the
/// env's, or the regional default. A blank `api_base` counts as unset.
fn dynamic_api_base(
    api_base: Option<&str>,
    model: &str,
    litellm_params: &LitellmParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let region = validated_mantle_region(
        configured_region(litellm_params)
            .or(split_mantle_region_prefix(model).0)
            .map(str::to_string)
            .or_else(|| env_value(env_lookup, &REGION_ENVS))
            .unwrap_or_else(|| BEDROCK_MANTLE_DEFAULT_REGION.to_string()),
        MantleRegionCheck::ProviderResolution,
    )?;
    Ok(api_base
        .filter(|base| !base.trim().is_empty())
        .map(str::to_string)
        .or_else(|| env_value(env_lookup, &[MANTLE_API_BASE_ENV]))
        .unwrap_or_else(|| format!("https://bedrock-mantle.{region}.api.aws/v1")))
}

/// Python's `build_mantle_native_messages_url`: one known suffix comes off the base's path and the
/// Messages route goes on, the caller's query string staying after it. A public Mantle host is
/// rebuilt on https in the resolved region.
fn build_mantle_native_messages_url(
    api_base: Option<&str>,
    model: &str,
    litellm_params: &LitellmParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Url, Error> {
    let base = parse_mantle_base(&dynamic_api_base(
        api_base,
        model,
        litellm_params,
        env_lookup,
    )?)?;
    let region = resolve_mantle_region(&base, litellm_params, env_lookup)?;
    if mantle_host_region(&base).is_none() {
        return Ok(with_mantle_messages_path(base, &BASE_SUFFIXES_TO_STRIP));
    }
    let mut url = parse_mantle_base(&format!(
        "https://bedrock-mantle.{region}.api.aws{MANTLE_MESSAGES_PATH}"
    ))?;
    url.set_query(base.query());
    Ok(url)
}

/// The Bedrock region `bedrock/mantle/` uses for the host and the SigV4 scope, in Python's
/// `_get_aws_region_name` order: `aws_region_name`, the model's ARN or `<region>/` path segment,
/// then the environment.
fn bedrock_region(
    model: &str,
    litellm_params: &LitellmParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    resolve_bedrock_region(
        region_from_model(model).as_deref(),
        &litellm_params.aws,
        env_lookup,
    )
}

/// Python's `_mantle_api_base_from_env`.
fn mantle_api_base_from_env(env_lookup: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    let base = env_lookup(MANTLE_API_BASE_ENV)?;
    let base = base.trim_end_matches('/');
    let stripped = MANTLE_OPENAI_BASE_SUFFIXES
        .iter()
        .find_map(|suffix| base.strip_suffix(suffix))
        .unwrap_or(base);
    Some(stripped.to_string()).filter(|base| !base.trim().is_empty())
}

/// Python's `build_mantle_messages_url`. A blank base counts as unset.
fn build_mantle_messages_url(
    api_base: Option<&str>,
    aws_bedrock_runtime_endpoint: Option<&str>,
    region: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let explicit = [api_base, aws_bedrock_runtime_endpoint]
        .into_iter()
        .flatten()
        .find(|base| !base.trim().is_empty())
        .map(str::to_string);
    match explicit.or_else(|| mantle_api_base_from_env(env_lookup)) {
        Some(base) => Ok(with_mantle_messages_path(
            parse_mantle_base(&base)?,
            &[MANTLE_MESSAGES_PATH],
        )
        .into()),
        None => Ok(format!(
            "https://bedrock-mantle.{region}.api.aws{MANTLE_MESSAGES_PATH}"
        )),
    }
}

fn bedrock_route_messages_url(
    api_base: Option<&str>,
    model: &str,
    litellm_params: &LitellmParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let region = validated_mantle_region(
        bedrock_region(model, litellm_params, env_lookup),
        MantleRegionCheck::Request,
    )?;
    build_mantle_messages_url(
        api_base,
        litellm_params.aws.aws_bedrock_runtime_endpoint.as_deref(),
        &region,
        env_lookup,
    )
}

fn mantle_secret_names(names: &'static [&'static str]) -> Box<[&'static str]> {
    names
        .iter()
        .copied()
        .chain(
            AwsParams::secret_names()
                .iter()
                .copied()
                .filter(|name| !names.contains(name)),
        )
        .collect()
}

fn without_exact(headers: Headers, name: &str) -> Headers {
    headers.into_iter().filter(|(key, _)| key != name).collect()
}

fn has_any_casing(headers: &Headers, name: &str) -> bool {
    headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case(name))
}

/// Python's `AmazonMantleMessagesConfig.validate_anthropic_messages_environment`.
fn mantle_headers(headers: Headers, litellm_params: &LitellmParams) -> Result<Headers, Error> {
    let headers = invoke_headers(headers, litellm_params)?;
    let project = litellm_params
        .aws_bedrock_project_id
        .as_deref()
        .filter(|project| !project.is_empty());
    let has_version = has_any_casing(&headers, "anthropic-version");
    let headers = match project {
        Some(project) => without_exact(headers, "anthropic-workspace-id")
            .into_iter()
            .chain([("anthropic-workspace-id".to_string(), project.to_string())])
            .collect(),
        None => headers,
    };
    Ok(match has_version {
        true => headers,
        false => headers
            .into_iter()
            .chain([(
                "anthropic-version".to_string(),
                DEFAULT_ANTHROPIC_API_VERSION.to_string(),
            )])
            .collect(),
    })
}

// Forwarded credentials mirror Python's `_sign_request`: a bearer replaces only the exact `Authorization` key.
pub(crate) fn bearer_environment(headers: Headers, token: &str) -> ValidatedEnvironment {
    let headers = without_exact(without_exact(headers, "Content-Type"), "Authorization")
        .into_iter()
        .chain([
            ("Content-Type".to_string(), "application/json".to_string()),
            ("Authorization".to_string(), format!("Bearer {token}")),
        ])
        .collect();
    ValidatedEnvironment {
        headers,
        auth: AuthScheme::Forwarded,
    }
}

fn sigv4_environment(
    headers: Headers,
    region: String,
    aws: &AwsParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> ValidatedEnvironment {
    let headers = match has_any_casing(&headers, "content-type") {
        true => headers,
        false => [("Content-Type".to_string(), "application/json".to_string())]
            .into_iter()
            .chain(headers)
            .collect(),
    };
    ValidatedEnvironment {
        headers,
        auth: AuthScheme::AwsSigV4 {
            region,
            service: BEDROCK_SERVICE,
            credentials: Box::new(AwsCredentialSource::from_params(aws, env_lookup)),
            caller_headers: CallerSigningHeaders::Botocore,
        },
    }
}

/// `bedrock_mantle/` auth: a non-blank key or Mantle env key as a bearer, otherwise SigV4 in the
/// URL's region with every caller `Authorization` dropped.
fn bedrock_mantle_environment(
    headers: Headers,
    api_key: Option<&str>,
    model: &str,
    litellm_params: &LitellmParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<ValidatedEnvironment, Error> {
    if let Some(token) = api_key
        .filter(|key| !key.trim().is_empty())
        .map(str::to_string)
        .or_else(|| env_value(env_lookup, &KEY_ENVS))
    {
        return Ok(bearer_environment(headers, &token));
    }
    let url = build_mantle_native_messages_url(
        litellm_params.api_base.as_deref(),
        model,
        litellm_params,
        env_lookup,
    )?;
    let signing_region = match mantle_host_region(&url) {
        Some(region) => region.to_string(),
        None => resolve_mantle_region(&url, litellm_params, env_lookup)?,
    };
    let unsigned: Headers = headers
        .into_iter()
        .filter(|(name, _)| !name.eq_ignore_ascii_case("authorization"))
        .collect();
    Ok(sigv4_environment(
        unsigned,
        validated_mantle_region(signing_region, MantleRegionCheck::Request)?,
        &litellm_params.aws,
        env_lookup,
    ))
}

/// `bedrock/mantle/` auth: Bedrock Invoke's bearer token, otherwise SigV4 in the Bedrock region
/// with the caller's headers kept for the signer.
fn bedrock_route_environment(
    headers: Headers,
    api_key: Option<&str>,
    model: &str,
    litellm_params: &LitellmParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> ValidatedEnvironment {
    let api_key = api_key.filter(|key| key.is_empty() || !key.trim().is_empty());
    match bearer_token(api_key, env_lookup) {
        Some(token) => bearer_environment(headers, &token),
        None => sigv4_environment(
            headers,
            bedrock_region(model, litellm_params, env_lookup),
            &litellm_params.aws,
            env_lookup,
        ),
    }
}

/// Python's `_move_betas_into_header`: the betas the Invoke transform put in the body's
/// `anthropic_beta` list become one header.
fn move_betas_into_header(request: &MessagesRequest, headers: Headers) -> Result<Headers, Error> {
    let betas: Option<Vec<String>> = request
        .params
        .extra
        .get("anthropic_beta")
        .map(Vec::<String>::deserialize)
        .transpose()
        .map_err(|error| Error::InvalidRequest(ErrorDetail::invalid("anthropic_beta", error)))?;
    let headers = without_exact(headers, "anthropic-beta");
    Ok(match betas {
        Some(betas) if !betas.is_empty() => headers
            .into_iter()
            .chain([("anthropic-beta".to_string(), betas.join(","))])
            .collect(),
        _ => headers,
    })
}

/// Python's `bedrock_user_profile_fields` with the signing strip: Mantle takes the profile id as a body
/// field, so every casing of the header leaves the headers and a non-empty first value is returned.
fn take_user_profile_id(headers: Headers) -> (Option<String>, Headers) {
    let (profile, headers): (Headers, Headers) = headers
        .into_iter()
        .partition(|(name, _)| name.eq_ignore_ascii_case(USER_PROFILE_ID_HEADER));
    let profile_id = profile
        .into_iter()
        .next()
        .map(|(_, value)| value)
        .filter(|value| !value.is_empty());
    (profile_id, headers)
}

/// Python's `AmazonMantleMessagesConfig.transform_anthropic_messages_request` up to the body: the
/// Invoke transform runs on `model_id` with the Mantle overrides, then the betas move into the
/// header and the user profile id into the body. The request keeps its own model, which `wire_body`
/// turns into `model_id`.
fn transform_mantle_messages_request(
    request: MessagesRequest,
    model_id: String,
    headers: Headers,
    context: &MessagesTransformContext,
    settings: &InvokeRequestSettings,
) -> Result<(MessagesRequest, Headers), Error> {
    let model = request.model.clone();
    let (request, headers) = transform_anthropic_messages_request(
        MessagesRequest {
            model: model_id,
            ..request
        },
        &context.warning_dropped_disabled_thinking(),
        headers,
        settings,
    )?;
    let headers = move_betas_into_header(&request, headers)?;
    let (profile_id, headers) = take_user_profile_id(headers);
    let stream = request
        .params
        .stream
        .filter(|stream| *stream == Recognized::Known(true));
    let extra = request
        .params
        .extra
        .into_iter()
        .chain(profile_id.map(|id| (USER_PROFILE_ID_FIELD.to_string(), Value::String(id))))
        .collect();
    Ok((
        MessagesRequest {
            model,
            params: MessagesOptionalParams {
                stream,
                extra,
                ..request.params
            },
            ..request
        },
        headers,
    ))
}

/// The body half of Python's Mantle transform. A body that is not a Mantle request object is sent as it is.
fn mantle_wire_body(body: Value, model_id: impl Fn(&str) -> String) -> Value {
    let Ok(wire) = MantleWireBody::deserialize(&body) else {
        return body;
    };
    let wire = MantleWireBody {
        model: wire
            .model
            .map(|model| model_id(&model))
            .filter(|model| !model.is_empty()),
        stream: wire
            .stream
            .filter(|stream| matches!(stream, StreamFlag::Flag(true))),
        ..wire
    };
    serde_json::to_value(wire).unwrap_or(body)
}

fn unbillable(reason: impl ToString) -> Error {
    Error::InvalidResponse(ErrorDetail::UnbillableUsage {
        reason: reason.to_string(),
    })
}

/// An absent or null `usage` bills zero tokens, an object gets its missing counts zero-filled, and
/// anything else is refused.
fn billed_usage(usage: Option<Value>) -> Result<BilledUsage, Error> {
    let usage = match usage {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(usage)) => usage,
        Some(other) => {
            return Err(unbillable(format!("got {}", json_type_name(&other))));
        }
    };
    let billed: BilledUsage = serde_json::from_value(Value::Object(usage)).map_err(unbillable)?;
    Ok(BilledUsage {
        input_tokens: Some(billed.input_tokens.unwrap_or(0)),
        output_tokens: Some(billed.output_tokens.unwrap_or(0)),
        ..billed
    })
}

fn usage_value(usage: BilledUsage) -> Result<Value, Error> {
    serde_json::to_value(usage).map_err(unbillable)
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

impl MessagesEndpoint for BedrockMantleMessagesConfig {
    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        litellm_params: &LitellmParams,
        _stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let api_base = api_base.or(litellm_params.api_base.as_deref());
        match self.route {
            MantleRoute::BedrockMantle => {
                build_mantle_native_messages_url(api_base, model, litellm_params, env_lookup)
                    .map(String::from)
            }
            MantleRoute::Bedrock => {
                bedrock_route_messages_url(api_base, model, litellm_params, env_lookup)
            }
        }
    }
}

impl MessagesAuth for BedrockMantleMessagesConfig {
    fn secret_names(&self) -> &'static [&'static str] {
        static BEDROCK_MANTLE_NAMES: LazyLock<Box<[&'static str]>> =
            LazyLock::new(|| mantle_secret_names(&BEDROCK_MANTLE_SECRET_NAMES));
        static BEDROCK_ROUTE_NAMES: LazyLock<Box<[&'static str]>> =
            LazyLock::new(|| mantle_secret_names(&BEDROCK_ROUTE_SECRET_NAMES));
        match self.route {
            MantleRoute::BedrockMantle => &BEDROCK_MANTLE_NAMES,
            MantleRoute::Bedrock => &BEDROCK_ROUTE_NAMES,
        }
    }

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        model: &str,
        _optional_params: &MessagesOptionalParams,
        litellm_params: &LitellmParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let headers = mantle_headers(headers, litellm_params)?;
        match self.route {
            MantleRoute::BedrockMantle => {
                bedrock_mantle_environment(headers, api_key, model, litellm_params, env_lookup)
            }
            MantleRoute::Bedrock => Ok(bedrock_route_environment(
                headers,
                api_key,
                model,
                litellm_params,
                env_lookup,
            )),
        }
    }
}

impl MessagesDialect for BedrockMantleMessagesConfig {
    fn shape_request(
        &self,
        request: MessagesRequest,
        options: MessagesShapeOptions,
    ) -> Result<MessagesRequest, Error> {
        ANTHROPIC_MESSAGES_CONFIG.shape_request(request, options)
    }

    fn clamp_requested_effort(
        &self,
        request: MessagesRequest,
        capabilities: &MessagesModelCapabilities,
    ) -> Result<MessagesRequest, Error> {
        clamp_adaptive_reasoning_effort_for_bedrock(request, capabilities)
    }

    fn logged_params(
        &self,
        params: MessagesOptionalParams,
        context: &MessagesTransformContext,
    ) -> Result<MessagesOptionalParams, Error> {
        with_logged_tools(params, context)
    }

    fn native_output_fallback(
        &self,
        request: &MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Option<MessagesTransformContext> {
        native_output_fallback_litellm_params(request, context)
    }

    fn request_body(
        &self,
        request: MessagesRequest,
        headers: &Headers,
        _url: &str,
        drop_params: bool,
    ) -> Result<MessagesRequest, Error> {
        drop_sampling_params_removed_by_requested_version(request, headers, drop_params)
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    fn finalize_request(
        &self,
        request: MessagesRequest,
        headers: Headers,
        context: &MessagesTransformContext,
    ) -> Result<(MessagesRequest, Headers), Error> {
        let model_id = self.route.model_id(&request.model);
        transform_mantle_messages_request(
            request,
            model_id,
            headers,
            context,
            &self.request_settings,
        )
    }

    fn wire_body(&self, body: Value) -> Value {
        mantle_wire_body(body, |model| self.route.model_id(model))
    }

    fn logged_model(&self, model: &str) -> String {
        match self.route {
            MantleRoute::BedrockMantle => split_mantle_region_prefix(model).1.to_string(),
            MantleRoute::Bedrock => model.to_string(),
        }
    }
}

impl MessagesReply for BedrockMantleMessagesConfig {
    fn transform_anthropic_messages_response(
        &self,
        _model: &str,
        response: MessagesResponse,
    ) -> Result<MessagesResponse, Error> {
        let usage = usage_value(billed_usage(response.usage.clone())?)?;
        Ok(MessagesResponse {
            usage: Some(usage),
            ..response
        })
    }

    fn transform_unchecked_response(
        &self,
        mut body: Map<String, Value>,
    ) -> Result<Map<String, Value>, Error> {
        let usage = usage_value(billed_usage(body.remove("usage"))?)?;
        body.insert("usage".to_string(), usage);
        Ok(body)
    }

    fn retries_without_native_output(
        &self,
        status: u16,
        error_text: &str,
        request_body: &Value,
        model: &str,
    ) -> bool {
        should_retry_anthropic_messages_without_native_output(
            status,
            error_text,
            request_body,
            model,
        )
    }

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

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::base_llm::messages::transformation::BaseMessagesConfig;

    #[rstest]
    #[case::bedrock_mantle_region_prefix(
        "bedrock_mantle",
        "bedrock_mantle/us-west-2/anthropic.claude-sonnet-5-5",
        Some(("bedrock_mantle", "anthropic.claude-sonnet-5-5"))
    )]
    #[case::bedrock_mantle_route(
        "bedrock",
        "bedrock/mantle/anthropic.claude-sonnet-5-5",
        Some(("bedrock", "anthropic.claude-sonnet-5-5"))
    )]
    #[case::invoke("bedrock", "bedrock/us.anthropic.claude-sonnet-5-5", None)]
    #[case::claude_platform("bedrock", "bedrock/claude_platform/mantle/claude-sonnet-5-5", None)]
    #[case::converse("bedrock", "bedrock/converse/mantle/claude-sonnet-5-5", None)]
    #[case::other_provider("anthropic", "mantle/claude-sonnet-5-5", None)]
    fn mantle_capabilities_are_looked_up_for_the_model_python_transforms(
        #[case] provider: &str,
        #[case] model: &str,
        #[case] expected: Option<(&'static str, &str)>,
    ) {
        assert_eq!(
            mantle_capability_lookup(provider, model),
            expected.map(|(provider, model)| MantleCapabilityLookup {
                provider,
                model: model.to_string(),
            })
        );
    }

    #[rstest]
    #[case::bedrock_mantle(
        "bedrock_mantle",
        "bedrock_mantle/anthropic.claude-sonnet-5-5",
        Some("anthropic.claude-sonnet-5-5")
    )]
    #[case::bedrock_mantle_region_prefix(
        "bedrock_mantle",
        "bedrock_mantle/us-west-2/anthropic.claude-sonnet-5-5",
        Some("anthropic.claude-sonnet-5-5")
    )]
    #[case::bedrock_mantle_keeps_the_mantle_segment(
        "bedrock_mantle",
        "bedrock_mantle/us-west-2/mantle/anthropic.claude-sonnet-5-5",
        Some("mantle/anthropic.claude-sonnet-5-5")
    )]
    #[case::bedrock_mantle_unprefixed(
        "bedrock_mantle",
        "us-west-2/anthropic.claude-sonnet-5-5",
        Some("anthropic.claude-sonnet-5-5")
    )]
    #[case::bedrock_mantle_route(
        "bedrock",
        "bedrock/mantle/anthropic.claude-sonnet-5-5",
        Some("mantle/anthropic.claude-sonnet-5-5")
    )]
    #[case::invoke("bedrock", "bedrock/us.anthropic.claude-sonnet-5-5", None)]
    #[case::other_provider("anthropic", "mantle/claude-sonnet-5-5", None)]
    fn the_logged_model_is_the_one_python_resolves(
        #[case] provider: &str,
        #[case] model: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(mantle_logged_model(provider, model).as_deref(), expected);
    }

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        }
    }

    fn params(api_base: Option<&str>, region: Option<&str>) -> LitellmParams {
        LitellmParams {
            api_base: api_base.map(str::to_string),
            aws: AwsParams {
                aws_region_name: region.map(str::to_string),
                ..AwsParams::default()
            },
            ..LitellmParams::default()
        }
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn environment(
        config: &dyn BaseMessagesConfig,
        forwarded: Headers,
        api_key: Option<&str>,
        model: &str,
        litellm_params: &LitellmParams,
        env_pairs: &'static [(&'static str, &'static str)],
    ) -> Result<ValidatedEnvironment, Error> {
        config.validate_environment(
            forwarded,
            api_key,
            model,
            &MessagesOptionalParams::default(),
            litellm_params,
            &env(env_pairs),
        )
    }

    fn signing_region(environment: &ValidatedEnvironment) -> Option<&str> {
        match &environment.auth {
            AuthScheme::AwsSigV4 { region, .. } => Some(region),
            _ => None,
        }
    }

    fn sent(
        config: &dyn BaseMessagesConfig,
        request: Value,
        forwarded: Headers,
    ) -> Result<(Value, Headers), Error> {
        let request: MessagesRequest = serde_json::from_value(request).unwrap();
        let context = MessagesTransformContext::new(MessagesModelCapabilities::default(), false);
        let (request, headers) = config.finalize_request(request, forwarded, &context)?;
        Ok((
            config.wire_body(serde_json::to_value(&request).unwrap()),
            headers,
        ))
    }

    fn request(model: &str, extra: Value) -> Value {
        let base = json!({
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        });
        let fields: Map<String, Value> = base
            .as_object()
            .unwrap()
            .clone()
            .into_iter()
            .chain(extra.as_object().unwrap().clone())
            .collect();
        Value::Object(fields)
    }

    fn header<'a>(headers: &'a Headers, name: &str) -> Vec<&'a str> {
        headers
            .iter()
            .filter(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
            .collect()
    }

    #[rstest]
    #[case::root("https://private.test", "https://private.test/anthropic/v1/messages")]
    #[case::openai(
        "https://private.test/openai/v1/",
        "https://private.test/anthropic/v1/messages"
    )]
    #[case::complete(
        "https://private.test/anthropic/v1/messages/",
        "https://private.test/anthropic/v1/messages"
    )]
    #[case::tenant(
        "https://private.test/tenant/v1",
        "https://private.test/tenant/anthropic/v1/messages"
    )]
    #[case::port_kept(
        "https://bedrock-mantle.us-west-2.api.aws:8443/v1",
        "https://bedrock-mantle.us-west-2.api.aws:8443/anthropic/v1/messages"
    )]
    #[case::public_host_rewritten_to_https(
        "http://bedrock-mantle.us-west-2.api.aws/openai/v1",
        "https://bedrock-mantle.us-west-2.api.aws/anthropic/v1/messages"
    )]
    #[case::blank_base_is_unset(
        "   ",
        "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"
    )]
    #[case::query_stays_after_the_route(
        "https://private.test/openai/v1?tenant=a",
        "https://private.test/anthropic/v1/messages?tenant=a"
    )]
    #[case::public_host_keeps_the_query(
        "https://bedrock-mantle.us-west-2.api.aws/v1?tenant=a",
        "https://bedrock-mantle.us-west-2.api.aws/anthropic/v1/messages?tenant=a"
    )]
    fn api_base_suffixes_are_stripped_once(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(
            BEDROCK_MANTLE_MESSAGES_CONFIG
                .get_complete_url(
                    Some(base),
                    "claude-test",
                    &LitellmParams::default(),
                    true,
                    &|_| None
                )
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::default(None, None, "claude-test", &[], "us-east-1")]
    #[case::aws_default_region_ignored(None, None, "claude-test", &[("AWS_DEFAULT_REGION", "eu-west-1")], "us-east-1")]
    #[case::mantle_region_env(None, None, "claude-test", &[("BEDROCK_MANTLE_REGION", "eu-west-1"), ("AWS_REGION", "ap-south-1")], "eu-west-1")]
    #[case::model_prefix_over_env(None, None, "us-west-2/claude-test", &[("AWS_REGION", "eu-west-1")], "us-west-2")]
    #[case::configured_region_over_prefix(None, Some("eu-west-1"), "us-west-2/claude-test", &[], "eu-west-1")]
    #[case::configured_region_over_public_host(Some("https://bedrock-mantle.us-west-2.api.aws"), Some("us-east-1"), "claude-test", &[], "us-east-1")]
    #[case::public_host_over_prefix(Some("https://bedrock-mantle.ap-south-1.api.aws/v1"), None, "us-west-2/claude-test", &[], "ap-south-1")]
    #[case::env_base_host(None, None, "claude-test", &[("BEDROCK_MANTLE_API_BASE", "https://bedrock-mantle.eu-central-1.api.aws/v1")], "eu-central-1")]
    #[case::arn_ignored(None, None, "arn:aws:bedrock:eu-west-1:123:model/claude-test", &[], "us-east-1")]
    #[case::unlisted_region_prefix_ignored(None, None, "ca-west-1/claude-test", &[], "us-east-1")]
    fn public_host_region_follows_python_order(
        #[case] api_base: Option<&str>,
        #[case] region: Option<&str>,
        #[case] model: &str,
        #[case] env_pairs: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        assert_eq!(
            BEDROCK_MANTLE_MESSAGES_CONFIG
                .get_complete_url(
                    None,
                    model,
                    &params(api_base, region),
                    false,
                    &env(env_pairs)
                )
                .unwrap(),
            format!("https://bedrock-mantle.{expected}.api.aws/anthropic/v1/messages")
        );
    }

    #[rstest]
    fn runtime_endpoint_is_ignored() {
        let litellm_params = LitellmParams {
            aws: AwsParams {
                aws_bedrock_runtime_endpoint: Some("https://runtime.test".into()),
                ..AwsParams::default()
            },
            ..LitellmParams::default()
        };
        assert_eq!(
            BEDROCK_MANTLE_MESSAGES_CONFIG
                .get_complete_url(None, "claude-test", &litellm_params, false, &|_| None)
                .unwrap(),
            "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"
        );
    }

    #[rstest]
    #[case::private_base_ignores_model_prefix(Some("https://private.test"), None, "us-west-2/claude-test", &[], "us-east-1")]
    #[case::private_base_uses_env(Some("https://private.test"), None, "claude-test", &[("AWS_REGION_NAME", "eu-west-1")], "eu-west-1")]
    #[case::public_host_pinned_to_configured_region(Some("https://bedrock-mantle.us-west-2.api.aws"), Some("us-east-1"), "claude-test", &[], "us-east-1")]
    #[case::model_prefix_default_host(None, None, "us-west-2/claude-test", &[], "us-west-2")]
    fn sigv4_signs_with_the_url_region(
        #[case] api_base: Option<&str>,
        #[case] region: Option<&str>,
        #[case] model: &str,
        #[case] env_pairs: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        let validated = environment(
            &BEDROCK_MANTLE_MESSAGES_CONFIG,
            vec![],
            None,
            model,
            &params(api_base, region),
            env_pairs,
        )
        .unwrap();
        assert_eq!(signing_region(&validated), Some(expected));
    }

    #[rstest]
    fn upper_case_public_host_is_sent_and_signed_lowercased() {
        let litellm_params = params(Some("https://BEDROCK-MANTLE.US-EAST-2.API.AWS"), None);
        assert_eq!(
            BEDROCK_MANTLE_MESSAGES_CONFIG
                .get_complete_url(None, "claude-test", &litellm_params, false, &|_| None)
                .unwrap(),
            "https://bedrock-mantle.us-east-2.api.aws/anthropic/v1/messages"
        );
        let validated = environment(
            &BEDROCK_MANTLE_MESSAGES_CONFIG,
            vec![],
            None,
            "claude-test",
            &litellm_params,
            &[],
        )
        .unwrap();
        assert_eq!(signing_region(&validated), Some("us-east-2"));
    }

    #[rstest]
    #[case::blank("   ", None)]
    #[case::set("key", Some("Bearer key"))]
    fn a_blank_api_key_counts_as_unset(#[case] api_key: &str, #[case] expected: Option<&str>) {
        let validated = environment(
            &BEDROCK_MANTLE_MESSAGES_CONFIG,
            vec![],
            Some(api_key),
            "claude-test",
            &params(None, Some("us-east-1")),
            &[],
        )
        .unwrap();
        assert_eq!(
            header(&validated.headers, "Authorization").first().copied(),
            expected
        );
        assert_eq!(signing_region(&validated).is_some(), expected.is_none());
    }

    #[rstest]
    #[case::url_injection(&BEDROCK_MANTLE_MESSAGES_CONFIG, Some("us-east-1.attacker.test/steal?"), &[], MantleRegionCheck::ProviderResolution)]
    #[case::uppercase(&BEDROCK_MANTLE_MESSAGES_CONFIG, Some("US-EAST-1"), &[], MantleRegionCheck::ProviderResolution)]
    #[case::environment(&BEDROCK_MANTLE_MESSAGES_CONFIG, None, &[("BEDROCK_MANTLE_REGION", "US-EAST-1")], MantleRegionCheck::ProviderResolution)]
    #[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG, Some("US-EAST-1"), &[], MantleRegionCheck::Request)]
    #[case::bedrock_mantle_route_environment(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG, None, &[("AWS_REGION_NAME", "US-EAST-1")], MantleRegionCheck::Request)]
    fn invalid_regions_name_where_python_refuses_them(
        #[case] config: &dyn BaseMessagesConfig,
        #[case] configured: Option<&str>,
        #[case] env: &'static [(&'static str, &'static str)],
        #[case] expected: MantleRegionCheck,
    ) {
        let lookup = |name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        };
        let refused = config.get_complete_url(
            None,
            "claude-test",
            &params(None, configured),
            false,
            &lookup,
        );
        let Err(Error::InvalidRequest(detail @ ErrorDetail::InvalidMantleRegion { check, .. })) =
            &refused
        else {
            panic!("expected an invalid region, got {refused:?}");
        };
        assert_eq!(*check, expected);
        let region = configured.unwrap_or_else(|| env[0].1);
        assert_eq!(
            detail.to_string(),
            format!(
                "Invalid AWS region format: '{region}'. Region names must contain only lowercase letters, digits, and hyphens."
            )
        );
    }

    #[rstest]
    #[case::plain("US-EAST-1", "'US-EAST-1'")]
    #[case::newline("us-east-1\nx", "'us-east-1\\nx'")]
    fn an_invalid_region_is_quoted_with_control_characters_escaped(
        #[case] region: &str,
        #[case] quoted: &str,
    ) {
        let refused = BEDROCK_MANTLE_MESSAGES_CONFIG.get_complete_url(
            None,
            "claude-test",
            &params(None, Some(region)),
            false,
            &|_: &str| None,
        );
        let Err(Error::InvalidRequest(detail)) = &refused else {
            panic!("expected an invalid region, got {refused:?}");
        };
        assert_eq!(
            detail.to_string(),
            format!(
                "Invalid AWS region format: {quoted}. Region names must contain only lowercase letters, digits, and hyphens."
            )
        );
    }

    #[rstest]
    #[case::explicit(Some("explicit"), "Bearer explicit")]
    #[case::mantle_key(None, "Bearer mantle")]
    fn bearer_prefers_the_mantle_key(#[case] api_key: Option<&str>, #[case] expected: &str) {
        let validated = environment(
            &BEDROCK_MANTLE_MESSAGES_CONFIG,
            vec![],
            api_key,
            "claude-test",
            &LitellmParams::default(),
            &[
                ("BEDROCK_MANTLE_API_KEY", "mantle"),
                ("AWS_BEARER_TOKEN_BEDROCK", "aws"),
            ],
        )
        .unwrap();
        assert_eq!(header(&validated.headers, "Authorization"), vec![expected]);
    }

    #[rstest]
    #[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG)]
    #[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG)]
    fn bearer_overwrites_only_the_exact_authorization_key(#[case] config: &dyn BaseMessagesConfig) {
        let validated = environment(
            config,
            headers(&[
                ("authorization", "stale"),
                ("Authorization", "old"),
                ("x-api-key", "caller"),
            ]),
            Some("token"),
            "claude-test",
            &LitellmParams::default(),
            &[],
        )
        .unwrap();
        assert_eq!(
            (
                header(&validated.headers, "authorization"),
                header(&validated.headers, "Authorization"),
                header(&validated.headers, "x-api-key"),
                header(&validated.headers, "Content-Type"),
            ),
            (
                vec!["stale"],
                vec!["Bearer token"],
                vec!["caller"],
                vec!["application/json"]
            )
        );
    }

    #[rstest]
    #[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG, "us-west-2/anthropic.claude-test")]
    #[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG, "mantle/anthropic.claude-test")]
    fn the_invoke_transform_sees_the_model_python_sends(
        #[case] config: &dyn BaseMessagesConfig,
        #[case] model: &str,
    ) {
        let Err(Error::InvalidRequest(refused)) = sent(
            config,
            request(
                model,
                json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}),
            ),
            Headers::new(),
        ) else {
            panic!("expected a request rejection");
        };
        assert!(
            refused
                .to_string()
                .contains("model 'anthropic.claude-test'"),
            "{refused}"
        );
    }

    #[rstest]
    fn sigv4_drops_every_authorization_casing_and_keeps_x_api_key() {
        let validated = environment(
            &BEDROCK_MANTLE_MESSAGES_CONFIG,
            headers(&[
                ("authorization", "stale"),
                ("Authorization", "Bearer x"),
                ("x-api-key", "caller"),
            ]),
            None,
            "claude-test",
            &LitellmParams::default(),
            &[],
        )
        .unwrap();
        assert!(
            !validated
                .headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        );
        assert_eq!(header(&validated.headers, "x-api-key"), vec!["caller"]);
    }

    #[rstest]
    #[case::absent(&[], vec!["2023-06-01"], vec!["application/json"])]
    #[case::caller_casing_kept(&[("Anthropic-Version", "2024-01-01"), ("content-type", "text/plain")], vec![], vec![])]
    fn version_and_content_type_defaults_respect_any_casing(
        #[case] forwarded: &[(&str, &str)],
        #[case] version: Vec<&str>,
        #[case] content_type: Vec<&str>,
    ) {
        let validated = environment(
            &BEDROCK_MANTLE_MESSAGES_CONFIG,
            headers(forwarded),
            None,
            "claude-test",
            &LitellmParams::default(),
            &[],
        )
        .unwrap();
        assert_eq!(
            (
                header(&validated.headers, "anthropic-version"),
                header(&validated.headers, "Content-Type"),
            ),
            (version, content_type)
        );
    }

    #[rstest]
    #[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG, true)]
    #[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG, false)]
    fn clear_thinking_edit_is_kept_only_on_bedrock_mantle(
        #[case] config: &dyn BaseMessagesConfig,
        #[case] kept: bool,
    ) {
        let (body, headers) = sent(
            config,
            request(
                "claude-test",
                json!({"context_management": {"edits": [{"type": "clear_thinking_20251015"}]}}),
            ),
            vec![],
        )
        .unwrap();
        assert_eq!(body.get("context_management").is_some(), kept);
        assert_eq!(
            header(&headers, "anthropic-beta"),
            if kept {
                vec!["context-management-2025-06-27"]
            } else {
                vec![]
            }
        );
    }

    #[rstest]
    #[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG)]
    #[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG)]
    fn mantle_logs_the_tools_the_invoke_transform_shapes(#[case] config: &dyn BaseMessagesConfig) {
        let request: MessagesRequest = serde_json::from_value(request(
            "claude-test",
            json!({"tools": [{
                "type": "custom",
                "name": "",
                "description": "d",
                "input_schema": {"type": "custom"},
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }]}),
        ))
        .unwrap();
        let context = MessagesTransformContext::new(MessagesModelCapabilities::default(), false);

        let logged = config.logged_params(request.params, &context).unwrap();

        assert_eq!(
            serde_json::to_value(logged.tools).unwrap(),
            json!([{
                "type": "custom",
                "name": "litellm_unnamed_tool_0",
                "description": "d",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"},
            }])
        );
    }

    #[rstest]
    #[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG)]
    #[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG)]
    fn mantle_shapes_the_body_like_invoke_with_its_overrides(
        #[case] config: &dyn BaseMessagesConfig,
    ) {
        let (body, headers) = sent(
            config,
            request(
                "claude-test",
                json!({
                    "compaction": {"trigger": 1000},
                    "service_tier": "auto",
                    "inference_geo": "us",
                    "stream": false,
                }),
            ),
            headers(&[(
                "anthropic-beta",
                "custom-beta,dangerous-tool-use-2026-09-03",
            )]),
        )
        .unwrap();
        assert_eq!(
            body.as_object().unwrap().keys().collect::<Vec<_>>(),
            vec!["messages", "max_tokens", "service_tier", "model"]
        );
        assert_eq!(
            header(&headers, "anthropic-beta"),
            vec!["dangerous-tool-use-2026-09-03"]
        );
    }

    #[rstest]
    #[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG)]
    #[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG)]
    fn null_safeguards_add_no_dangerous_tool_use_beta(#[case] config: &dyn BaseMessagesConfig) {
        let (_, headers) = sent(
            config,
            request("claude-test", json!({"safeguards": null})),
            vec![],
        )
        .unwrap();
        assert!(header(&headers, "anthropic-beta").is_empty());
    }

    #[rstest]
    #[case::lowercase(&[("anthropic-user-profile-id", "uprof_caller")], Some("uprof_caller"))]
    #[case::mixed_case(&[("Anthropic-User-Profile-Id", "uprof_caller")], Some("uprof_caller"))]
    #[case::empty(&[("anthropic-user-profile-id", "")], None)]
    #[case::absent(&[], None)]
    fn the_user_profile_header_becomes_the_body_field(
        #[values(&BEDROCK_MANTLE_MESSAGES_CONFIG, &BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG)]
        config: &dyn BaseMessagesConfig,
        #[case] profile: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        let forwarded = profile
            .iter()
            .chain(&[("x-other", "keep")])
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        let (body, headers) = sent(config, request("claude-test", json!({})), forwarded).unwrap();
        assert_eq!(
            body.get("user_profile_id").and_then(Value::as_str),
            expected
        );
        assert!(
            !headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("anthropic-user-profile-id"))
        );
        assert_eq!(header(&headers, "x-other"), vec!["keep"]);
    }

    #[rstest]
    #[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG)]
    #[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG)]
    fn server_web_search_is_rejected(#[case] config: &dyn BaseMessagesConfig) {
        assert!(matches!(
            sent(
                config,
                request(
                    "claude-test",
                    json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]})
                ),
                vec![]
            ),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[rstest]
    #[case::bedrock_mantle_strips_region(&BEDROCK_MANTLE_MESSAGES_CONFIG, "us-west-2/claude-test", true, json!({"model": "claude-test", "stream": true}))]
    #[case::bedrock_mantle_route_keeps_region(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG, "mantle/eu-west-1/anthropic.claude-test", true, json!({"model": "eu-west-1/anthropic.claude-test", "stream": true}))]
    #[case::stream_false_is_omitted(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG, "mantle/anthropic.claude-test", false, json!({"model": "anthropic.claude-test"}))]
    fn wire_body_sends_model_and_stream_last(
        #[case] config: &dyn BaseMessagesConfig,
        #[case] model: &str,
        #[case] stream: bool,
        #[case] expected: Value,
    ) {
        let wire = config.wire_body(json!({
            "model": model,
            "stream": stream,
            "anthropic_version": "sdk",
            "anthropic_beta": ["sdk"],
        }));
        assert_eq!(wire, expected);
        assert_eq!(
            wire.as_object()
                .unwrap()
                .keys()
                .next_back()
                .map(String::as_str),
            Some(if stream { "stream" } else { "model" })
        );
    }

    #[rstest]
    #[case::string_stream_dropped(json!({"model": "mantle/m", "stream": "true", "max_tokens": 1}), json!({"max_tokens": 1, "model": "m"}))]
    #[case::not_an_object(json!(["mantle/m"]), json!(["mantle/m"]))]
    fn wire_body_keeps_only_a_boolean_true_stream(#[case] body: Value, #[case] expected: Value) {
        assert_eq!(
            BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG.wire_body(body),
            expected
        );
    }

    #[rstest]
    #[case::missing(None, json!({"input_tokens": 0, "output_tokens": 0}))]
    #[case::null(Some(json!(null)), json!({"input_tokens": 0, "output_tokens": 0}))]
    #[case::partial(Some(json!({"output_tokens": 7, "cache_read_input_tokens": 3})), json!({"input_tokens": 0, "output_tokens": 7, "cache_read_input_tokens": 3}))]
    fn response_usage_is_zero_filled(#[case] usage: Option<Value>, #[case] expected: Value) {
        let usage = BEDROCK_MANTLE_MESSAGES_CONFIG
            .transform_anthropic_messages_response("claude-test", replied_usage(usage))
            .unwrap()
            .usage
            .unwrap();
        assert_eq!(usage, expected);
        assert_eq!(
            usage
                .as_object()
                .unwrap()
                .keys()
                .take(2)
                .collect::<Vec<_>>(),
            vec!["input_tokens", "output_tokens"]
        );
    }

    #[rstest]
    #[case::bedrock_mantle_array(&BEDROCK_MANTLE_MESSAGES_CONFIG, json!([1]), "an array")]
    #[case::bedrock_mantle_empty_string(&BEDROCK_MANTLE_MESSAGES_CONFIG, json!(""), "a string")]
    #[case::bedrock_mantle_route_number(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG, json!(0), "a number")]
    fn a_non_object_usage_is_refused_instead_of_billed_as_zero(
        #[case] config: &dyn BaseMessagesConfig,
        #[case] usage: Value,
        #[case] actual: &str,
    ) {
        let refused =
            config.transform_anthropic_messages_response("claude-test", replied_usage(Some(usage)));
        let Err(Error::InvalidResponse(ErrorDetail::UnbillableUsage { reason })) = refused else {
            panic!("expected a refused usage, got {refused:?}");
        };
        assert_eq!(reason, format!("got {actual}"));
    }

    fn replied_usage(usage: Option<Value>) -> MessagesResponse {
        let response: MessagesResponse = serde_json::from_value(json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [],
            "stop_reason": "end_turn",
            "stop_sequence": null,
        }))
        .unwrap();
        MessagesResponse { usage, ..response }
    }

    struct Warnings(std::sync::mpsc::Sender<String>);

    impl litellm_tracing::Sink for Warnings {
        fn enabled(&self, metadata: &litellm_tracing::Metadata<'_>) -> bool {
            *metadata.level() == litellm_tracing::Level::WARN
        }

        fn emit(&self, record: &litellm_tracing::Record) {
            self.0.send(record.message.clone()).unwrap();
        }
    }

    #[rstest]
    #[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG, true)]
    #[case::bedrock_mantle_route(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG, true)]
    #[case::bedrock_invoke(&crate::bedrock::messages::invoke::transformation::BEDROCK_INVOKE_MESSAGES_CONFIG, false)]
    #[case::anthropic(&crate::anthropic::messages::transformation::ANTHROPIC_MESSAGES_CONFIG, false)]
    fn dropping_disabled_thinking_warns_only_on_mantle(
        #[case] config: &dyn BaseMessagesConfig,
        #[case] warns: bool,
    ) {
        let (sender, receiver) = std::sync::mpsc::channel();
        let request: MessagesRequest = serde_json::from_value(request(
            "anthropic.claude-test",
            json!({"thinking": {"type": "disabled"}}),
        ))
        .unwrap();
        let context = MessagesTransformContext::new(
            MessagesModelCapabilities {
                thinking_always_on: true,
                ..MessagesModelCapabilities::default()
            },
            false,
        );
        let (sent, _) = litellm_tracing::Logger::new(Warnings(sender)).scope(|| {
            let request = config
                .transform_anthropic_messages_request(request, &context)
                .unwrap();
            config.finalize_request(request, vec![], &context).unwrap()
        });
        let warned = receiver.try_iter().any(|message| {
            message.starts_with(
                "Dropping `thinking={'type': 'disabled'}` for model=anthropic.claude-test:",
            ) && message.ends_with("billed as output tokens.")
        });
        assert_eq!((sent.params.thinking, warned), (None, warns));
    }

    fn route_params(region: Option<&str>, runtime_endpoint: Option<&str>) -> LitellmParams {
        LitellmParams {
            aws: AwsParams {
                aws_region_name: region.map(str::to_string),
                aws_bedrock_runtime_endpoint: runtime_endpoint.map(str::to_string),
                ..AwsParams::default()
            },
            ..LitellmParams::default()
        }
    }

    fn route_url(
        api_base: Option<&str>,
        model: &str,
        litellm_params: &LitellmParams,
        env_pairs: &'static [(&'static str, &'static str)],
    ) -> String {
        BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG
            .get_complete_url(api_base, model, litellm_params, false, &env(env_pairs))
            .unwrap()
    }

    fn route_environment(
        forwarded: &[(&str, &str)],
        api_key: Option<&str>,
        model: &str,
        litellm_params: &LitellmParams,
        env_pairs: &'static [(&'static str, &'static str)],
    ) -> ValidatedEnvironment {
        BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG
            .validate_environment(
                forwarded
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.to_string()))
                    .collect(),
                api_key,
                model,
                &MessagesOptionalParams::default(),
                litellm_params,
                &env(env_pairs),
            )
            .unwrap()
    }

    #[rstest]
    #[case::v1_suffix_kept(Some("https://x.test/v1"), None, &[], "https://x.test/v1/anthropic/v1/messages")]
    #[case::query_after_full_path_kept(
        Some("https://x.test/anthropic/v1/messages?x=1"),
        None,
        &[],
        "https://x.test/anthropic/v1/messages?x=1"
    )]
    #[case::query_stays_after_the_route(
        Some("https://x.test/v1?tenant=a"),
        None,
        &[],
        "https://x.test/v1/anthropic/v1/messages?tenant=a"
    )]
    #[case::blank_base_is_unset(
        Some("   "),
        Some("https://runtime.test"),
        &[],
        "https://runtime.test/anthropic/v1/messages"
    )]
    #[case::upper_case_host_lowercased(
        Some("https://BEDROCK-MANTLE.US-EAST-2.API.AWS"),
        None,
        &[],
        "https://bedrock-mantle.us-east-2.api.aws/anthropic/v1/messages"
    )]
    #[case::public_host_never_rewritten(
        Some("https://bedrock-mantle.us-west-2.api.aws/custom"),
        None,
        &[],
        "https://bedrock-mantle.us-west-2.api.aws/custom/anthropic/v1/messages"
    )]
    #[case::port_kept(
        Some("https://bedrock-mantle.us-west-2.api.aws:8443"),
        None,
        &[],
        "https://bedrock-mantle.us-west-2.api.aws:8443/anthropic/v1/messages"
    )]
    #[case::runtime_endpoint(None, Some("https://runtime.test/v1"), &[], "https://runtime.test/v1/anthropic/v1/messages")]
    #[case::api_base_over_runtime_and_env(
        Some("https://base.test"),
        Some("https://runtime.test"),
        &[("BEDROCK_MANTLE_API_BASE", "https://env.test")],
        "https://base.test/anthropic/v1/messages"
    )]
    #[case::env_openai_suffix_stripped(
        None,
        None,
        &[("BEDROCK_MANTLE_API_BASE", "https://env.test/openai/v1/")],
        "https://env.test/anthropic/v1/messages"
    )]
    #[case::env_anthropic_suffix_loses_only_v1(
        None,
        None,
        &[("BEDROCK_MANTLE_API_BASE", "https://env.test/anthropic/v1")],
        "https://env.test/anthropic/anthropic/v1/messages"
    )]
    fn explicit_bases_follow_build_mantle_messages_url(
        #[case] api_base: Option<&str>,
        #[case] runtime_endpoint: Option<&str>,
        #[case] env_pairs: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        assert_eq!(
            route_url(
                api_base,
                "mantle/anthropic.claude-test",
                &route_params(Some("us-east-1"), runtime_endpoint),
                env_pairs
            ),
            expected
        );
    }

    #[rstest]
    #[case::mantle_region_env_ignored(
        "mantle/anthropic.claude-test",
        None,
        &[("BEDROCK_MANTLE_REGION", "eu-west-1"), ("AWS_REGION", "ap-south-1")],
        "ap-south-1"
    )]
    #[case::aws_default_region(
        "mantle/anthropic.claude-test",
        None,
        &[("AWS_DEFAULT_REGION", "eu-west-1")],
        "eu-west-1"
    )]
    #[case::region_path_beats_the_env(
        "mantle/eu-west-1/anthropic.claude-test",
        None,
        &[("AWS_REGION", "us-east-2")],
        "eu-west-1"
    )]
    #[case::region_path_after_the_bedrock_prefix(
        "bedrock/mantle/eu-west-1/anthropic.claude-test",
        None,
        &[("AWS_REGION", "us-east-2")],
        "eu-west-1"
    )]
    #[case::gov_cloud_region_path(
        "bedrock/mantle/us-gov-west-1/anthropic.claude-test",
        None,
        &[("AWS_REGION", "us-east-2")],
        "us-gov-west-1"
    )]
    #[case::aws_region_name_beats_the_region_path(
        "mantle/eu-west-1/anthropic.claude-test",
        Some("ap-south-1"),
        &[("AWS_REGION", "us-east-2")],
        "ap-south-1"
    )]
    #[case::arn_region_beats_the_region_path(
        "mantle/eu-west-1/arn:aws:bedrock:eu-central-1:123456789012:inference-profile/anthropic.claude-test",
        None,
        &[("AWS_REGION", "us-east-2")],
        "eu-central-1"
    )]
    #[case::non_region_segment_falls_through_to_the_env(
        "mantle/anthropic/claude-test",
        None,
        &[("AWS_REGION", "us-east-2")],
        "us-east-2"
    )]
    #[case::arn_region(
        "mantle/arn:aws:bedrock:eu-central-1:123456789012:inference-profile/anthropic.claude-test",
        None,
        &[("AWS_REGION", "us-east-2")],
        "eu-central-1"
    )]
    fn host_and_signing_region_follow_the_invoke_order(
        #[case] model: &str,
        #[case] aws_region_name: Option<&str>,
        #[case] env_pairs: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        let litellm_params = route_params(aws_region_name, None);
        assert_eq!(
            route_url(None, model, &litellm_params, env_pairs),
            format!("https://bedrock-mantle.{expected}.api.aws/anthropic/v1/messages")
        );
        assert!(matches!(
            route_environment(&[], None, model, &litellm_params, env_pairs).auth,
            AuthScheme::AwsSigV4 { region, .. } if region == expected
        ));
    }

    #[rstest]
    fn the_region_path_stays_in_the_body_model() {
        assert_eq!(
            amazon_mantle_model_id("mantle/eu-west-1/anthropic.claude-test"),
            "eu-west-1/anthropic.claude-test"
        );
    }

    #[rstest]
    fn default_region_is_the_invoke_default() {
        let invoke_default = resolve_bedrock_region(None, &AwsParams::default(), &|_| None);
        assert_eq!(
            route_url(
                None,
                "mantle/anthropic.claude-test",
                &LitellmParams::default(),
                &[]
            ),
            format!("https://bedrock-mantle.{invoke_default}.api.aws/anthropic/v1/messages")
        );
    }

    #[rstest]
    #[case::mantle_key_ignored(&[("BEDROCK_MANTLE_API_KEY", "mantle")], None)]
    #[case::bedrock_token(&[("AWS_BEARER_TOKEN_BEDROCK", "aws")], Some("Bearer aws"))]
    fn bearer_reads_only_the_bedrock_token(
        #[case] env_pairs: &'static [(&'static str, &'static str)],
        #[case] expected: Option<&str>,
    ) {
        let validated = route_environment(
            &[],
            None,
            "mantle/anthropic.claude-test",
            &route_params(Some("us-east-1"), None),
            env_pairs,
        );
        let authorization = validated
            .headers
            .iter()
            .find(|(name, _)| name == "Authorization")
            .map(|(_, value)| value.as_str());
        assert_eq!(authorization, expected);
        assert_eq!(
            matches!(validated.auth, AuthScheme::AwsSigV4 { .. }),
            expected.is_none()
        );
    }

    #[rstest]
    fn a_blank_api_key_counts_as_unset_on_the_bedrock_route() {
        let validated = route_environment(
            &[],
            Some("   "),
            "mantle/anthropic.claude-test",
            &route_params(Some("us-east-1"), None),
            &[("AWS_BEARER_TOKEN_BEDROCK", "aws")],
        );
        assert!(
            validated
                .headers
                .iter()
                .any(|(name, value)| name == "Authorization" && value == "Bearer aws")
        );
    }

    #[rstest]
    fn sigv4_keeps_forwarded_credentials_for_the_signer() {
        let forwarded = [("authorization", "Bearer caller"), ("x-api-key", "caller")];
        let validated = route_environment(
            &forwarded,
            None,
            "mantle/anthropic.claude-test",
            &route_params(Some("us-east-1"), None),
            &[],
        );
        assert!(forwarded.iter().all(|(name, value)| {
            validated
                .headers
                .iter()
                .any(|(key, sent)| key == name && sent == value)
        }));
        assert!(matches!(
            validated.auth,
            AuthScheme::AwsSigV4 {
                caller_headers: CallerSigningHeaders::Botocore,
                ..
            }
        ));
    }

    #[rstest]
    #[case::absent(json!({}), Ok(None))]
    #[case::empty(json!({"anthropic_beta": []}), Ok(None))]
    #[case::listed(json!({"anthropic_beta": ["a", "b"]}), Ok(Some("a,b")))]
    #[case::not_a_list(json!({"anthropic_beta": "a"}), Err(()))]
    fn betas_move_into_one_header(
        #[case] extra: Value,
        #[case] expected: Result<Option<&str>, ()>,
    ) {
        let fields = json!({"model": "m", "max_tokens": 1, "messages": []})
            .as_object()
            .unwrap()
            .clone()
            .into_iter()
            .chain(extra.as_object().unwrap().clone())
            .collect::<Map<_, _>>();
        let request: MessagesRequest = serde_json::from_value(Value::Object(fields)).unwrap();
        let moved = move_betas_into_header(
            &request,
            vec![("anthropic-beta".to_string(), "stale".to_string())],
        );
        let beta = moved.map(|headers| {
            headers
                .into_iter()
                .find(|(name, _)| name == "anthropic-beta")
                .map(|(_, value)| value)
        });
        assert_eq!(
            beta.as_ref().map(Option::as_deref).map_err(|_| ()),
            expected
        );
    }

    #[rstest]
    #[case::bedrock_mantle_under_sdk_2(&BEDROCK_MANTLE_MESSAGES_CONFIG, "2026-10-21", None)]
    #[case::bedrock_mantle_under_sdk_1(&BEDROCK_MANTLE_MESSAGES_CONFIG, "2023-06-01", Some(1.0))]
    #[case::bedrock_mantle_route_under_sdk_2(&BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG, "2026-10-21", None)]
    #[case::bedrock_mantle_route_under_sdk_1(
        &BEDROCK_MANTLE_ROUTE_MESSAGES_CONFIG,
        "2023-06-01",
        Some(1.0)
    )]
    fn temperature_follows_the_forwarded_api_version(
        #[case] config: &BedrockMantleMessagesConfig,
        #[case] version: &str,
        #[case] temperature: Option<f64>,
    ) {
        let body = config
            .request_body(
                serde_json::from_value(json!({
                    "model": "anthropic.claude-test",
                    "max_tokens": 16,
                    "temperature": 1.0,
                    "messages": [{"role": "user", "content": "hi"}]
                }))
                .unwrap(),
                &vec![("anthropic-version".to_string(), version.to_string())],
                "https://bedrock-mantle.us-west-2.api.aws/anthropic/v1/messages",
                true,
            )
            .unwrap();
        assert_eq!(
            body.params
                .temperature
                .as_ref()
                .and_then(Recognized::known)
                .copied(),
            temperature
        );
    }
}
