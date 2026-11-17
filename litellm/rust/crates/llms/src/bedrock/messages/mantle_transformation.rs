//! Bedrock Mantle's Anthropic Messages endpoint on the `bedrock/mantle/` route, Python's
//! `AmazonMantleMessagesConfig`: the Bedrock Invoke request transform with Mantle's overrides,
//! the betas moved into a header, the model id and `stream` back in the body, the Mantle URL and
//! zero-filled usage. `bedrock_mantle/` reuses this shaping.

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
    anthropic::messages::{
        handler::shape_anthropic_messages_request,
        transformation::retry_anthropic_messages_after_http_error,
    },
    base_llm::{
        auth::AuthScheme,
        messages::{
            context::{MessagesModelCapabilities, MessagesShapeOptions, MessagesTransformContext},
            transformation::{BaseMessagesConfig, Headers, ValidatedEnvironment},
        },
    },
    bedrock::common_utils::is_valid_region,
};

use super::invoke_transformations::{
    anthropic_claude_transformation::{
        bearer_token, invoke_headers, native_output_fallback_litellm_params, region_from_model,
        should_retry_anthropic_messages_without_native_output,
    },
    request::{
        BEDROCK_INVOKE_REQUEST_SETTINGS, InvokeRequestSettings,
        clamp_adaptive_reasoning_effort_for_bedrock, transform_anthropic_messages_request,
        with_logged_tools,
    },
};

pub(crate) const MANTLE_MESSAGES_PATH: &str = "/anthropic/v1/messages";
pub(crate) const MANTLE_API_BASE_ENV: &str = "BEDROCK_MANTLE_API_BASE";
const MANTLE_OPENAI_BASE_SUFFIXES: [&str; 2] = ["/openai/v1", "/v1"];
const DEFAULT_ANTHROPIC_API_VERSION: &str = "2023-06-01";
const MANTLE_ALLOWED_TOP_LEVEL_FIELDS: [&str; 1] = ["service_tier"];
const SECRET_NAMES: [&str; 5] = [
    AWS_BEARER_TOKEN_BEDROCK,
    MANTLE_API_BASE_ENV,
    AWS_REGION_NAME,
    AWS_REGION,
    AWS_DEFAULT_REGION,
];

pub(crate) const AMAZON_MANTLE_REQUEST_SETTINGS: InvokeRequestSettings = InvokeRequestSettings {
    beta_headers_provider: BetaProvider::BedrockMantle,
    filters_caller_betas: false,
    drops_dangerous_tool_use_beta_without_safeguards: false,
    extra_allowed_top_level_fields: &MANTLE_ALLOWED_TOP_LEVEL_FIELDS,
    ..BEDROCK_INVOKE_REQUEST_SETTINGS
};

// TODO: `bedrock/mantle/` is slated for deprecation in favour of `bedrock_mantle/`. Folding it in must keep
// its defaults (us-west-2, no BEDROCK_MANTLE_API_KEY, this URL building, no claude gate) or ship a deprecation notice.
/// The `bedrock/mantle/` Messages config: Bedrock region resolution and SigV4 or
/// `AWS_BEARER_TOKEN_BEDROCK` auth, an explicit base, runtime endpoint or `BEDROCK_MANTLE_API_BASE`
/// before the regional host, and the model id with its first `mantle/` dropped.
pub struct AmazonMantleMessagesConfig;
pub const AMAZON_MANTLE_MESSAGES_CONFIG: AmazonMantleMessagesConfig = AmazonMantleMessagesConfig;

fn without_exact(headers: Headers, name: &str) -> Headers {
    headers.into_iter().filter(|(key, _)| key != name).collect()
}

fn has_any_casing(headers: &Headers, name: &str) -> bool {
    headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case(name))
}

/// Python's `AmazonMantleMessagesConfig.validate_anthropic_messages_environment`.
pub(crate) fn mantle_headers(
    headers: Headers,
    litellm_params: &LitellmParams,
) -> Result<Headers, Error> {
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

pub(crate) fn sigv4_environment(
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

/// Python's `AmazonMantleMessagesConfig.transform_anthropic_messages_request` up to the body: the Invoke
/// transform runs on `model_id` with the Mantle overrides, then the betas move into the header. The
/// request keeps its own model, which `wire_body` turns into `model_id`.
pub(crate) fn transform_mantle_messages_request(
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
    let stream = request
        .params
        .stream
        .filter(|stream| *stream == Recognized::Known(true));
    Ok((
        MessagesRequest {
            model,
            params: MessagesOptionalParams {
                stream,
                ..request.params
            },
            ..request
        },
        headers,
    ))
}

/// The model id `bedrock/mantle/` sends: the model without its first `mantle/`.
pub fn amazon_mantle_model_id(model: &str) -> String {
    model.replacen("mantle/", "", 1)
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

/// The body half of Python's Mantle transform. A body that is not a Mantle request object is sent as it is.
pub(crate) fn mantle_wire_body(body: Value, model_id: impl Fn(&str) -> String) -> Value {
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

/// Python's `AmazonMantleMessagesConfig.transform_anthropic_messages_response`.
pub(crate) fn with_zero_usage(response: MessagesResponse) -> Result<MessagesResponse, Error> {
    let usage = usage_value(billed_usage(response.usage.clone())?)?;
    Ok(MessagesResponse {
        usage: Some(usage),
        ..response
    })
}

/// A reply that is not a message, such as a 200 error envelope, still bills zero-filled usage, as
/// Python's Mantle response transform fills it on whatever the provider answered.
pub(crate) fn unchecked_with_zero_usage(
    mut body: Map<String, Value>,
) -> Result<Map<String, Value>, Error> {
    let usage = usage_value(billed_usage(body.remove("usage"))?)?;
    body.insert("usage".to_string(), usage);
    Ok(body)
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

/// A Mantle base parsed as a URL, refused when it is not one.
pub(crate) fn parse_mantle_base(base: &str) -> Result<Url, Error> {
    Url::parse(base).map_err(|error| Error::InvalidRequest(ErrorDetail::invalid("api_base", error)))
}

/// `base` with the first of `suffixes` its path ends in taken off and the Messages route put on.
/// The query string stays after the path.
pub(crate) fn with_mantle_messages_path(mut base: Url, suffixes: &[&str]) -> Url {
    let path = base.path().trim_end_matches('/');
    let stripped = suffixes
        .iter()
        .find_map(|suffix| path.strip_suffix(suffix))
        .unwrap_or(path);
    let path = format!("{stripped}{MANTLE_MESSAGES_PATH}");
    base.set_path(&path);
    base
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

/// The Bedrock region for the host and the SigV4 scope, in Python's `_get_aws_region_name` order:
/// `aws_region_name`, the model's ARN or `<region>/` path segment, then the environment.
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

pub(crate) fn validated_mantle_region(
    region: String,
    check: MantleRegionCheck,
) -> Result<String, Error> {
    match is_valid_region(&region) {
        true => Ok(region),
        false => Err(Error::InvalidRequest(ErrorDetail::InvalidMantleRegion {
            region,
            check,
        })),
    }
}

pub(crate) fn mantle_secret_names(names: &'static [&'static str]) -> Box<[&'static str]> {
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

impl BaseMessagesConfig for AmazonMantleMessagesConfig {
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

    fn clamp_requested_effort(
        &self,
        request: MessagesRequest,
        capabilities: &MessagesModelCapabilities,
    ) -> Result<MessagesRequest, Error> {
        clamp_adaptive_reasoning_effort_for_bedrock(request, capabilities)
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        litellm_params: &LitellmParams,
        _stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let region = validated_mantle_region(
            bedrock_region(model, litellm_params, env_lookup),
            MantleRegionCheck::Request,
        )?;
        build_mantle_messages_url(
            api_base.or(litellm_params.api_base.as_deref()),
            litellm_params.aws.aws_bedrock_runtime_endpoint.as_deref(),
            &region,
            env_lookup,
        )
    }

    fn finalize_request(
        &self,
        request: MessagesRequest,
        headers: Headers,
        context: &MessagesTransformContext,
    ) -> Result<(MessagesRequest, Headers), Error> {
        let model_id = amazon_mantle_model_id(&request.model);
        transform_mantle_messages_request(
            request,
            model_id,
            headers,
            context,
            &AMAZON_MANTLE_REQUEST_SETTINGS,
        )
    }

    fn native_output_fallback(
        &self,
        request: &MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Option<MessagesTransformContext> {
        native_output_fallback_litellm_params(request, context)
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

    fn transform_anthropic_messages_response(
        &self,
        _model: &str,
        response: MessagesResponse,
    ) -> Result<MessagesResponse, Error> {
        with_zero_usage(response)
    }

    fn transform_unchecked_response(
        &self,
        body: Map<String, Value>,
    ) -> Result<Map<String, Value>, Error> {
        unchecked_with_zero_usage(body)
    }

    fn logged_params(
        &self,
        params: MessagesOptionalParams,
        context: &MessagesTransformContext,
    ) -> Result<MessagesOptionalParams, Error> {
        with_logged_tools(params, context)
    }

    fn wire_body(&self, body: Value) -> Value {
        mantle_wire_body(body, amazon_mantle_model_id)
    }

    fn secret_names(&self) -> &'static [&'static str] {
        static NAMES: LazyLock<Box<[&'static str]>> =
            LazyLock::new(|| mantle_secret_names(&SECRET_NAMES));
        &NAMES
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
        let api_key = api_key.filter(|key| key.is_empty() || !key.trim().is_empty());
        Ok(match bearer_token(api_key, env_lookup) {
            Some(token) => bearer_environment(headers, &token),
            None => sigv4_environment(
                headers,
                bedrock_region(model, litellm_params, env_lookup),
                &litellm_params.aws,
                env_lookup,
            ),
        })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        }
    }

    fn params(region: Option<&str>, runtime_endpoint: Option<&str>) -> LitellmParams {
        LitellmParams {
            aws: AwsParams {
                aws_region_name: region.map(str::to_string),
                aws_bedrock_runtime_endpoint: runtime_endpoint.map(str::to_string),
                ..AwsParams::default()
            },
            ..LitellmParams::default()
        }
    }

    fn url(
        api_base: Option<&str>,
        model: &str,
        litellm_params: &LitellmParams,
        env_pairs: &'static [(&'static str, &'static str)],
    ) -> String {
        AMAZON_MANTLE_MESSAGES_CONFIG
            .get_complete_url(api_base, model, litellm_params, false, &env(env_pairs))
            .unwrap()
    }

    fn environment(
        forwarded: &[(&str, &str)],
        api_key: Option<&str>,
        model: &str,
        litellm_params: &LitellmParams,
        env_pairs: &'static [(&'static str, &'static str)],
    ) -> ValidatedEnvironment {
        AMAZON_MANTLE_MESSAGES_CONFIG
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
            url(
                api_base,
                "mantle/anthropic.claude-test",
                &params(Some("us-east-1"), runtime_endpoint),
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
        let litellm_params = params(aws_region_name, None);
        assert_eq!(
            url(None, model, &litellm_params, env_pairs),
            format!("https://bedrock-mantle.{expected}.api.aws/anthropic/v1/messages")
        );
        assert!(matches!(
            environment(&[], None, model, &litellm_params, env_pairs).auth,
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
            url(
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
        let validated = environment(
            &[],
            None,
            "mantle/anthropic.claude-test",
            &params(Some("us-east-1"), None),
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
    fn a_blank_api_key_counts_as_unset() {
        let validated = environment(
            &[],
            Some("   "),
            "mantle/anthropic.claude-test",
            &params(Some("us-east-1"), None),
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
        let validated = environment(
            &forwarded,
            None,
            "mantle/anthropic.claude-test",
            &params(Some("us-east-1"), None),
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
}
