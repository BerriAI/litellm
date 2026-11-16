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
    recognized::python_truthy,
};
use litellm_router_types::LitellmParams;
use serde_json::{Map, Value};

use crate::{
    Error, ErrorDetail,
    anthropic::{
        common_utils::ensure_api_route,
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::retry_anthropic_messages_after_http_error,
        },
    },
    base_llm::{
        auth::AuthScheme,
        messages::{
            context::{MessagesModelCapabilities, MessagesShapeOptions, MessagesTransformContext},
            transformation::{BaseMessagesConfig, Headers, ValidatedEnvironment},
        },
    },
    bedrock::common_utils::validated_region,
};

use super::invoke_transformations::{
    anthropic_claude_transformation::{
        bearer_token, get_aws_region_from_model_arn, invoke_headers,
        native_output_fallback_litellm_params,
        should_retry_anthropic_messages_without_native_output,
    },
    request::{
        BEDROCK_INVOKE_REQUEST_SETTINGS, InvokeRequestSettings,
        clamp_adaptive_reasoning_effort_for_bedrock, transform_anthropic_messages_request,
    },
};

pub(crate) const MANTLE_MESSAGES_PATH: &str = "/anthropic/v1/messages";
pub(crate) const MANTLE_API_BASE_ENV: &str = "BEDROCK_MANTLE_API_BASE";
const MANTLE_OPENAI_BASE_SUFFIXES: [&str; 2] = ["/openai/v1", "/v1"];
const DEFAULT_ANTHROPIC_API_VERSION: &str = "2023-06-01";
const ZERO_USAGE_FIELDS: [&str; 2] = ["input_tokens", "output_tokens"];
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
        .extra
        .get("aws_bedrock_project_id")
        .and_then(Value::as_str)
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

/// Python's `_move_betas_into_header`.
fn move_betas_into_header(request: &MessagesRequest, headers: Headers) -> Result<Headers, Error> {
    let body = serde_json::to_value(request).map_err(|error| {
        Error::InvalidRequest(ErrorDetail::failed(
            "Bedrock Mantle request serialization",
            error,
        ))
    })?;
    let not_a_list = || {
        Error::InvalidRequest(ErrorDetail::Expected {
            subject: "anthropic_beta",
            expected: "a list of strings",
        })
    };
    let betas: Vec<&str> = match body
        .get("anthropic_beta")
        .filter(|betas| python_truthy(betas))
    {
        None => Vec::new(),
        Some(Value::Array(betas)) => betas
            .iter()
            .map(|beta| beta.as_str().ok_or_else(not_a_list))
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(not_a_list()),
    };
    let headers = without_exact(headers, "anthropic-beta");
    Ok(match betas.is_empty() {
        true => headers,
        false => headers
            .into_iter()
            .chain([("anthropic-beta".to_string(), betas.join(","))])
            .collect(),
    })
}

/// Python's `AmazonMantleMessagesConfig.transform_anthropic_messages_request` up to the body: the Invoke
/// transform with the Mantle overrides, then the betas move into the header.
pub(crate) fn transform_mantle_messages_request(
    request: MessagesRequest,
    headers: Headers,
    context: &MessagesTransformContext,
    settings: &InvokeRequestSettings,
) -> Result<(MessagesRequest, Headers), Error> {
    let (request, headers) =
        transform_anthropic_messages_request(request, context, headers, settings)?;
    let headers = move_betas_into_header(&request, headers)?;
    Ok((request, headers))
}

/// The body half of Python's Mantle transform: `model` and `stream` go last, `stream` only when it is `True`.
pub(crate) fn mantle_wire_body(body: Value, model_id: impl Fn(&str) -> String) -> Value {
    let Value::Object(fields) = body else {
        return body;
    };
    let model = fields
        .get("model")
        .and_then(Value::as_str)
        .map(&model_id)
        .filter(|model| !model.is_empty());
    let streaming = fields.get("stream") == Some(&Value::Bool(true));
    let kept = fields.into_iter().filter(|(name, _)| {
        !matches!(
            name.as_str(),
            "anthropic_version" | "anthropic_beta" | "model" | "stream"
        )
    });
    let mantle_fields = model
        .map(|model| ("model".to_string(), Value::String(model)))
        .into_iter()
        .chain(streaming.then(|| ("stream".to_string(), Value::Bool(true))));
    Value::Object(kept.chain(mantle_fields).collect::<Map<_, _>>())
}

/// Python's `AmazonMantleMessagesConfig.transform_anthropic_messages_response`.
pub(crate) fn with_zero_usage(response: MessagesResponse) -> MessagesResponse {
    let existing = match response.usage {
        Some(ref usage) if python_truthy(usage) => usage.clone(),
        _ => Value::Object(Map::new()),
    };
    let Value::Object(existing) = existing else {
        return response;
    };
    let zero_filled = ZERO_USAGE_FIELDS.map(|name| {
        (
            name.to_string(),
            existing.get(name).cloned().unwrap_or(Value::from(0)),
        )
    });
    let usage = zero_filled
        .into_iter()
        .chain(
            existing
                .into_iter()
                .filter(|(name, _)| !ZERO_USAGE_FIELDS.contains(&name.as_str())),
        )
        .collect::<Map<_, _>>();
    MessagesResponse {
        usage: Some(Value::Object(usage)),
        ..response
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
    Some(stripped.to_string()).filter(|base| !base.is_empty())
}

/// Python's `build_mantle_messages_url`.
fn build_mantle_messages_url(
    api_base: Option<&str>,
    aws_bedrock_runtime_endpoint: Option<&str>,
    region: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let explicit = [api_base, aws_bedrock_runtime_endpoint]
        .into_iter()
        .flatten()
        .find(|base| !base.is_empty())
        .map(str::to_string);
    match explicit.or_else(|| mantle_api_base_from_env(env_lookup)) {
        Some(base) => ensure_api_route(&base, MANTLE_MESSAGES_PATH),
        None => format!("https://bedrock-mantle.{region}.api.aws{MANTLE_MESSAGES_PATH}"),
    }
}

fn bedrock_region(
    model: &str,
    litellm_params: &LitellmParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    resolve_bedrock_region(
        get_aws_region_from_model_arn(model).as_deref(),
        &litellm_params.aws,
        env_lookup,
    )
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
        let region = validated_region(bedrock_region(model, litellm_params, env_lookup))?;
        Ok(build_mantle_messages_url(
            api_base.or(litellm_params.api_base.as_deref()),
            litellm_params.aws.aws_bedrock_runtime_endpoint.as_deref(),
            &region,
            env_lookup,
        ))
    }

    fn finalize_request(
        &self,
        request: MessagesRequest,
        headers: Headers,
        context: &MessagesTransformContext,
    ) -> Result<(MessagesRequest, Headers), Error> {
        transform_mantle_messages_request(
            request,
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
        Ok(with_zero_usage(response))
    }

    fn wire_body(&self, body: Value) -> Value {
        mantle_wire_body(body, |model| model.replacen("mantle/", "", 1))
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
        &[("BEDROCK_MANTLE_REGION", "eu-west-1"), ("AWS_REGION", "ap-south-1")],
        "ap-south-1"
    )]
    #[case::aws_default_region(
        "mantle/anthropic.claude-test",
        &[("AWS_DEFAULT_REGION", "eu-west-1")],
        "eu-west-1"
    )]
    #[case::region_prefix_ignored(
        "mantle/eu-west-1/anthropic.claude-test",
        &[("AWS_REGION", "us-east-2")],
        "us-east-2"
    )]
    #[case::arn_region(
        "mantle/arn:aws:bedrock:eu-central-1:123456789012:inference-profile/anthropic.claude-test",
        &[("AWS_REGION", "us-east-2")],
        "eu-central-1"
    )]
    fn host_and_signing_region_follow_the_invoke_order(
        #[case] model: &str,
        #[case] env_pairs: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        let litellm_params = params(None, None);
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
}
