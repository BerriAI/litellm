use litellm_host::interceptors::{RetriedRequest, RetryReason};

use crate::{
    anthropic::messages::transformation::ANTHROPIC_MESSAGES_CONFIG,
    base_llm::messages::context::{
        MessagesModelCapabilities, MessagesShapeOptions, MessagesTransformContext,
        NativeOutputSupport,
    },
};
use litellm_auth::CredentialPlacement;
use litellm_secret_value::SecretValue;

use litellm_auth_aws::{
    AwsCredentialSource, CallerSigningHeaders, bedrock_runtime_endpoint,
    constants::{
        AWS_BEARER_TOKEN_BEDROCK, AWS_BEDROCK_RUNTIME_ENDPOINT, AWS_DEFAULT_REGION, AWS_REGION,
        AWS_REGION_NAME, BEDROCK_SERVICE,
    },
    resolve_bedrock_region,
};
use litellm_llms_types::formats::messages::{MessagesOptionalParams, MessagesRequest};
use litellm_llms_types::providers::bedrock::{INVOKE_PATH, INVOKE_STREAM_PATH};
use litellm_router_types::LitellmParams;
use serde_json::Value;

use crate::{
    Error,
    base_llm::{
        auth::AuthScheme,
        messages::{
            streaming::StreamDecoder,
            transformation::{
                Headers, MessagesAuth, MessagesDialect, MessagesEndpoint, MessagesReply,
                ValidatedEnvironment,
            },
        },
    },
};

use crate::bedrock::{
    chat::invoke_transformations::base_invoke_transformation::{
        GUARDRAIL_CONFIG_PARAM, bedrock_invoke_guardrail_headers,
    },
    common_utils::{
        split_bedrock_region_path,
        structured_output::{
            NATIVE_OUTPUT_UNSUPPORTED, bedrock_native_output_format,
            bedrock_native_output_retry_requested, is_bedrock_native_output_rejection,
            requested_output_format, sent_output_config_format,
        },
        validated_region,
    },
    request_metadata::{bedrock_request_metadata_headers, merge_bedrock_invoke_headers},
};

use super::{betas::with_thinking_binding_beta, request, stream};

const INVOKE_MODEL_PREFIX: &str = "invoke/";

const MODEL_ID_PARAM: &str = "model_id";

const MODEL_SPEC_PREFIXES: [&str; 2] = ["nova-2/", "nova/"];

const SECRET_NAMES: &[&str] = &[
    AWS_BEARER_TOKEN_BEDROCK,
    AWS_BEDROCK_RUNTIME_ENDPOINT,
    AWS_REGION_NAME,
    AWS_REGION,
    AWS_DEFAULT_REGION,
];

pub struct BedrockInvokeMessagesConfig;

pub const BEDROCK_INVOKE_MESSAGES_CONFIG: BedrockInvokeMessagesConfig = BedrockInvokeMessagesConfig;

pub(crate) fn bearer_token(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    match api_key {
        Some(key) => Some(key.to_string()),
        None => env_lookup(AWS_BEARER_TOKEN_BEDROCK),
    }
    .filter(|token| !token.is_empty())
}

/// Port of Python's `BaseAWSLLM.encode_model_id`.
pub fn encode_model_id(text: &str) -> String {
    text.bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => {
                char::from(byte).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}

/// Port of Python's `BaseAWSLLM.get_bedrock_model_id` for Claude.
fn get_bedrock_model_id(model: &str, model_id: Option<&str>) -> String {
    if let Some(model_id) = model_id {
        return encode_model_id(model_id).replacen(INVOKE_MODEL_PREFIX, "", 1);
    }
    let (_, stripped) = split_bedrock_region_path(model);
    if stripped.starts_with("arn:") {
        return encode_model_id(stripped);
    }
    let model_id = stripped.replacen(INVOKE_MODEL_PREFIX, "", 1);
    match MODEL_SPEC_PREFIXES
        .iter()
        .find(|spec| model_id.contains(**spec))
    {
        Some(spec) => encode_model_id(&model_id.replace(spec, "")),
        None => model_id,
    }
}

fn is_arn_partition_char(character: char) -> bool {
    character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-'
}

/// Port of Python's `contains_bedrock_arn`.
fn contains_bedrock_arn(model: &str) -> bool {
    model.match_indices("arn:aws").any(|(index, prefix)| {
        let rest = &model[index + prefix.len()..];
        let after_partition = match rest.strip_prefix('-') {
            Some(partition) => {
                let trimmed = partition.trim_start_matches(is_arn_partition_char);
                if trimmed.len() == partition.len() {
                    return false;
                }
                trimmed
            }
            None => rest,
        };
        after_partition.starts_with(":bedrock")
    })
}

/// Port of Python's `BaseAWSLLM._get_aws_region_from_model_arn`.
pub(crate) fn get_aws_region_from_model_arn(model: &str) -> Option<String> {
    contains_bedrock_arn(model)
        .then(|| model.split(':').nth(3))
        .flatten()
        .filter(|region| !region.is_empty() && region.chars().all(is_arn_partition_char))
        .map(str::to_string)
}

/// The region a Bedrock model names: an ARN's region, else a `<region>/` path segment after the
/// routing prefixes, as Python's `BaseAWSLLM._get_aws_region_name` reads it before the environment.
pub(crate) fn region_from_model(model: &str) -> Option<String> {
    get_aws_region_from_model_arn(model)
        .or_else(|| split_bedrock_region_path(model).0.map(str::to_string))
}

/// Port of Python's `AmazonInvokeConfig.get_complete_url`.
fn invoke_url(
    api_base: Option<&str>,
    model: &str,
    litellm_params: &LitellmParams,
    stream: bool,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let aws = &litellm_params.aws;
    let model_id = get_bedrock_model_id(
        model,
        litellm_params
            .extra
            .get(MODEL_ID_PARAM)
            .and_then(Value::as_str),
    );
    let region = resolve_bedrock_region(region_from_model(model).as_deref(), aws, env_lookup);
    let path = if stream {
        INVOKE_STREAM_PATH
    } else {
        INVOKE_PATH
    };
    let configured = |value: Option<&str>| {
        value
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    let endpoint = match configured(api_base)
        .or_else(|| configured(aws.aws_bedrock_runtime_endpoint.as_deref()))
        .or_else(|| env_lookup(AWS_BEDROCK_RUNTIME_ENDPOINT))
    {
        Some(endpoint) => endpoint,
        None => bedrock_runtime_endpoint(&validated_region(region)?),
    };
    Ok(format!(
        "{}/model/{model_id}/{path}",
        endpoint.trim_end_matches('/')
    ))
}

/// The header half of Python's `validate_anthropic_messages_environment`.
pub(crate) fn invoke_headers(
    headers: Headers,
    litellm_params: &LitellmParams,
) -> Result<Headers, Error> {
    let guardrail = match litellm_params.extra.get(GUARDRAIL_CONFIG_PARAM) {
        None | Some(Value::Null) => Vec::new(),
        Some(raw) => bedrock_invoke_guardrail_headers(raw)?,
    };
    Ok(merge_bedrock_invoke_headers(
        headers,
        guardrail,
        bedrock_request_metadata_headers(&litellm_params.extra),
    ))
}

fn without_native_output(context: &MessagesTransformContext) -> MessagesTransformContext {
    let mut fallback = *context;
    fallback.thinking.capabilities.bedrock.native_output = NativeOutputSupport {
        enabled: false,
        opt_in: false,
        replaces_unforceable_tool: false,
        ..fallback.thinking.capabilities.bedrock.native_output
    };
    fallback
}

/// Port of Python's `native_output_fallback_litellm_params`.
pub(crate) fn native_output_fallback_litellm_params(
    request: &MessagesRequest,
    context: &MessagesTransformContext,
) -> Option<MessagesTransformContext> {
    let native = context.thinking.capabilities.bedrock.native_output;
    let Ok(Value::Object(params)) = serde_json::to_value(&request.params) else {
        return None;
    };
    if !bedrock_native_output_retry_requested(&params, native) {
        return None;
    }
    let output_format = requested_output_format(&params)?;
    bedrock_native_output_format(output_format, native)
        .ok()
        .flatten()
        .map(|_| without_native_output(context))
}

/// Port of Python's `should_retry_anthropic_messages_without_native_output`.
pub(crate) fn should_retry_anthropic_messages_without_native_output(
    status: u16,
    error_text: &str,
    request_body: &Value,
    model: &str,
) -> bool {
    let retries = sent_output_config_format(request_body)
        && is_bedrock_native_output_rejection(status, error_text);
    if retries {
        let opt_out_advice = if error_text.contains(NATIVE_OUTPUT_UNSUPPORTED) {
            " Set `supports_bedrock_native_structured_output: false` in this deployment's model_info to skip the rejected call."
        } else {
            ""
        };
        tracing::warn!(
            "Bedrock rejected native structured output for model={model}, so litellm sent the request again without it.{opt_out_advice}"
        );
    }
    retries
}

impl MessagesEndpoint for BedrockInvokeMessagesConfig {
    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        litellm_params: &LitellmParams,
        stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        invoke_url(api_base, model, litellm_params, stream, env_lookup)
    }
}

impl MessagesAuth for BedrockInvokeMessagesConfig {
    fn secret_names(&self) -> &'static [&'static str] {
        SECRET_NAMES
    }

    /// Python reads `api_key` as the Bedrock bearer token and consults the env only when the
    /// caller passed none. Without one the request is signed with SigV4.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        model: &str,
        _optional_params: &MessagesOptionalParams,
        litellm_params: &LitellmParams,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let headers = invoke_headers(headers, litellm_params)?;
        if let Some(token) = bearer_token(api_key, env_lookup) {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Credential {
                    placement: CredentialPlacement::Bearer,
                    secret: SecretValue::new(token),
                },
            });
        }
        let params = &litellm_params.aws;
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::AwsSigV4 {
                region: resolve_bedrock_region(
                    region_from_model(model).as_deref(),
                    params,
                    env_lookup,
                ),
                service: BEDROCK_SERVICE,
                credentials: Box::new(
                    AwsCredentialSource::from_params(params, env_lookup).with_session_attribution(
                        litellm_params
                            .extra
                            .get(crate::bedrock::session_identity::REQUIRE_SESSION_ATTRIBUTION)
                            .and_then(serde_json::Value::as_bool)
                            .unwrap_or(false),
                    ),
                ),
                caller_headers: CallerSigningHeaders::Botocore,
            },
        })
    }
}

impl MessagesDialect for BedrockInvokeMessagesConfig {
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
        request::clamp_adaptive_reasoning_effort_for_bedrock(request, capabilities)
    }

    fn native_output_fallback(
        &self,
        request: &MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Option<MessagesTransformContext> {
        native_output_fallback_litellm_params(request, context)
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("content-type", "application/json")]
    }

    /// Python's `transform_anthropic_messages_request`, which picks the betas from the caller's
    /// header and the request and sends them in the body.
    fn finalize_request(
        &self,
        request: MessagesRequest,
        headers: Headers,
        context: &MessagesTransformContext,
    ) -> Result<(MessagesRequest, Headers), Error> {
        request::transform_anthropic_messages_request(
            request,
            context,
            headers,
            &request::BEDROCK_INVOKE_REQUEST_SETTINGS,
        )
    }

    fn wire_body(&self, body: Value) -> Value {
        match body {
            Value::Object(mut fields) => {
                fields.shift_remove("model");
                fields.shift_remove("stream");
                Value::Object(fields)
            }
            other => other,
        }
    }
}

impl MessagesReply for BedrockInvokeMessagesConfig {
    /// Port of Python's `get_async_streaming_response_iterator`.
    fn stream_decoder(&self) -> Option<StreamDecoder> {
        Some(stream::get_async_streaming_response_iterator)
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
        ANTHROPIC_MESSAGES_CONFIG
            .retry_after_http_error(status, error_text, request_body, context)
            .map(with_thinking_binding_beta)
    }

    fn retries_past_attempt_limit(
        &self,
        last_reasons: &[RetryReason],
        status: u16,
        error_text: &str,
    ) -> bool {
        ANTHROPIC_MESSAGES_CONFIG.retries_past_attempt_limit(last_reasons, status, error_text)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use litellm_auth::AwsParams;
    use litellm_auth_aws::constants::DEFAULT_BEDROCK_REGION;

    use super::*;

    fn in_region(region: &str) -> LitellmParams {
        LitellmParams {
            aws: AwsParams {
                aws_region_name: Some(region.into()),
                ..AwsParams::default()
            },
            ..LitellmParams::default()
        }
    }

    fn with_runtime_endpoint(endpoint: &str) -> LitellmParams {
        LitellmParams {
            aws: AwsParams {
                aws_bedrock_runtime_endpoint: Some(endpoint.into()),
                ..in_region("us-west-2").aws
            },
            ..LitellmParams::default()
        }
    }

    #[rstest]
    #[case::invoke_in_the_params_region(
        None,
        in_region("us-west-2"),
        false,
        None,
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3/invoke"
    )]
    #[case::stream_in_the_params_region(
        None,
        in_region("us-west-2"),
        true,
        None,
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3/invoke-with-response-stream"
    )]
    #[case::api_base_outranks_the_params_endpoint(
        Some("https://base.test/"),
        with_runtime_endpoint("https://params.test"),
        false,
        None,
        "https://base.test/model/anthropic.claude-3/invoke"
    )]
    #[case::a_blank_api_base_does_not_hide_the_params_endpoint(
        Some("  "),
        with_runtime_endpoint("https://params.test"),
        false,
        Some("https://env.test"),
        "https://params.test/model/anthropic.claude-3/invoke"
    )]
    #[case::params_endpoint_outranks_the_environment(
        None,
        with_runtime_endpoint("https://params.test"),
        false,
        Some("https://env.test"),
        "https://params.test/model/anthropic.claude-3/invoke"
    )]
    #[case::environment_outranks_the_region_template(
        None,
        in_region("us-west-2"),
        false,
        Some("https://env.test"),
        "https://env.test/model/anthropic.claude-3/invoke"
    )]
    fn url_follows_python_endpoint_precedence_and_the_stream_path(
        #[case] api_base: Option<&str>,
        #[case] litellm_params: LitellmParams,
        #[case] stream: bool,
        #[case] env_endpoint: Option<&str>,
        #[case] expected: &str,
    ) {
        let env = |name: &str| {
            (name == AWS_BEDROCK_RUNTIME_ENDPOINT)
                .then_some(env_endpoint)
                .flatten()
                .map(str::to_string)
        };
        assert_eq!(
            BedrockInvokeMessagesConfig
                .get_complete_url(
                    api_base,
                    "anthropic.claude-3",
                    &litellm_params,
                    stream,
                    &env
                )
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::host_injection("evil.example.com/x?")]
    #[case::uppercase("US-WEST-2")]
    #[case::port("us-west-2:8443")]
    fn a_region_outside_the_aws_alphabet_never_builds_a_url(#[case] region: &str) {
        let url = BedrockInvokeMessagesConfig.get_complete_url(
            None,
            "anthropic.claude-3",
            &in_region(region),
            false,
            &|_| None,
        );
        assert!(matches!(url, Err(Error::InvalidRequest(_))), "{url:?}");
    }

    #[test]
    fn a_configured_endpoint_does_not_read_the_region() {
        assert_eq!(
            BedrockInvokeMessagesConfig
                .get_complete_url(
                    Some("https://base.test"),
                    "anthropic.claude-3",
                    &in_region("evil.example.com/x?"),
                    false,
                    &|_| None,
                )
                .unwrap(),
            "https://base.test/model/anthropic.claude-3/invoke"
        );
    }

    #[test]
    fn sigv4_scope_and_credentials_come_from_the_litellm_params() {
        let litellm_params = LitellmParams {
            aws: AwsParams {
                aws_access_key_id: Some("AKIAPARAMS".into()),
                aws_secret_access_key: Some("params-secret".into()),
                ..in_region("eu-central-1").aws
            },
            ..LitellmParams::default()
        };
        let validated = BedrockInvokeMessagesConfig
            .validate_environment(
                Vec::new(),
                None,
                "anthropic.claude-3",
                &MessagesOptionalParams::default(),
                &litellm_params,
                &|_| None,
            )
            .unwrap();
        let AuthScheme::AwsSigV4 {
            region,
            credentials,
            ..
        } = validated.auth
        else {
            panic!("expected SigV4, got {:?}", validated.auth);
        };
        let AwsCredentialSource::HostSupplied(credentials) = *credentials else {
            panic!("expected the params' static keys, got {credentials:?}");
        };
        assert_eq!(
            (region.as_str(), credentials.access_key_id()),
            ("eu-central-1", "AKIAPARAMS")
        );
    }

    #[rstest]
    #[case::an_explicit_key_is_a_bearer_token(Some("token"), None, Some("token"))]
    #[case::the_env_token_is_a_bearer_token(None, Some("env-token"), Some("env-token"))]
    #[case::no_token_signs_with_sigv4(None, None, None)]
    fn requests_sign_only_without_a_bearer_token(
        #[case] api_key: Option<&str>,
        #[case] env_token: Option<&str>,
        #[case] expected_bearer: Option<&str>,
    ) {
        let env = |name: &str| {
            (name == AWS_BEARER_TOKEN_BEDROCK)
                .then(|| env_token.map(str::to_string))
                .flatten()
        };
        let validated = BedrockInvokeMessagesConfig
            .validate_environment(
                vec![("authorization".into(), "Bearer forwarded".into())],
                api_key,
                "anthropic.claude-3",
                &MessagesOptionalParams::default(),
                &LitellmParams::default(),
                &env,
            )
            .unwrap();
        match (validated.auth, expected_bearer) {
            (
                AuthScheme::Credential {
                    placement: CredentialPlacement::Bearer,
                    secret,
                },
                Some(expected),
            ) => assert_eq!(secret.expose(), expected),
            (
                AuthScheme::AwsSigV4 {
                    region, service, ..
                },
                None,
            ) => {
                assert_eq!(
                    (region.as_str(), service),
                    (DEFAULT_BEDROCK_REGION, BEDROCK_SERVICE)
                );
            }
            (other, _) => panic!("unexpected auth {other:?}"),
        }
    }

    const ARN: &str = "arn:aws:bedrock:us-east-1:086734376398:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0";
    const ENCODED_ARN: &str = "arn%3Aaws%3Abedrock%3Aus-east-1%3A086734376398%3Ainference-profile%2Fglobal.anthropic.claude-sonnet-4-5-20250929-v1%3A0";

    #[rstest]
    #[case::bedrock_prefixed_arn(
        "bedrock/arn:aws:bedrock:us-east-1:086734376398:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        None,
        ENCODED_ARN
    )]
    #[case::compound_prefixed_arn(
        "bedrock/invoke/arn:aws:bedrock:us-east-1:086734376398:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        None,
        ENCODED_ARN
    )]
    #[case::bare_arn(ARN, None, ENCODED_ARN)]
    #[case::regular_id(
        "anthropic.claude-3-sonnet-20240229-v1:0",
        None,
        "anthropic.claude-3-sonnet-20240229-v1:0"
    )]
    #[case::invoke_prefix(
        "invoke/anthropic.claude-3-sonnet-20240229-v1:0",
        None,
        "anthropic.claude-3-sonnet-20240229-v1:0"
    )]
    #[case::model_id_override(
        "anthropic.claude-3",
        Some("arn:aws:bedrock:us-east-1:1:application-inference-profile/abc"),
        "arn%3Aaws%3Abedrock%3Aus-east-1%3A1%3Aapplication-inference-profile%2Fabc"
    )]
    fn invoke_model_ids_match_python(
        #[case] model: &str,
        #[case] model_id: Option<&str>,
        #[case] expected: &str,
    ) {
        assert_eq!(get_bedrock_model_id(model, model_id), expected);
    }

    #[rstest]
    #[case::region_segment_leaves_the_path(
        "bedrock/invoke/us-west-2/anthropic.claude-3-5-sonnet-20240620-v1:0",
        None,
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3-5-sonnet-20240620-v1:0/invoke",
        "us-west-2"
    )]
    #[case::region_segment_picks_the_region(
        "invoke/eu-central-1/anthropic.claude-3-5-sonnet-20240620-v1:0",
        None,
        "https://bedrock-runtime.eu-central-1.amazonaws.com/model/anthropic.claude-3-5-sonnet-20240620-v1:0/invoke",
        "eu-central-1"
    )]
    #[case::gov_cloud_region_segment_without_a_route(
        "bedrock/us-gov-west-1/anthropic.claude-sonnet-4-5-20250929-v1:0",
        None,
        "https://bedrock-runtime.us-gov-west-1.amazonaws.com/model/anthropic.claude-sonnet-4-5-20250929-v1:0/invoke",
        "us-gov-west-1"
    )]
    #[case::params_region_beats_a_region_segment(
        "invoke/us-west-2/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        Some("us-east-1"),
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke",
        "us-east-1"
    )]
    #[case::arn_after_a_region_segment_picks_the_region(
        "invoke/us-west-2/arn:aws:bedrock:eu-west-1:123456789012:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        None,
        "https://bedrock-runtime.eu-west-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aeu-west-1%3A123456789012%3Ainference-profile%2Fus.anthropic.claude-sonnet-4-5-20250929-v1%3A0/invoke",
        "eu-west-1"
    )]
    #[case::arn_after_any_segment_picks_the_region(
        "invoke/foo/arn:aws:bedrock:eu-west-1:1:x",
        None,
        "https://bedrock-runtime.eu-west-1.amazonaws.com/model/foo/arn:aws:bedrock:eu-west-1:1:x/invoke",
        "eu-west-1"
    )]
    #[case::partitioned_arn(
        "invoke/arn:aws-us-gov:bedrock:us-gov-west-1:1:inference-profile/x",
        None,
        "https://bedrock-runtime.us-gov-west-1.amazonaws.com/model/arn%3Aaws-us-gov%3Abedrock%3Aus-gov-west-1%3A1%3Ainference-profile%2Fx/invoke",
        "us-gov-west-1"
    )]
    #[case::empty_partition_is_not_an_arn(
        "invoke/arn:aws-:bedrock:eu-west-1:1:x",
        None,
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/arn%3Aaws-%3Abedrock%3Aeu-west-1%3A1%3Ax/invoke",
        "us-west-2"
    )]
    #[case::uppercase_arn_region_is_ignored(
        "invoke/arn:aws:bedrock:EU:1:x",
        None,
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/arn%3Aaws%3Abedrock%3AEU%3A1%3Ax/invoke",
        "us-west-2"
    )]
    #[case::nova_spec_is_removed_and_the_id_encoded(
        "invoke/us-west-2/nova/anthropic.claude-3",
        None,
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3/invoke",
        "us-west-2"
    )]
    #[case::one_inner_invoke_segment_is_removed(
        "invoke/us-west-2/invoke/anthropic.claude-3",
        None,
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3/invoke",
        "us-west-2"
    )]
    fn url_and_signing_region_match_python_for_every_model_shape(
        #[case] model: &str,
        #[case] region_param: Option<&str>,
        #[case] expected_url: &str,
        #[case] expected_region: &str,
    ) {
        let litellm_params = region_param.map_or_else(LitellmParams::default, in_region);
        let url = BedrockInvokeMessagesConfig
            .get_complete_url(None, model, &litellm_params, false, &|_| None)
            .unwrap();
        let validated = BedrockInvokeMessagesConfig
            .validate_environment(
                Vec::new(),
                None,
                model,
                &MessagesOptionalParams::default(),
                &litellm_params,
                &|_| None,
            )
            .unwrap();
        let AuthScheme::AwsSigV4 {
            region,
            caller_headers,
            ..
        } = validated.auth
        else {
            panic!("expected SigV4, got {:?}", validated.auth);
        };
        assert_eq!(
            (url.as_str(), region.as_str(), caller_headers),
            (
                expected_url,
                expected_region,
                CallerSigningHeaders::Botocore
            )
        );
    }

    #[test]
    fn an_arn_url_is_percent_encoded_in_the_arn_region() {
        assert_eq!(
            BedrockInvokeMessagesConfig
                .get_complete_url(
                    None,
                    &format!("bedrock/{ARN}"),
                    &LitellmParams::default(),
                    true,
                    &|_| None
                )
                .unwrap(),
            format!(
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/{ENCODED_ARN}/invoke-with-response-stream"
            )
        );
    }

    #[test]
    fn the_model_id_param_overrides_the_url_model() {
        let litellm_params: LitellmParams = serde_json::from_value(serde_json::json!({
            "model": "bedrock/anthropic.claude-3",
            "aws_region_name": "us-west-2",
            "model_id": "my/profile",
        }))
        .unwrap();
        assert_eq!(
            BedrockInvokeMessagesConfig
                .get_complete_url(None, "anthropic.claude-3", &litellm_params, false, &|_| {
                    None
                })
                .unwrap(),
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/my%2Fprofile/invoke"
        );
    }

    #[test]
    fn guardrail_and_metadata_headers_are_added_for_both_auth_schemes() {
        let litellm_params: LitellmParams = serde_json::from_value(serde_json::json!({
            "model": "bedrock/anthropic.claude-3",
            "guardrailConfig": {"guardrailIdentifier": "gid", "guardrailVersion": "2", "trace": "enabled"},
        }))
        .unwrap();
        let validated = BedrockInvokeMessagesConfig
            .validate_environment(
                Vec::new(),
                Some("token"),
                "anthropic.claude-3",
                &MessagesOptionalParams::default(),
                &litellm_params,
                &|_| None,
            )
            .unwrap();
        assert_eq!(
            validated.headers,
            vec![
                (
                    "X-Amzn-Bedrock-GuardrailIdentifier".to_string(),
                    "gid".to_string()
                ),
                (
                    "X-Amzn-Bedrock-GuardrailVersion".to_string(),
                    "2".to_string()
                ),
                ("X-Amzn-Bedrock-Trace".to_string(), "ENABLED".to_string()),
            ]
        );
    }

    fn native_context(native_output: NativeOutputSupport) -> MessagesTransformContext {
        let mut context = MessagesTransformContext::default();
        context.thinking.capabilities.bedrock.native_output = native_output;
        context
    }

    fn structured_request() -> MessagesRequest {
        serde_json::from_value(serde_json::json!({
            "model": "anthropic.claude-sonnet-5",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "output_config": {"format": {"type": "json_schema", "schema": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}}},
        }))
        .unwrap()
    }

    #[rstest]
    #[case::native_enabled(NativeOutputSupport { enabled: true, mapped: true, ..NativeOutputSupport::default() }, true)]
    #[case::native_opt_in(NativeOutputSupport { opt_in: true, ..NativeOutputSupport::default() }, true)]
    #[case::inline_only(NativeOutputSupport::default(), false)]
    fn a_native_output_request_carries_an_inline_fallback(
        #[case] native_output: NativeOutputSupport,
        #[case] expected: bool,
    ) {
        let fallback = BedrockInvokeMessagesConfig
            .native_output_fallback(&structured_request(), &native_context(native_output));
        assert_eq!(fallback.is_some(), expected);
        if let Some(fallback) = fallback {
            let (rebuilt, _) = BedrockInvokeMessagesConfig
                .finalize_request(structured_request(), Vec::new(), &fallback)
                .unwrap();
            let body = serde_json::to_value(rebuilt).unwrap();
            assert_eq!(body.get("output_config"), None);
            assert_eq!(
                body["messages"][0]["content"][1]["type"],
                serde_json::json!("text")
            );
        }
    }

    #[rstest]
    #[case::rejected_native_output(400, "output_config.format: Extra inputs are not permitted", serde_json::json!({"output_config": {"format": {}}}), true)]
    #[case::inline_body_is_not_retried(400, "output_config.format: Extra inputs are not permitted", serde_json::json!({"messages": []}), false)]
    #[case::other_error(429, "Too many requests", serde_json::json!({"output_config": {"format": {}}}), false)]
    fn native_output_rejections_retry_without_it(
        #[case] status: u16,
        #[case] text: &str,
        #[case] body: Value,
        #[case] expected: bool,
    ) {
        assert_eq!(
            BedrockInvokeMessagesConfig.retries_without_native_output(status, text, &body, "m"),
            expected
        );
    }

    #[test]
    fn the_wire_body_drops_model_and_stream() {
        assert_eq!(
            BedrockInvokeMessagesConfig
                .wire_body(serde_json::json!({"model": "m", "stream": true, "max_tokens": 1})),
            serde_json::json!({"max_tokens": 1})
        );
    }
}
