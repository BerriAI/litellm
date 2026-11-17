use std::sync::LazyLock;

use litellm_host::interceptors::RetriedRequest;
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest, MessagesResponse},
    providers::anthropic::BetaProvider,
};
use litellm_router_types::LitellmParams;
use serde_json::Value;

use crate::{
    Error, MantleRegionCheck,
    anthropic::{
        common_utils::strip_trailing_slashes,
        messages::{
            handler::shape_anthropic_messages_request,
            transformation::retry_anthropic_messages_after_http_error,
        },
    },
    base_llm::messages::{
        context::{MessagesModelCapabilities, MessagesShapeOptions, MessagesTransformContext},
        transformation::{BaseMessagesConfig, Headers, ValidatedEnvironment},
    },
    bedrock::messages::{
        invoke_transformations::{
            anthropic_claude_transformation::{
                native_output_fallback_litellm_params,
                should_retry_anthropic_messages_without_native_output,
            },
            request::{InvokeRequestSettings, clamp_adaptive_reasoning_effort_for_bedrock},
        },
        mantle_transformation::{
            AMAZON_MANTLE_REQUEST_SETTINGS, MANTLE_API_BASE_ENV, MANTLE_MESSAGES_PATH,
            amazon_mantle_model_id, bearer_environment, mantle_headers, mantle_secret_names,
            mantle_wire_body, sigv4_environment, transform_mantle_messages_request,
            validated_mantle_region, with_zero_usage,
        },
    },
};

const BEDROCK_MANTLE: &str = "bedrock_mantle";
const BEDROCK: &str = "bedrock";
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
const KEY_ENVS: [&str; 2] = ["BEDROCK_MANTLE_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"];
const SECRET_NAMES: [&str; 6] = [
    "BEDROCK_MANTLE_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
    MANTLE_API_BASE_ENV,
    "BEDROCK_MANTLE_REGION",
    "AWS_REGION_NAME",
    "AWS_REGION",
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

const BEDROCK_MANTLE_REQUEST_SETTINGS: InvokeRequestSettings = InvokeRequestSettings {
    custom_llm_provider: BetaProvider::BedrockMantle,
    context_management_edits: &CONTEXT_MANAGEMENT_EDITS,
    ..AMAZON_MANTLE_REQUEST_SETTINGS
};

pub struct BedrockMantleAnthropicMessagesConfig;
pub const BEDROCK_MANTLE_MESSAGES_CONFIG: BedrockMantleAnthropicMessagesConfig =
    BedrockMantleAnthropicMessagesConfig;

fn first_non_empty<'a>(values: impl IntoIterator<Item = Option<&'a str>>) -> Option<&'a str> {
    values.into_iter().flatten().find(|value| !value.is_empty())
}

fn env_value(env_lookup: &dyn Fn(&str) -> Option<String>, names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| env_lookup(name).filter(|value| !value.is_empty()))
}

/// The region group of Python's `MANTLE_HOST_RE`, `^https?://bedrock-mantle\.([^/.]+)\.api\.aws(?=/|$)`,
/// matched case-insensitively.
fn mantle_host_region(base: &str) -> Option<&str> {
    let lowered = base.to_ascii_lowercase();
    let scheme = ["https://", "http://"]
        .into_iter()
        .find(|scheme| lowered.starts_with(scheme))?;
    let host = "bedrock-mantle.";
    let rest = lowered[scheme.len()..].strip_prefix(host)?;
    let region_len = rest.find(['/', '.']).unwrap_or(rest.len());
    let after = rest[region_len..].strip_prefix(".api.aws")?;
    if region_len == 0 || !(after.is_empty() || after.starts_with('/')) {
        return None;
    }
    let start = scheme.len() + host.len();
    Some(&base[start..start + region_len])
}

/// Python's `split_mantle_region_prefix`, which `get_llm_provider` applies before the config sees the model.
fn split_mantle_region_prefix(model: &str) -> (Option<&str>, &str) {
    match model.split_once('/') {
        Some((head, tail)) if BEDROCK_REGIONS.contains(&head) => (Some(head), tail),
        _ => (None, model),
    }
}

/// The model Python's Mantle transform sends: `get_llm_provider` drops the region prefix, then
/// `AmazonMantleMessagesConfig` drops the first `mantle/`.
pub fn mantle_model_id(model: &str) -> String {
    split_mantle_region_prefix(model)
        .1
        .replacen("mantle/", "", 1)
}

/// The provider and model Python's Mantle Messages configs probe capabilities under: the shared
/// Anthropic transform reads the route's provider, the Bedrock Invoke helpers read `bedrock`, and
/// both get the model id the route sends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MantleCapabilityLookup {
    pub provider: &'static str,
    pub model: String,
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

/// Python's `BedrockModelInfo._model_has_route_prefix`.
fn has_route_prefix(model: &str, prefix: &str) -> bool {
    model.starts_with(prefix) || model.contains(&format!("/{prefix}"))
}

fn configured_region(litellm_params: &LitellmParams) -> Option<&str> {
    first_non_empty([litellm_params.aws.aws_region_name.as_deref()])
}

/// Python's `resolve_mantle_region`.
fn resolve_mantle_region(
    api_base: Option<&str>,
    litellm_params: &LitellmParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    if let Some(region) = configured_region(litellm_params) {
        return validated_mantle_region(region.to_string(), MantleRegionCheck::ProviderResolution);
    }
    let base = first_non_empty([api_base])
        .map(str::to_string)
        .or_else(|| env_value(env_lookup, &[MANTLE_API_BASE_ENV]));
    let host_region = base
        .as_deref()
        .and_then(|base| mantle_host_region(base.trim_end_matches('/')))
        .map(str::to_string);
    Ok(host_region
        .or_else(|| env_value(env_lookup, &REGION_ENVS))
        .unwrap_or_else(|| BEDROCK_MANTLE_DEFAULT_REGION.to_string()))
}

/// The base Python's `get_llm_provider` hands the config: the caller's, the env's, or the regional default.
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
    Ok(first_non_empty([api_base])
        .map(str::to_string)
        .or_else(|| env_value(env_lookup, &[MANTLE_API_BASE_ENV]))
        .unwrap_or_else(|| format!("https://bedrock-mantle.{region}.api.aws/v1")))
}

/// Python's `build_mantle_native_messages_url`.
fn build_mantle_native_messages_url(
    api_base: Option<&str>,
    model: &str,
    litellm_params: &LitellmParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let base = dynamic_api_base(api_base, model, litellm_params, env_lookup)?;
    let region = resolve_mantle_region(Some(&base), litellm_params, env_lookup)?;
    let configured = base.trim_end_matches('/');
    let stripped = BASE_SUFFIXES_TO_STRIP
        .iter()
        .find_map(|suffix| configured.strip_suffix(suffix))
        .unwrap_or(configured);
    let host = match mantle_host_region(stripped) {
        Some(_) => format!("https://bedrock-mantle.{region}.api.aws"),
        None => stripped.to_string(),
    };
    Ok(format!(
        "{}/{}",
        strip_trailing_slashes(&host),
        MANTLE_MESSAGES_PATH.trim_start_matches('/')
    ))
}

impl BaseMessagesConfig for BedrockMantleAnthropicMessagesConfig {
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
        build_mantle_native_messages_url(
            api_base.or(litellm_params.api_base.as_deref()),
            model,
            litellm_params,
            env_lookup,
        )
    }

    fn finalize_request(
        &self,
        request: MessagesRequest,
        headers: Headers,
        context: &MessagesTransformContext,
    ) -> Result<(MessagesRequest, Headers), Error> {
        let model_id = mantle_model_id(&request.model);
        transform_mantle_messages_request(
            request,
            model_id,
            headers,
            context,
            &BEDROCK_MANTLE_REQUEST_SETTINGS,
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
        mantle_wire_body(body, mantle_model_id)
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
        if let Some(token) = first_non_empty([api_key])
            .map(str::to_string)
            .or_else(|| env_value(env_lookup, &KEY_ENVS))
        {
            return Ok(bearer_environment(headers, &token));
        }
        let url = self.get_complete_url(None, model, litellm_params, false, env_lookup)?;
        let signing_region = match mantle_host_region(url.trim_end_matches('/')) {
            Some(region) => region.to_string(),
            None => resolve_mantle_region(Some(&url), litellm_params, env_lookup)?,
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

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    fn logged_model(&self, model: &str) -> String {
        split_mantle_region_prefix(model).1.to_string()
    }
}

#[cfg(test)]
mod tests {
    use litellm_auth::AwsParams;
    use litellm_llms_types::formats::messages::MessagesResponse;
    use rstest::rstest;
    use serde_json::{Map, json};

    use super::*;
    use crate::{
        ErrorDetail,
        base_llm::{auth::AuthScheme, messages::context::MessagesModelCapabilities},
        bedrock::messages::mantle_transformation::AMAZON_MANTLE_MESSAGES_CONFIG,
    };

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
    #[case::whitespace_base_is_used("   ", "   /anthropic/v1/messages")]
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
    fn upper_case_public_host_fails_sigv4_but_not_bearer() {
        let litellm_params = params(Some("https://BEDROCK-MANTLE.US-EAST-1.API.AWS"), None);
        assert_eq!(
            BEDROCK_MANTLE_MESSAGES_CONFIG
                .get_complete_url(None, "claude-test", &litellm_params, false, &|_| None)
                .unwrap(),
            "https://bedrock-mantle.US-EAST-1.api.aws/anthropic/v1/messages"
        );
        assert!(matches!(
            environment(
                &BEDROCK_MANTLE_MESSAGES_CONFIG,
                vec![],
                None,
                "claude-test",
                &litellm_params,
                &[]
            ),
            Err(Error::InvalidRequest(ErrorDetail::InvalidMantleRegion {
                region,
                check: MantleRegionCheck::Request,
            })) if region == "US-EAST-1"
        ));
        assert!(
            environment(
                &BEDROCK_MANTLE_MESSAGES_CONFIG,
                vec![],
                Some("key"),
                "claude-test",
                &litellm_params,
                &[]
            )
            .is_ok()
        );
    }

    #[rstest]
    #[case::url_injection(&BEDROCK_MANTLE_MESSAGES_CONFIG, Some("us-east-1.attacker.test/steal?"), &[], MantleRegionCheck::ProviderResolution)]
    #[case::uppercase(&BEDROCK_MANTLE_MESSAGES_CONFIG, Some("US-EAST-1"), &[], MantleRegionCheck::ProviderResolution)]
    #[case::environment(&BEDROCK_MANTLE_MESSAGES_CONFIG, None, &[("BEDROCK_MANTLE_REGION", "US-EAST-1")], MantleRegionCheck::ProviderResolution)]
    #[case::bedrock_mantle_route(&AMAZON_MANTLE_MESSAGES_CONFIG, Some("US-EAST-1"), &[], MantleRegionCheck::Request)]
    #[case::bedrock_mantle_route_environment(&AMAZON_MANTLE_MESSAGES_CONFIG, None, &[("AWS_REGION_NAME", "US-EAST-1")], MantleRegionCheck::Request)]
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
    #[case::bedrock_mantle_route(&AMAZON_MANTLE_MESSAGES_CONFIG)]
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
    #[case::bedrock_mantle_route(&AMAZON_MANTLE_MESSAGES_CONFIG, "mantle/anthropic.claude-test")]
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
    #[case::bedrock_mantle_route(&AMAZON_MANTLE_MESSAGES_CONFIG, false)]
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
    #[case::bedrock_mantle_route(&AMAZON_MANTLE_MESSAGES_CONFIG)]
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
    #[case::bedrock_mantle_route(&AMAZON_MANTLE_MESSAGES_CONFIG)]
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
    #[case::bedrock_mantle(&BEDROCK_MANTLE_MESSAGES_CONFIG)]
    #[case::bedrock_mantle_route(&AMAZON_MANTLE_MESSAGES_CONFIG)]
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
    #[case::bedrock_mantle_route_keeps_region(&AMAZON_MANTLE_MESSAGES_CONFIG, "mantle/eu-west-1/anthropic.claude-test", true, json!({"model": "eu-west-1/anthropic.claude-test", "stream": true}))]
    #[case::stream_false_is_omitted(&AMAZON_MANTLE_MESSAGES_CONFIG, "mantle/anthropic.claude-test", false, json!({"model": "anthropic.claude-test"}))]
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
    #[case::missing(None, json!({"input_tokens": 0, "output_tokens": 0}))]
    #[case::null(Some(json!(null)), json!({"input_tokens": 0, "output_tokens": 0}))]
    #[case::not_an_object(Some(json!([1])), json!({"input_tokens": 0, "output_tokens": 0}))]
    #[case::partial(Some(json!({"output_tokens": 7, "cache_read_input_tokens": 3})), json!({"input_tokens": 0, "output_tokens": 7, "cache_read_input_tokens": 3}))]
    fn response_usage_is_zero_filled(#[case] usage: Option<Value>, #[case] expected: Value) {
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
        let response = MessagesResponse { usage, ..response };
        let usage = BEDROCK_MANTLE_MESSAGES_CONFIG
            .transform_anthropic_messages_response("claude-test", response)
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
}
