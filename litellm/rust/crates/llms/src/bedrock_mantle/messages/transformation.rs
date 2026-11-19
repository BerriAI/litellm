//! Bedrock Mantle's Anthropic Messages endpoint on the `bedrock_mantle/` provider, Python's
//! `BedrockMantleAnthropicMessagesConfig`: the `bedrock/mantle/` shaping with this provider's own
//! region order, base suffix stripping, `BEDROCK_MANTLE_API_KEY` bearer auth, SigV4 signed in the
//! URL's region, and the capability keys both Mantle routes probe.

use std::sync::LazyLock;

use litellm_host::interceptors::RetriedRequest;
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest, MessagesResponse},
    providers::anthropic::BetaProvider,
};
use litellm_router_types::LitellmParams;
use serde_json::{Map, Value};
use url::Url;

use crate::{
    Error, MantleRegionCheck,
    anthropic::messages::{
        handler::shape_anthropic_messages_request,
        transformation::retry_anthropic_messages_after_http_error,
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
            request::{
                InvokeRequestSettings, clamp_adaptive_reasoning_effort_for_bedrock,
                with_logged_tools,
            },
        },
        mantle_transformation::{
            AMAZON_MANTLE_REQUEST_SETTINGS, MANTLE_API_BASE_ENV, MANTLE_MESSAGES_PATH,
            amazon_mantle_model_id, bearer_environment, mantle_headers, mantle_secret_names,
            mantle_wire_body, parse_mantle_base, sigv4_environment,
            transform_mantle_messages_request, unchecked_with_zero_usage, validated_mantle_region,
            with_mantle_messages_path, with_zero_usage,
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

/// The `bedrock_mantle/` Messages config for Claude models. Its region comes from
/// `aws_region_name`, a public Mantle host, a region prefix on the model or the env, in Python's
/// order, and it sends the model without that prefix.
pub struct BedrockMantleAnthropicMessagesConfig;
pub const BEDROCK_MANTLE_MESSAGES_CONFIG: BedrockMantleAnthropicMessagesConfig =
    BedrockMantleAnthropicMessagesConfig;

fn env_value(env_lookup: &dyn Fn(&str) -> Option<String>, names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| env_lookup(name).filter(|value| !value.trim().is_empty()))
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
        _ => routed.to_string(),
    })
}

/// Python's `BedrockModelInfo._model_has_route_prefix`.
fn has_route_prefix(model: &str, prefix: &str) -> bool {
    model.starts_with(prefix) || model.contains(&format!("/{prefix}"))
}

fn configured_region(litellm_params: &LitellmParams) -> Option<&str> {
    litellm_params
        .aws
        .aws_region_name
        .as_deref()
        .filter(|region| !region.is_empty())
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

/// The base Python's `get_llm_provider` hands the config: the caller's, the env's, or the regional
/// default. A blank `api_base` counts as unset.
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
        .map(String::from)
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
    #[case::string_stream_dropped(json!({"model": "mantle/m", "stream": "true", "max_tokens": 1}), json!({"max_tokens": 1, "model": "m"}))]
    #[case::not_an_object(json!(["mantle/m"]), json!(["mantle/m"]))]
    fn wire_body_keeps_only_a_boolean_true_stream(#[case] body: Value, #[case] expected: Value) {
        assert_eq!(AMAZON_MANTLE_MESSAGES_CONFIG.wire_body(body), expected);
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
    #[case::bedrock_mantle_route_number(&AMAZON_MANTLE_MESSAGES_CONFIG, json!(0), "a number")]
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
    #[case::bedrock_mantle_route(&AMAZON_MANTLE_MESSAGES_CONFIG, true)]
    #[case::bedrock_invoke(&crate::bedrock::messages::invoke_transformations::anthropic_claude_transformation::BEDROCK_ANTHROPIC_MESSAGES_CONFIG, false)]
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
}
