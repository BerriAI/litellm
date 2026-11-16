use std::sync::LazyLock;

use litellm_auth_aws::is_bedrock_region;
use litellm_host::interceptors::RetriedRequest;
use litellm_llms_types::{
    formats::messages::{MessagesOptionalParams, MessagesRequest, MessagesResponse},
    providers::anthropic::BetaProvider,
};
use litellm_router_types::LitellmParams;
use serde_json::Value;

use crate::{
    Error,
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
    bedrock::{
        common_utils::validated_region,
        messages::{
            invoke_transformations::{
                anthropic_claude_transformation::{
                    native_output_fallback_litellm_params,
                    should_retry_anthropic_messages_without_native_output,
                },
                request::{InvokeRequestSettings, clamp_adaptive_reasoning_effort_for_bedrock},
            },
            mantle_transformation::{
                AMAZON_MANTLE_REQUEST_SETTINGS, MANTLE_API_BASE_ENV, MANTLE_MESSAGES_PATH,
                bearer_environment, mantle_headers, mantle_secret_names, mantle_wire_body,
                sigv4_environment, transform_mantle_messages_request, with_zero_usage,
            },
        },
    },
};

const BEDROCK_MANTLE_DEFAULT_REGION: &str = "us-east-1";
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

fn first_truthy<'a>(values: impl IntoIterator<Item = Option<&'a str>>) -> Option<&'a str> {
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
        Some((head, tail)) if is_bedrock_region(head) => (Some(head), tail),
        _ => (None, model),
    }
}

fn configured_region(litellm_params: &LitellmParams) -> Option<&str> {
    first_truthy([litellm_params.aws.aws_region_name.as_deref()])
}

/// Python's `resolve_mantle_region`.
fn resolve_mantle_region(
    api_base: Option<&str>,
    litellm_params: &LitellmParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    if let Some(region) = configured_region(litellm_params) {
        return validated_region(region.to_string());
    }
    let base = first_truthy([api_base])
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
    let region = validated_region(
        configured_region(litellm_params)
            .or(split_mantle_region_prefix(model).0)
            .map(str::to_string)
            .or_else(|| env_value(env_lookup, &REGION_ENVS))
            .unwrap_or_else(|| BEDROCK_MANTLE_DEFAULT_REGION.to_string()),
    )?;
    Ok(first_truthy([api_base])
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
        transform_mantle_messages_request(
            request,
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
        mantle_wire_body(body, |model| {
            split_mantle_region_prefix(model)
                .1
                .replacen("mantle/", "", 1)
        })
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
        if let Some(token) = first_truthy([api_key])
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
            validated_region(signing_region)?,
            &litellm_params.aws,
            env_lookup,
        ))
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }
}
