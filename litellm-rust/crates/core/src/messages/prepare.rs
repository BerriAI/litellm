use litellm_core_utils::{
    dot_notation_indexing::delete_nested_value,
    get_llm_provider_logic::{CustomLlmProvider, get_custom_llm_provider},
};
use litellm_llms::{
    anthropic::common_utils::{
        flatten_unencrypted_web_search_results, sanitize_tool_use_ids, strip_empty_content_blocks,
        strip_provider_specific_fields,
    },
    base_llm::anthropic_messages::transformation::MessagesTransformContext,
};
use litellm_types::llms::anthropic_messages::anthropic_request::{
    AnthropicMessage, AnthropicMessagesRequest,
};
use serde_json::{Map, Value, json};

use super::{
    Error,
    common_utils::{messages_provider_config, string_headers},
};
use crate::messages::types::{MessagesRequest, MessagesShaping, ProviderMessagesRequest};

pub(super) fn prepare_provider_request(
    request: MessagesRequest<'_>,
) -> Result<ProviderMessagesRequest, Error> {
    let provider_info = get_custom_llm_provider(request.model, request.custom_llm_provider)
        .or_else(|| {
            request
                .custom_llm_provider
                .map(|provider| CustomLlmProvider {
                    model: request.model,
                    custom_llm_provider: provider,
                })
        })
        .ok_or_else(|| {
            Error::InvalidProvider(
                "unable to resolve custom_llm_provider for messages request".to_string(),
            )
        })?;
    let model = provider_info.model.to_string();
    let provider = provider_info.custom_llm_provider;

    let config = messages_provider_config(provider)
        .ok_or_else(|| Error::InvalidProvider(provider.to_string()))?;
    let env_lookup = |key: &str| std::env::var(key).ok();

    let body = request
        .shaping
        .additional_drop_params
        .iter()
        .fold(request.body, |body, path| delete_nested_value(body, path));
    let typed_request: AnthropicMessagesRequest = serde_json::from_value(body).map_err(|err| {
        Error::InvalidRequest(format!("invalid Anthropic messages request: {err}"))
    })?;
    let sanitized = sanitize_request(
        AnthropicMessagesRequest {
            model: model.clone(),
            ..typed_request
        },
        &request.shaping,
    );
    let transformed = config.transform_anthropic_messages_request(
        sanitized,
        &MessagesTransformContext::new(request.shaping.capabilities, request.shaping.drop_params),
    )?;

    let forwarded = string_headers(request.extra_headers)?;
    let authenticated = config.authenticate(forwarded, request.api_key, &env_lookup)?;
    let headers = config.request_headers(
        with_default_headers(authenticated, config.default_headers()),
        &transformed,
    );

    let body = serde_json::to_value(transformed).map_err(|err| {
        Error::InvalidRequest(format!(
            "failed to serialize Anthropic messages request: {err}"
        ))
    })?;

    let url = config.get_complete_url(request.api_base, &model, &env_lookup)?;

    Ok(ProviderMessagesRequest {
        provider: provider.to_string(),
        model,
        config,
        url,
        body,
        upstream_headers: headers,
        timeout: request.timeout,
    })
}

/// The route-level cleanup Python's `anthropic_messages` runs before any provider config:
/// history sanitizers, the `metadata` allowlist and the reasoning auto summary.
fn sanitize_request(
    request: AnthropicMessagesRequest,
    shaping: &MessagesShaping,
) -> AnthropicMessagesRequest {
    AnthropicMessagesRequest {
        messages: sanitize_messages(request.messages),
        metadata: request.metadata.as_ref().map(allowed_metadata),
        thinking: with_reasoning_auto_summary(request.thinking, shaping.reasoning_auto_summary),
        ..request
    }
}

fn sanitize_messages(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    strip_provider_specific_fields(flatten_unencrypted_web_search_results(
        sanitize_tool_use_ids(strip_empty_content_blocks(messages)),
    ))
}

/// Only the fields Anthropic's `metadata` accepts reach the provider; LiteLLM-specific
/// metadata travels under `litellm_metadata` instead.
fn allowed_metadata(metadata: &Value) -> Value {
    let user_id = metadata.get("user_id").filter(|value| !value.is_null());
    Value::Object(
        user_id
            .map(|value| ("user_id".to_string(), value.clone()))
            .into_iter()
            .collect::<Map<String, Value>>(),
    )
}

fn with_reasoning_auto_summary(thinking: Option<Value>, enabled: bool) -> Option<Value> {
    let Some(Value::Object(thinking)) = thinking else {
        return thinking;
    };
    if !enabled || thinking.get("type").and_then(Value::as_str) == Some("disabled") {
        return Some(Value::Object(thinking));
    }
    Some(Value::Object(
        thinking
            .into_iter()
            .filter(|(key, _)| key != "display")
            .chain([("display".to_string(), json!("summarized"))])
            .collect(),
    ))
}

fn with_default_headers(
    headers: Vec<(String, String)>,
    defaults: &[(&str, &str)],
) -> Vec<(String, String)> {
    let missing: Vec<(String, String)> = defaults
        .iter()
        .filter(|(name, _)| {
            !headers
                .iter()
                .any(|(header, _)| header.eq_ignore_ascii_case(name))
        })
        .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
        .collect();
    headers.into_iter().chain(missing).collect()
}
