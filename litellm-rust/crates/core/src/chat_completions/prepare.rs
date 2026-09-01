use serde_json::Value;

use crate::error::{CoreError, CoreResult};
use crate::http_utils::has_header;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::common_utils::{chat_completions_provider_config, string_headers};
use super::transformation::{ChatCompletionsAuth, ChatCompletionsProviderConfig};
use super::types::{ChatCompletionsRequest, ChatMessage, ProviderChatCompletionsRequest};

pub(super) fn resolve_provider_config<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
) -> CoreResult<(String, &'static dyn ChatCompletionsProviderConfig)> {
    let provider_info = get_custom_llm_provider(model, custom_llm_provider)
        .or_else(|| {
            custom_llm_provider.map(|provider| CustomLlmProvider {
                model,
                custom_llm_provider: provider,
            })
        })
        .ok_or_else(|| {
            CoreError::InvalidProvider(
                "unable to resolve custom_llm_provider for chat completions request".to_string(),
            )
        })?;
    let config = chat_completions_provider_config(provider_info.custom_llm_provider)
        .ok_or_else(|| CoreError::InvalidProvider(provider_info.custom_llm_provider.to_string()))?;
    Ok((provider_info.model.to_string(), config))
}

pub(super) fn parse_messages(messages: Value) -> CoreResult<Vec<ChatMessage>> {
    serde_json::from_value(messages).map_err(|err| {
        CoreError::InvalidRequest(format!("invalid chat completions messages: {err}"))
    })
}

pub(super) fn prepare_chat_completions_call(
    request: ChatCompletionsRequest<'_>,
) -> CoreResult<ProviderChatCompletionsRequest> {
    let (model, config) = resolve_provider_config(request.model, request.custom_llm_provider)?;
    let env_lookup = |key: &str| std::env::var(key).ok();

    let messages = parse_messages(request.messages)?;
    if messages.is_empty() {
        return Err(CoreError::InvalidRequest(
            "chat completions requires at least one message".to_string(),
        ));
    }
    if let Some(reason) = config.unsupported_reason(&messages, &request.optional_params) {
        return Err(CoreError::Unsupported(reason.0));
    }

    let mut headers = string_headers(request.extra_headers)?;
    let auth = config.auth(
        request.api_key,
        &model,
        &request.optional_params,
        &env_lookup,
    )?;
    match &auth {
        ChatCompletionsAuth::Header { name, value } => {
            if !config.defers_to_forwarded_auth(&headers) {
                headers.retain(|(header, _)| !header.eq_ignore_ascii_case(name));
                headers.push(((*name).to_string(), value.clone()));
            }
        }
        ChatCompletionsAuth::Bearer { token } => {
            headers.retain(|(name, _)| !name.eq_ignore_ascii_case("authorization"));
            headers.push(("authorization".to_string(), format!("Bearer {token}")));
        }
        ChatCompletionsAuth::AwsSigV4 { .. } => {}
    }

    for (name, value) in config.default_headers() {
        if !has_header(&headers, name) {
            headers.push(((*name).to_string(), (*value).to_string()));
        }
    }

    let url = config.complete_url(
        request.api_base,
        &model,
        &request.optional_params,
        &env_lookup,
    )?;
    let transformed =
        config.transform_request(&model, messages, request.optional_params.clone())?;

    Ok(ProviderChatCompletionsRequest {
        model,
        config,
        url,
        body: transformed.body,
        upstream_headers: headers,
        auth,
        optional_params: request.optional_params,
        timeout: request.timeout,
    })
}
