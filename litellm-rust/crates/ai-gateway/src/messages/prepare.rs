use litellm_core::CoreError;
use litellm_core::CoreResult;
use litellm_core::messages::transformation::MessagesAuthStrategy;
use litellm_core::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::common_utils::{has_header, messages_provider_config, string_headers};
use super::types::{MessagesRequest, ProviderMessagesRequest};

pub(super) fn prepare_messages_call(
    request: MessagesRequest<'_>,
) -> CoreResult<ProviderMessagesRequest> {
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
            CoreError::InvalidProvider(
                "unable to resolve custom_llm_provider for messages request".to_string(),
            )
        })?;
    let model = provider_info.model.to_string();
    let provider = provider_info.custom_llm_provider;

    let config = messages_provider_config(provider)
        .ok_or_else(|| CoreError::InvalidProvider(provider.to_string()))?;
    let env_lookup = |key: &str| std::env::var(key).ok();

    let mut headers = string_headers(request.extra_headers)?;

    let auth_strategy = config.auth_strategy();
    if !has_header(&headers, auth_strategy.header_name()) {
        let api_key = config.resolve_api_key(request.api_key, &env_lookup)?;
        let auth_header = match auth_strategy {
            MessagesAuthStrategy::Bearer => {
                ("authorization".to_string(), format!("Bearer {api_key}"))
            }
            MessagesAuthStrategy::Header(name) => (name.to_string(), api_key),
        };
        headers.push(auth_header);
    }

    for (name, value) in config.default_headers() {
        if !has_header(&headers, name) {
            headers.push((name.to_string(), value.to_string()));
        }
    }

    let url = config.complete_url(request.api_base, &model, &env_lookup)?;
    let typed_request = serde_json::from_value(request.body).map_err(|err| {
        CoreError::InvalidRequest(format!("invalid Anthropic messages request: {err}"))
    })?;
    let transformed = config.transform_request(typed_request)?;
    let body = serde_json::to_value(transformed).map_err(|err| {
        CoreError::InvalidRequest(format!(
            "failed to serialize Anthropic messages request: {err}"
        ))
    })?;

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
