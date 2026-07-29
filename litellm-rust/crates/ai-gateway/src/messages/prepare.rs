use litellm_core::CoreError;
use litellm_core::CoreResult;
use litellm_core::messages::transformation::MessagesAuthStrategy;
use litellm_core::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};
use serde_json::Value;

use super::common_utils::{has_bearer_auth, has_header, messages_provider_config, string_headers};
use super::types::{MessagesRequest, ProviderMessagesRequest};
use crate::constants::BEDROCK_MESSAGES_PROVIDER;

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

    let is_bedrock = provider == BEDROCK_MESSAGES_PROVIDER;
    let auth_strategy = if is_bedrock {
        MessagesAuthStrategy::Header("authorization")
    } else {
        config.auth_strategy()
    };
    let bearer_token = if is_bedrock {
        request
            .api_key
            .map(str::to_string)
            .or_else(|| env_lookup("AWS_BEARER_TOKEN_BEDROCK"))
            .filter(|token| !token.trim().is_empty())
    } else {
        None
    };
    let auth_header = match auth_strategy {
        MessagesAuthStrategy::Bearer
            if has_header(&headers, "authorization")
                || (config.accepts_bearer_auth() && has_bearer_auth(&headers)) =>
        {
            None
        }
        MessagesAuthStrategy::Header(name)
            if has_header(&headers, name)
                || (config.accepts_bearer_auth() && has_bearer_auth(&headers)) =>
        {
            None
        }
        MessagesAuthStrategy::Bearer => {
            let api_key = config.resolve_api_key(request.api_key, &env_lookup)?;
            Some(("authorization".to_string(), format!("Bearer {api_key}")))
        }
        MessagesAuthStrategy::Header(name) => {
            let api_key = config.resolve_api_key(request.api_key, &env_lookup)?;
            Some((name.to_string(), api_key))
        }
    };
    if let Some(header) = auth_header {
        headers.push(header);
    }

    for (name, value) in config.default_headers() {
        if !has_header(&headers, name) {
            headers.push((name.to_string(), value.to_string()));
        }
    }

    let _stream = request.body.get("stream").and_then(Value::as_bool) == Some(true);
    let url = config.complete_url(request.api_base, &model, &env_lookup)?;
    let signing_region = config.signing_region(request.api_base, &env_lookup);
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
        signing_region,
        bearer_token,
        timeout: request.timeout,
    })
}
