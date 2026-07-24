use litellm_core::CoreError;
use litellm_core::CoreResult;
use litellm_core::messages::transformation::MessagesAuthKind;
use litellm_core::messages::transformation::MessagesAuthStrategy;
use litellm_core::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};
use serde_json::Value;

use super::common_utils::{has_bearer_auth, has_header, messages_provider_config, string_headers};
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

    let stream = request.body.get("stream").and_then(Value::as_bool) == Some(true);
    let auth_kind = config.auth_kind(&model, &env_lookup)?;
    let headers = match &auth_kind {
        MessagesAuthKind::AwsSigV4 { .. } => Vec::new(),
        MessagesAuthKind::ApiKey {
            strategy,
            accepts_bearer,
        } => {
            let mut headers = string_headers(request.extra_headers)?;
            let already_authorized = has_header(&headers, strategy.header_name())
                || (*accepts_bearer && has_bearer_auth(&headers));
            if !already_authorized {
                let api_key = config.resolve_api_key(request.api_key, &env_lookup)?;
                let auth_header = match strategy {
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
            headers
        }
    };
    let url = config.complete_url(request.api_base, &model, stream, &env_lookup)?;
    let typed_request = serde_json::from_value(request.body).map_err(|err| {
        CoreError::InvalidRequest(format!("invalid Anthropic messages request: {err}"))
    })?;
    let body = config.upstream_body(typed_request)?;

    Ok(ProviderMessagesRequest {
        model,
        config,
        auth_kind,
        streaming: config.streaming(),
        url,
        body,
        upstream_headers: headers,
        timeout: request.timeout,
    })
}
