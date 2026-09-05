use crate::error::Error;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::common_utils::{has_bearer_auth, has_header, messages_provider_config, string_headers};
use super::transformation::{AnthropicMessagesProviderConfig, MessagesAuthStrategy};
use super::types::{MessagesRequest, ProviderMessagesRequest};
use serde_json::{Map, Value};

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

    let headers =
        validate_environment(config, request.extra_headers, request.api_key, &env_lookup)?;

    let typed_request = serde_json::from_value(request.body).map_err(|err| {
        Error::InvalidRequest(format!("invalid Anthropic messages request: {err}"))
    })?;
    let transformed = config.transform_request(typed_request)?;
    let body = serde_json::to_value(transformed).map_err(|err| {
        Error::InvalidRequest(format!(
            "failed to serialize Anthropic messages request: {err}"
        ))
    })?;

    let url = config.complete_url(request.api_base, &model, &env_lookup)?;

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

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
fn validate_environment(
    config: &dyn AnthropicMessagesProviderConfig,
    extra_headers: Option<Map<String, Value>>,
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<(String, String)>, Error> {
    let mut headers = string_headers(extra_headers)?;

    let auth_strategy = config.auth_strategy();
    let already_authorized = has_header(&headers, auth_strategy.header_name())
        || (config.accepts_bearer_auth() && has_bearer_auth(&headers));
    if !already_authorized {
        let api_key = config.resolve_api_key(api_key, env_lookup)?;
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

    Ok(headers)
}
