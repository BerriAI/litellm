use crate::error::Error;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::common_utils::{has_bearer_auth, has_header, messages_provider_config, string_headers};
use super::transformation::{AnthropicMessagesProviderConfig, MessagesAuthStrategy};
use super::types::{MessagesEndpoint, MessagesOptions, MessagesRequest, ProviderMessagesRequest};
use serde_json::{Map, Value};

pub fn build_provider_request(request: MessagesRequest) -> Result<ProviderMessagesRequest, Error> {
    build_provider_request_with_environment(request, &|key: &str| std::env::var(key).ok())
}

pub(crate) fn build_provider_request_with_environment(
    request: MessagesRequest,
    environment: &dyn Fn(&str) -> Option<String>,
) -> Result<ProviderMessagesRequest, Error> {
    let endpoint = build_endpoint_with_environment(
        MessagesOptions {
            model: request.model,
            api_key: request.api_key,
            api_base: request.api_base,
            custom_llm_provider: request.custom_llm_provider,
            extra_headers: request.extra_headers,
            timeout: request.timeout,
        },
        environment,
    )?;
    let transformed = endpoint.config.transform_request(request.body)?;
    let body = serde_json::to_value(transformed).map_err(|err| {
        Error::InvalidRequest(format!(
            "failed to serialize Anthropic messages request: {err}"
        ))
    })?;
    let authorized = endpoint.capture_body(body)?;
    let headers = authorized.headers().to_vec();
    Ok(endpoint.settle(authorized, headers))
}

pub fn build_endpoint(request: MessagesOptions) -> Result<MessagesEndpoint, Error> {
    build_endpoint_with_environment(request, &|key: &str| std::env::var(key).ok())
}

fn build_endpoint_with_environment(
    request: MessagesOptions,
    environment: &dyn Fn(&str) -> Option<String>,
) -> Result<MessagesEndpoint, Error> {
    let provider_info =
        get_custom_llm_provider(&request.model, request.custom_llm_provider.as_deref())
            .or_else(|| {
                request
                    .custom_llm_provider
                    .as_deref()
                    .map(|provider| CustomLlmProvider {
                        model: &request.model,
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
    let headers = validate_environment(
        config,
        request.extra_headers,
        request.api_key.as_deref(),
        environment,
    )?;

    let url = config.complete_url(request.api_base.as_deref(), &model, environment)?;

    Ok(MessagesEndpoint {
        provider: provider.to_string(),
        model,
        config,
        url,
        headers,
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
