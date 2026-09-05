use serde_json::Value;

use crate::error::Error;
use crate::http_utils::has_header;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::common_utils::{chat_completions_provider_config, string_headers};
use super::transformation::{ChatCompletionsAuth, ChatCompletionsProviderConfig};
use super::types::{
    ChatCompletionsRequest, ChatMessage, ProviderChatCompletionsRequest,
    ResolvedChatCompletionsRequest,
};

pub(super) fn resolve_provider_config<'a>(
    model: &'a str,
    custom_llm_provider: Option<&'a str>,
) -> Result<(String, &'static dyn ChatCompletionsProviderConfig), Error> {
    let provider_info = get_custom_llm_provider(model, custom_llm_provider)
        .or_else(|| {
            custom_llm_provider.map(|provider| CustomLlmProvider {
                model,
                custom_llm_provider: provider,
            })
        })
        .ok_or_else(|| {
            Error::InvalidProvider(
                "unable to resolve custom_llm_provider for chat completions request".to_string(),
            )
        })?;
    let config = chat_completions_provider_config(provider_info.custom_llm_provider)
        .ok_or_else(|| Error::InvalidProvider(provider_info.custom_llm_provider.to_string()))?;
    Ok((provider_info.model.to_string(), config))
}

pub(super) fn parse_messages(messages: Value) -> Result<Vec<ChatMessage>, Error> {
    serde_json::from_value(messages)
        .map_err(|err| Error::InvalidRequest(format!("invalid chat completions messages: {err}")))
}

pub(super) fn resolve_request(
    request: ChatCompletionsRequest<'_>,
) -> Result<ResolvedChatCompletionsRequest<'_>, Error> {
    let (model, config) = resolve_provider_config(request.model, request.custom_llm_provider)?;
    let messages = parse_messages(request.messages)?;
    if messages.is_empty() {
        return Err(Error::InvalidRequest(
            "chat completions requires at least one message".to_string(),
        ));
    }
    if let Some(reason) = config.unsupported_reason(&messages, &request.optional_params) {
        return Err(Error::Unsupported(reason.0));
    }
    Ok(ResolvedChatCompletionsRequest {
        model,
        config,
        messages,
        optional_params: request.optional_params,
        api_key: request.api_key,
        api_base: request.api_base,
        extra_headers: request.extra_headers,
        timeout: request.timeout,
    })
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
fn validate_environment(
    request: &ResolvedChatCompletionsRequest<'_>,
    model: &str,
    config: &dyn ChatCompletionsProviderConfig,
) -> Result<(Vec<(String, String)>, ChatCompletionsAuth), Error> {
    let env_lookup = |key: &str| std::env::var(key).ok();
    let mut headers = string_headers(request.extra_headers.clone())?;
    let auth = config.auth(
        request.api_key,
        model,
        &request.optional_params,
        &env_lookup,
    )?;
    match &auth {
        ChatCompletionsAuth::Header { name, value } => {
            // The deployment's credential replaces whatever the caller forwarded
            // under the same name, mirroring Python's
            // `{**headers, **anthropic_headers}`: letting a request header win
            // would let its sender choose the principal the call bills to.
            //
            // The exception is a scheme the provider hands off to entirely, such
            // as an Anthropic OAuth bearer, where Python drops `x-api-key`
            // instead of resolving one. Re-adding it there would put the
            // credential into a header the host removed on purpose.
            if !config.defers_to_forwarded_auth(&headers) {
                headers.retain(|(header, _)| !header.eq_ignore_ascii_case(name));
                headers.push(((*name).to_string(), value.clone()));
            }
        }
        ChatCompletionsAuth::Bearer { token } => {
            // Bedrock's `get_request_headers` assigns `headers["Authorization"]`
            // unconditionally once a bearer token resolves, so the deployment's
            // identity outranks whatever the caller forwarded. Keeping the
            // caller's would bill and authorize the call as a different
            // principal than the same deployment uses on Python.
            //
            // The `Header` arm below keeps the opposite precedence on purpose:
            // Anthropic's transform honours a forwarded OAuth bearer.
            headers.retain(|(name, _)| !name.eq_ignore_ascii_case("authorization"));
            headers.push(("authorization".to_string(), format!("Bearer {token}")));
        }
        // SigV4 signs the serialized body, so the handler adds its headers.
        ChatCompletionsAuth::AwsSigV4 { .. } => {}
    }

    for (name, value) in config.default_headers() {
        if !has_header(&headers, name) {
            headers.push(((*name).to_string(), (*value).to_string()));
        }
    }
    Ok((headers, auth))
}

pub(super) fn prepare_provider_request(
    request: ResolvedChatCompletionsRequest<'_>,
) -> Result<ProviderChatCompletionsRequest, Error> {
    let (headers, auth) = validate_environment(&request, &request.model, request.config)?;
    let model = request.model;
    let config = request.config;
    let env_lookup = |key: &str| std::env::var(key).ok();
    let url = config.complete_url(
        request.api_base,
        &model,
        &request.optional_params,
        &env_lookup,
    )?;
    let transformed =
        config.transform_request(&model, request.messages, request.optional_params.clone())?;

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
