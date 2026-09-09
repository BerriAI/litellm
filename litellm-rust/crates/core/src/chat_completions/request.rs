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

pub fn resolve_request(
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
    services: &impl crate::providers::auth::ChatAuthorizationServices,
    request: &ResolvedChatCompletionsRequest<'_>,
    model: &str,
    config: &dyn ChatCompletionsProviderConfig,
) -> Result<(Vec<(String, String)>, ChatCompletionsAuth), Error> {
    let env_lookup = |key: &str| services.environment(key);
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

pub fn build_provider_request(
    request: ResolvedChatCompletionsRequest<'_>,
) -> Result<ProviderChatCompletionsRequest, Error> {
    build_provider_request_with_services(
        crate::providers::auth::native_authorization_services(),
        request,
    )
}

pub fn build_provider_request_with_services(
    services: &impl crate::providers::auth::ChatAuthorizationServices,
    request: ResolvedChatCompletionsRequest<'_>,
) -> Result<ProviderChatCompletionsRequest, Error> {
    let (headers, auth) = validate_environment(services, &request, &request.model, request.config)?;
    let model = request.model;
    let config = request.config;
    let env_lookup = |key: &str| services.environment(key);
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

pub async fn build_pre_call_request(
    request: ChatCompletionsRequest<'_>,
) -> Result<super::types::ChatPreCallRequest, Error> {
    build_pre_call_request_with_services(
        crate::providers::auth::native_authorization_services(),
        request,
    )
    .await
}

pub async fn build_pre_call_request_with_services(
    services: &impl crate::providers::auth::ChatAuthorizationServices,
    request: ChatCompletionsRequest<'_>,
) -> Result<super::types::ChatPreCallRequest, Error> {
    build_resolved_pre_call_request_with_services(services, resolve_request(request)?).await
}

pub(crate) async fn build_resolved_pre_call_request_with_services(
    services: &impl crate::providers::auth::ChatAuthorizationServices,
    request: ResolvedChatCompletionsRequest<'_>,
) -> Result<super::types::ChatPreCallRequest, Error> {
    use super::types::{ChatEndpoint, ChatPreCallRequest};
    use crate::lifecycle::{PreCallBody, RequestBodyPolicy, WireBody};

    let built = build_provider_request_with_services(services, request)?;
    let endpoint = ChatEndpoint {
        model: built.model.clone(),
        config: built.config,
        url: built.url.clone(),
        auth: built.auth.clone(),
        optional_params: built.optional_params.clone(),
        timeout: built.timeout,
    };
    match built.config.request_body_policy() {
        RequestBodyPolicy::StructuredAtSend => {
            let mut generated = built
                .body
                .as_object()
                .cloned()
                .ok_or_else(|| Error::InvalidRequest("chat body must be an object".into()))?;
            let parameter_fields = built
                .optional_params
                .keys()
                .filter(|name| generated.contains_key(*name))
                .cloned()
                .collect::<Vec<_>>();
            for name in &parameter_fields {
                generated.remove(name);
            }
            Ok(ChatPreCallRequest {
                endpoint,
                body: PreCallBody::StructuredAtSend {
                    callback: generated,
                },
                parameter_fields,
                headers: built.upstream_headers,
            })
        }
        RequestBodyPolicy::SerializedAtBuild => {
            let logging_body = serde_json::to_string(&built.body).map_err(|error| {
                Error::InvalidRequest(format!("could not encode chat request: {error}"))
            })?;
            let authorized = built
                .config
                .authorize(
                    services,
                    built.authorization_context(),
                    WireBody::from_serialized(logging_body.clone()),
                )
                .await?;
            let headers = authorized.headers().to_vec();
            Ok(ChatPreCallRequest {
                endpoint,
                body: PreCallBody::SerializedAtBuild {
                    callback: logging_body,
                    authorized,
                },
                parameter_fields: Vec::new(),
                headers,
            })
        }
        RequestBodyPolicy::StructuredAtBuild => Err(Error::Unsupported(
            "chat structured-at-build request body policy",
        )),
    }
}

pub fn unchanged_pre_call_readback(
    request: &super::types::ChatPreCallRequest,
) -> Result<super::types::ChatPreCallReadback, Error> {
    use super::types::ChatPreCallReadback;
    use crate::lifecycle::{PreCallBody, WireBody};

    match &request.body {
        PreCallBody::StructuredAtSend { callback } => {
            let mut body = callback.clone();
            for name in &request.parameter_fields {
                if let Some(value) = request.endpoint.optional_params.get(name) {
                    body.insert(name.clone(), value.clone());
                }
            }
            Ok(ChatPreCallReadback::StructuredAtSend {
                body: WireBody::encode(&body, "chat completions request")?,
                headers: request.headers.clone(),
            })
        }
        PreCallBody::SerializedAtBuild { .. } | PreCallBody::StructuredAtBuild { .. } => {
            Ok(ChatPreCallReadback::CapturedAtBuild {
                headers: request.headers.clone(),
            })
        }
    }
}

pub async fn settle_pre_call_request_with_services(
    services: &dyn crate::providers::auth::ChatAuthorizationServices,
    request: super::types::ChatPreCallRequest,
    readback: super::types::ChatPreCallReadback,
) -> Result<super::types::SettledChatRequest, Error> {
    use super::types::{ChatPreCallReadback, ChatPreCallRequest};
    use crate::lifecycle::PreCallBody;

    let ChatPreCallRequest { endpoint, body, .. } = request;
    match (body, readback) {
        (
            PreCallBody::StructuredAtSend { .. },
            ChatPreCallReadback::StructuredAtSend { body, headers },
        ) => endpoint.authorize(services, body, headers).await,
        (
            PreCallBody::SerializedAtBuild { authorized, .. }
            | PreCallBody::StructuredAtBuild { authorized, .. },
            ChatPreCallReadback::CapturedAtBuild { headers },
        ) => Ok(endpoint.settle_authorized(authorized, headers)),
        _ => Err(Error::InvalidRequest(
            "chat completions pre-call readback did not match its body policy".into(),
        )),
    }
}
