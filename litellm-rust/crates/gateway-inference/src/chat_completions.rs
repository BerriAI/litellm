use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{
    Json,
    body::Bytes,
    extract::{Path, State},
    http::{HeaderMap, header::CONTENT_TYPE},
    response::{IntoResponse, Response},
};
use litellm_host_http::Sse;
use litellm_inference_chat::{
    ChatCompletionsRoute,
    openai::{OpenAiChatCall, OpenAiChatCompletions},
    stream::{ChatCompletionsStream, ChatStreamCall, ReportedUsage},
    types::ChatCompletionsCall,
};
use serde_json::{Map, Value};

use litellm_gateway_hooks::Route;

use crate::{Deployment, Error, Gateway, JsonBody, request};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    headers: HeaderMap,
    body: JsonBody,
) -> Result<impl IntoResponse, Error> {
    handle(&gateway, &identity, &headers, body, None).await
}

pub(crate) async fn create_from_model_path(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    Path(model): Path<String>,
    headers: HeaderMap,
    body: JsonBody,
) -> Result<impl IntoResponse, Error> {
    handle(&gateway, &identity, &headers, body, Some(&model)).await
}

/// The body, naming the model from the request path when it names none itself.
fn with_model(body: Map<String, Value>, model: Option<&str>) -> Map<String, Value> {
    match (body.get("model"), model) {
        (None | Some(Value::Null), Some(model)) => body
            .into_iter()
            .chain([("model".into(), Value::from(model))])
            .collect(),
        _ => body,
    }
}

async fn handle(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    headers: &HeaderMap,
    JsonBody { object, raw }: JsonBody,
    model: Option<&str>,
) -> Result<Response, Error> {
    let body = with_model(object, model);
    gateway.check_body(&body)?;
    let group = request::resolve_group(gateway, &body)?;
    let name = request::model_name(&body)?.to_owned();
    let tracking = crate::hooks::admit(
        gateway,
        identity,
        Route::ChatCompletions,
        headers,
        &raw,
        &body,
        &name,
    )
    .await?;
    let (mut body, cache_options) = crate::caching::prepare(identity, body)?;
    let edit = |body: &mut Map<String, Value>| {
        if let Some(tracking) = &tracking {
            tracking.edit_body(body);
        }
    };
    edit(&mut body);
    let reparse = || -> Result<Map<String, Value>, Error> {
        let body = with_model(request::object(&raw)?, model);
        let mut body = crate::caching::prepare(identity, body)?.0;
        edit(&mut body);
        Ok(body)
    };
    let route = gateway.chat_completions.clone();
    let route = match &gateway.cache {
        Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
            cache.clone(),
            cache_options.scope.clone(),
        )),
        None => route,
    };

    // Chat forwards no client headers of its own; hooks may set some.
    let extra_headers = tracking
        .as_ref()
        .and_then(|tracking| tracking.edit_headers(Vec::new()))
        .filter(|headers| !headers.is_empty())
        .map(|headers| {
            headers
                .into_iter()
                .map(|(name, value)| (name, Value::from(value)))
                .collect::<Map<String, Value>>()
        });
    let extra_headers = &extra_headers;
    let cache_headers = crate::caching::CacheHeaders::default();
    let mut body = Some(body);
    let routed = gateway
        .models
        .route_where(
            group,
            request::usable(identity),
            |attempt| {
                // Every attempt hands its body to the provider. The first takes the parsed
                // body and a retry parses the request again, so no request clones a body
                // that can hold a million tokens.
                let body = body.take().map_or_else(&reparse, Ok);
                let route = route.clone();
                let cache_headers = cache_headers.clone();
                let name = name.as_str();
                if let Some(tracking) = &tracking {
                    tracking.attempt(name, &attempt);
                }
                async move {
                    let deployment = attempt.deployment;
                    crate::observe::record(name, deployment);
                    request::authorize(identity, name, deployment).await?;
                    let body = body?;
                    if ChatCompletionsRoute::serves_openai(
                        &deployment.model,
                        deployment.custom_llm_provider.as_deref(),
                    ) {
                        return openai(gateway, deployment, body, extra_headers.clone()).await;
                    }
                    let (messages, optional_params) = split(body);
                    let stream = ChatCompletionsRoute::is_stream(&optional_params);
                    let call = ChatCompletionsCall {
                        model: deployment.model.clone(),
                        messages,
                        optional_params,
                        api_key: deployment.api_key.clone(),
                        api_base: deployment.api_base.clone(),
                        custom_llm_provider: deployment.custom_llm_provider.clone(),
                        extra_headers: extra_headers.clone(),
                        timeout: deployment.timeout,
                        drop_params: Some(drop_params(gateway, deployment)),
                        model_info: gateway.model_info(),
                    };
                    if stream {
                        return translated_stream(route, call, name).await;
                    }
                    Ok(litellm_host_http::serve_unary(
                        route.machine(call, cache_options.policy),
                        (),
                        cache_headers,
                        litellm_host_http::Unary::new(Json),
                        None,
                    )
                    .await?)
                }
            },
            Error::failure,
        )
        .await
        .ok_or_else(|| Error::UnknownModel(name.clone()))
        .and_then(|routed| routed)
        .map(|routed| cache_headers.apply(request::hold(routed)));
    match tracking {
        Some(tracking) => tracking.finish(routed).await,
        None => routed,
    }
}

/// The `messages` array and the remaining optional params of a chat request.
fn split(mut body: Map<String, Value>) -> (Value, Map<String, Value>) {
    let messages = body.remove("messages").unwrap_or_default();
    body.remove("model");
    (messages, body)
}

/// The deployment's `litellm_params` a provider reads beyond its connection: its
/// `api_version`, `base_model` and every key the config adds.
fn provider_params(deployment: &Deployment) -> Map<String, Value> {
    let params = &deployment.litellm_params;
    let mut provider_params: Map<String, Value> = params
        .extra
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    if let Some(api_version) = &params.api_version {
        provider_params.insert("api_version".into(), api_version.clone().into());
    }
    if let Some(base_model) = &deployment.base_model {
        provider_params.insert("base_model".into(), base_model.clone().into());
    }
    provider_params
}

/// `litellm.drop_params` or the deployment's own: either one set drops what the provider
/// cannot take, as Python's `get_optional_params` reads them.
fn drop_params(gateway: &Gateway, deployment: &Deployment) -> bool {
    gateway.drop_params()
        || deployment
            .litellm_params
            .drop_params
            .as_ref()
            .is_some_and(|flag| flag.is_set())
}

async fn openai(
    gateway: &Gateway,
    deployment: &Deployment,
    body: Map<String, Value>,
    extra_headers: Option<Map<String, Value>>,
) -> Result<Response, Error> {
    let machine = gateway.chat_completions.clone().openai_machine(
        OpenAiChatCall {
            model: deployment.model.clone(),
            body,
            api_key: deployment.api_key.clone(),
            api_base: deployment.api_base.clone(),
            custom_llm_provider: deployment.custom_llm_provider.clone(),
            extra_headers,
            timeout: deployment.timeout,
            litellm_params: provider_params(deployment),
            drop_params: drop_params(gateway, deployment),
            model_info: gateway.model_info(),
        },
        None,
    );
    let encoder = Sse::<OpenAiChatCompletions, _, _>::new(
        |body: Bytes| ([(CONTENT_TYPE, "application/json")], body),
        |error| Bytes::from(Error::from(error).openai_sse_frame()),
    );
    Ok(litellm_host_http::serve(machine, (), (), encoder, None).await?)
}

/// A stream from a provider chat translates (Anthropic, Bedrock), sent as OpenAI chunks. The
/// usage it ends with rides on the response for the hooks, even when the client did not ask
/// to see it.
async fn translated_stream(
    route: ChatCompletionsRoute,
    call: ChatCompletionsCall,
    model: &str,
) -> Result<Response, Error> {
    let usage = ReportedUsage::default();
    let machine = route.stream_machine(
        ChatStreamCall {
            call,
            response_model: model.to_owned(),
            usage: usage.clone(),
        },
        None,
    );
    let encoder = Sse::<ChatCompletionsStream, _, _>::new(
        |body: Bytes| ([(CONTENT_TYPE, "application/json")], body),
        |error| Bytes::from(Error::from(error).openai_sse_frame()),
    );
    let mut response = litellm_host_http::serve(machine, (), (), encoder, None).await?;
    response.extensions_mut().insert(usage);
    Ok(response)
}
