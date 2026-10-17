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
                        return openai(gateway, deployment, body).await;
                    }
                    let (messages, optional_params) = split(body);
                    let call = ChatCompletionsCall {
                        model: deployment.model.clone(),
                        messages,
                        optional_params,
                        api_key: deployment.api_key.clone(),
                        api_base: deployment.api_base.clone(),
                        custom_llm_provider: deployment.custom_llm_provider.clone(),
                        extra_headers: None,
                        timeout: deployment.timeout,
                    };
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

async fn openai(
    gateway: &Gateway,
    deployment: &Deployment,
    body: Map<String, Value>,
) -> Result<Response, Error> {
    let machine = gateway.chat_completions.clone().openai_machine(
        OpenAiChatCall {
            model: deployment.model.clone(),
            body,
            api_key: deployment.api_key.clone(),
            api_base: deployment.api_base.clone(),
            custom_llm_provider: deployment.custom_llm_provider.clone(),
            extra_headers: None,
            timeout: deployment.timeout,
        },
        None,
    );
    let encoder = Sse::<OpenAiChatCompletions, _, _>::new(
        |body: Bytes| ([(CONTENT_TYPE, "application/json")], body),
        |error| Bytes::from(Error::from(error).openai_sse_frame()),
    );
    Ok(litellm_host_http::serve(machine, (), (), encoder, None).await?)
}
