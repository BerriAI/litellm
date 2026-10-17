//! `POST /v1/messages`, as the Python proxy's `anthropic_response` serves it.

use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use litellm_host_http::Sse;
use litellm_inference_messages::{MessagesCall, messages_body, route::Messages};
use litellm_llms_types::headers::{ProviderSpecificHeader, ProviderSpecificHeaders};
use serde_json::{Map, Value};

use crate::{Deployment, Error, Gateway, JsonBody, RequestId, request};

/// Client headers Python forwards to Anthropic-speaking providers on every call.
const ANTHROPIC_API_HEADERS: [&str; 2] = ["anthropic-version", "anthropic-beta"];
const OPENROUTER_API_HEADERS: [&str; 2] = [
    "x-openrouter-metadata",
    "x-openrouter-experimental-metadata",
];
const ANTHROPIC_API_HEADER_PROVIDERS: &str =
    "anthropic,bedrock,bedrock_mantle,vertex_ai,openrouter,deepseek";

pub async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    RequestId(request_id): RequestId,
    headers: HeaderMap,
    body: Result<JsonBody, Error>,
) -> impl IntoResponse {
    let result = match body {
        Ok(body) => handle(&gateway, &identity, &headers, body).await,
        Err(error) => Err(error),
    };
    result.map_err(|error| (error.status(), Json(error.body(request_id.as_deref()))))
}

async fn handle(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    headers: &HeaderMap,
    JsonBody { object, raw }: JsonBody,
) -> Result<Response, Error> {
    gateway.check_body(&object)?;
    let group = request::resolve_group(gateway, &object)?;
    let name = request::model_name(&object)?.to_owned();
    let (body, cache_options) = crate::caching::prepare(identity, object)?;
    let reparse = || -> Result<Map<String, Value>, Error> {
        Ok(crate::caching::prepare(identity, request::object(&raw)?)?.0)
    };
    let route = gateway.messages.clone();
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
                async move {
                    crate::observe::record(name, attempt.deployment);
                    request::authorize(identity, name, attempt.deployment).await?;
                    let call = project(attempt.deployment, body?, headers)?;
                    let machine = route.machine(call, cache_options.policy);
                    let stream = Sse::<Messages, _, _>::new(Json, |error| {
                        Bytes::from(Error::from(error).sse_frame())
                    });
                    Ok(litellm_host_http::serve(machine, (), cache_headers, stream, None).await?)
                }
            },
            Error::failure,
        )
        .await
        .ok_or_else(|| Error::UnknownModel(name.clone()))??;
    Ok(cache_headers.apply(request::hold(routed)))
}

fn project(
    deployment: &Deployment,
    body: Map<String, Value>,
    headers: &HeaderMap,
) -> Result<MessagesCall, Error> {
    let body = body
        .into_iter()
        .map(|(name, value)| match name.as_str() {
            "model" => (name, Value::from(deployment.model.as_str())),
            _ => (name, value),
        })
        .collect();
    Ok(MessagesCall {
        body: messages_body(body)?,
        api_key: deployment.api_key.clone(),
        api_base: deployment.api_base.clone(),
        custom_llm_provider: deployment.custom_llm_provider.clone(),
        litellm_params: deployment.litellm_params.clone(),
        extra_headers: None,
        provider_specific_header: anthropic_api_headers(headers),
        timeout: deployment.timeout,
        shaping: deployment.shaping.clone(),
    })
}

fn anthropic_api_headers(headers: &HeaderMap) -> Option<ProviderSpecificHeaders> {
    let scopes = [
        (
            ANTHROPIC_API_HEADER_PROVIDERS,
            ANTHROPIC_API_HEADERS.as_slice(),
        ),
        ("openrouter", OPENROUTER_API_HEADERS.as_slice()),
    ];
    let forwarded: Vec<ProviderSpecificHeader> = scopes
        .into_iter()
        .filter_map(|(provider, names)| {
            let extra_headers: Map<String, Value> = names
                .iter()
                .filter_map(|name| {
                    let value = headers.get(*name)?.to_str().ok()?;
                    Some(((*name).to_owned(), Value::from(value)))
                })
                .collect();
            (!extra_headers.is_empty()).then(|| ProviderSpecificHeader {
                custom_llm_provider: provider.into(),
                extra_headers,
            })
        })
        .collect();
    (!forwarded.is_empty()).then_some(ProviderSpecificHeaders::Many(forwarded))
}
