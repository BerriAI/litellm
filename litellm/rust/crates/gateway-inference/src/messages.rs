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
use litellm_cache_response::CachePolicy;
use litellm_host_http::Sse;
use litellm_inference_messages::{MessagesCall, MessagesRoute, messages_body, route::Messages};
use litellm_llms_types::headers::{ProviderSpecificHeader, ProviderSpecificHeaders};
use serde_json::{Map, Value};

use litellm_router_core::Deployment;

use crate::{
    Error, Gateway, JsonObject, RequestId,
    routing::{self, Target},
};

/// Client headers Python forwards to Anthropic-speaking providers on every call.
const ANTHROPIC_API_HEADERS: [&str; 2] = ["anthropic-version", "anthropic-beta"];
const OPENROUTER_API_HEADERS: [&str; 2] = [
    "x-openrouter-metadata",
    "x-openrouter-experimental-metadata",
];
const ANTHROPIC_API_HEADER_PROVIDERS: &str =
    "anthropic,bedrock,bedrock_mantle,vertex_ai,openrouter,deepseek";
const DEPLOYMENT_DEFAULT_PARAMS: [&str; 23] = [
    "max_tokens",
    "metadata",
    "stop_sequences",
    "stream",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "top_k",
    "inference_geo",
    "top_p",
    "mcp_servers",
    "context_management",
    "compaction",
    "container",
    "output_format",
    "speed",
    "output_config",
    "cache_control",
    "reasoning_effort",
    "safeguards",
    "service_tier",
    "thread",
];

pub async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    RequestId(request_id): RequestId,
    headers: HeaderMap,
    body: Result<JsonObject, Error>,
) -> impl IntoResponse {
    let result = match body {
        Ok(JsonObject(body)) => handle(gateway, &identity, &headers, body).await,
        Err(error) => Err(error),
    };
    result.map_err(|error| (error.status(), Json(error.body(request_id.as_deref()))))
}

async fn handle(
    gateway: Arc<Gateway>,
    identity: &AuthenticatedRequest,
    headers: &HeaderMap,
    body: Map<String, Value>,
) -> Result<Response, Error> {
    let (body, cache_options) = crate::caching::prepare(identity, body)?;
    let route = gateway.messages.clone();
    let route = match &gateway.cache {
        Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
            cache.clone(),
            cache_options.scope.clone(),
        )),
        None => route,
    };
    let policy = cache_options.policy;
    let forwarded = anthropic_api_headers(headers);
    routing::call(
        gateway,
        identity,
        "aanthropic_messages",
        "/v1/messages",
        body,
        move |target| attempt(route.clone(), policy, forwarded.clone(), target),
    )
    .await
}

async fn attempt(
    route: MessagesRoute,
    policy: CachePolicy,
    forwarded: Option<ProviderSpecificHeaders>,
    target: Target,
) -> Result<Response, Error> {
    let call = project(target, forwarded)?;
    let machine = route.machine(call, policy);
    let stream =
        Sse::<Messages, _, _>::new(Json, |error| Bytes::from(Error::from(error).sse_frame()));
    let headers = crate::caching::CacheHeaders::default();
    let response = litellm_host_http::serve(machine, (), headers.clone(), stream, None).await?;
    Ok(headers.apply(response))
}

fn project(
    target: Target,
    forwarded: Option<ProviderSpecificHeaders>,
) -> Result<MessagesCall, Error> {
    let api_key = target.api_key();
    let Target {
        deployment,
        body,
        timeout,
        shaping,
    } = target;
    let params = &deployment.params;
    let tools =
        merged_tools(&deployment, body.get("tools")).map(|tools| ("tools".to_owned(), tools));
    let client_params = body.into_iter().map(|(name, value)| match name.as_str() {
        "model" => (name, Value::from(params.model.as_str())),
        _ => (name, value),
    });
    let body = deployment_default_params(&deployment)
        .chain(client_params)
        .chain(tools)
        .collect();
    Ok(MessagesCall {
        body: messages_body(body)?,
        api_key,
        api_base: params.api_base.clone(),
        custom_llm_provider: params.custom_llm_provider.clone(),
        litellm_params: params.clone(),
        extra_headers: None,
        provider_specific_header: forwarded,
        timeout,
        shaping,
    })
}

fn deployment_default_params(
    deployment: &Deployment,
) -> impl Iterator<Item = (String, Value)> + '_ {
    DEPLOYMENT_DEFAULT_PARAMS.iter().filter_map(|name| {
        let value = deployment.params.extra.get(*name)?;
        Some(((*name).to_owned(), value.clone()))
    })
}

fn merged_tools(deployment: &Deployment, client_tools: Option<&Value>) -> Option<Value> {
    let deployment_tools = deployment.params.extra.get("tools")?.as_array()?;
    let client_tools: &[Value] = match client_tools.unwrap_or(&Value::Null) {
        Value::Null => &[],
        Value::Array(tools) => tools,
        _ => return None,
    };
    Some(
        deployment_tools
            .iter()
            .chain(client_tools)
            .cloned()
            .collect(),
    )
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
