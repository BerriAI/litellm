use std::sync::Arc;

use axum::{Json, body::Bytes, extract::State, response::Response};
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_host_http::Sse;
use litellm_inference_responses::{route::Responses, types::ResponsesCall};
use serde_json::{Map, Value};

use crate::{Error, Gateway, JsonObject, request};

const DEPLOYMENT_DEFAULT_PARAMS: &[&str] = &[
    "include",
    "instructions",
    "max_output_tokens",
    "parallel_tool_calls",
    "previous_response_id",
    "reasoning",
    "store",
    "background",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_p",
    "truncation",
    "user",
    "service_tier",
    "safety_identifier",
    "prompt",
    "max_tool_calls",
    "prompt_cache_key",
    "prompt_cache_retention",
    "prompt_cache_options",
    "moderation",
    "stream_options",
    "top_logprobs",
    "partial_images",
    "context_management",
];

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    JsonObject(body): JsonObject,
) -> Result<Response, Error> {
    let deployment = request::resolve_deployment(&gateway, "/responses", &body)?;
    request::authorize_model(&identity, deployment, &body).await?;
    let (body, cache_options) = crate::caching::prepare(&identity, body)?;
    let body: Map<String, Value> =
        request::deployment_defaults(deployment, DEPLOYMENT_DEFAULT_PARAMS)
            .chain(body)
            .collect();
    let route = gateway.responses.clone();
    let route = match &gateway.cache {
        Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
            cache.clone(),
            cache_options.scope.clone(),
        )),
        None => route,
    };

    let call = ResponsesCall {
        model: deployment.model.clone(),
        input: body.get("input").cloned().unwrap_or_default(),
        optional_params: body
            .into_iter()
            .filter(|(name, _)| !matches!(name.as_str(), "model" | "input"))
            .collect(),
        api_key: deployment.api_key.clone(),
        api_base: deployment.api_base.clone(),
        custom_llm_provider: deployment.custom_llm_provider.clone(),
        extra_headers: None,
        timeout: deployment.timeout,
    };
    let machine = route.machine(call, cache_options.policy);
    let stream = Sse::<Responses, _, _>::new(Json, |error| {
        Bytes::from(Error::from(error).responses_sse_frame())
    });
    let headers = crate::caching::CacheHeaders::default();
    let response = litellm_host_http::serve(machine, (), headers.clone(), stream, None).await?;
    Ok(headers.apply(response))
}
