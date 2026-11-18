//! `POST /v1/messages`, as the Python proxy's `anthropic_response` serves it.

use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{
    Extension, Json,
    body::Bytes,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use litellm_cache_response::CachePolicy;
use litellm_host::call::CallOutput;
use litellm_host_http::Sse;
use litellm_inference_messages::{
    MessagesBody, MessagesCall, MessagesRoute, MessagesShaping, messages_body, route::Messages,
};
use litellm_llms_types::formats::messages::MessagesOptionalParams;
use litellm_llms_types::headers::{ProviderSpecificHeader, ProviderSpecificHeaders};
use litellm_router_core::{Attempt, Deployment, Op, Served};
use serde_json::{Map, Value};

use crate::{
    Ended, Error, Gateway, JsonObject, RequestId,
    caching::CacheHeaders,
    headers::Routing,
    model_restamp::{ReportedModel, Stamp},
    request,
    routing::{self, Host, RouteAttempt, RouteChunks},
    stream_event::{self, Read},
    stream_outcome::{self, Streamed},
};

/// Client headers Python forwards to Anthropic-speaking providers on every call.
const ANTHROPIC_API_HEADERS: [&str; 2] = ["anthropic-version", "anthropic-beta"];
const OPENROUTER_API_HEADERS: [&str; 2] = [
    "x-openrouter-metadata",
    "x-openrouter-experimental-metadata",
];
const ANTHROPIC_API_HEADER_PROVIDERS: &str =
    "anthropic,bedrock,bedrock_mantle,vertex_ai,openrouter,deepseek";
const TOOLS: &str = "tools";

pub async fn create<H: Host>(
    State(gateway): State<Arc<Gateway<H>>>,
    identity: AuthenticatedRequest,
    RequestId(request_id): RequestId,
    headers: HeaderMap,
    body: Result<JsonObject, Error>,
) -> impl IntoResponse {
    let result = match body {
        Ok(JsonObject(body)) => handle(&gateway, &identity, &headers, body).await,
        Err(error) => Err(error.into()),
    };
    result.map_err(|ended| {
        (
            ended.error.status(),
            ended.routing(),
            Json(ended.error.body(request_id.as_deref())),
        )
    })
}

async fn handle<H: Host>(
    gateway: &Gateway<H>,
    identity: &AuthenticatedRequest,
    headers: &HeaderMap,
    body: Map<String, Value>,
) -> Result<Response, Ended> {
    let model = request::model(&body)?.to_owned();
    let (body, cache_options) = crate::caching::prepare(identity, body)?;
    let route = gateway.messages.clone();
    let route = match &gateway.cache {
        Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
            cache.clone(),
            cache_options.scope.clone(),
        )),
        None => route,
    };
    let attempt = MessagesAttempt {
        gateway,
        route,
        body,
        headers,
        policy: cache_options.policy,
    };
    let routed = routing::call(
        gateway,
        identity,
        Op::ANTHROPIC_MESSAGES,
        "/v1/messages",
        &model,
        &attempt,
    )
    .await?;
    let routing = Routing::served(&routed);
    let reported = ReportedModel::of(&model, &routed);
    let (ran, cache) = routed.output.response;
    let stream =
        Sse::<Messages, _, _>::new(Json, |error| Bytes::from(Error::from(error).sse_frame()));
    let ran = reported.stream(reported.complete(ran), message_start);
    let response = litellm_host_http::render(stream, stream_outcome::bytes(ran))?;
    Ok((Extension(routing), cache.apply(response)).into_response())
}

/// Where a Messages stream names its model: the `message_start` event, the first one that is
/// not a `ping`, as `AnthropicStreamModelRestamper` reads it. The event is edited as the JSON
/// object it was sent as, so its other fields keep their values and their order.
fn message_start(data: &str, model: &str) -> Stamp {
    let Ok(mut event) = serde_json::from_str::<Map<String, Value>>(data) else {
        return Stamp::Never;
    };
    match event.get("type").and_then(Value::as_str) {
        Some("ping") => return Stamp::Later,
        Some("message_start") => {}
        _ => return Stamp::Never,
    }
    let Some(Value::Object(message)) = event.get_mut("message") else {
        return Stamp::Never;
    };
    if message.get("model").and_then(Value::as_str) == Some(model) {
        return Stamp::Never;
    }
    message.insert("model".to_owned(), Value::from(model));
    Stamp::Now(Value::Object(event).to_string())
}

struct MessagesAttempt<'a, H> {
    gateway: &'a Gateway<H>,
    route: MessagesRoute,
    body: Map<String, Value>,
    headers: &'a HeaderMap,
    policy: CachePolicy,
}

impl<H: Host> RouteAttempt for MessagesAttempt<'_, H> {
    type Outcome = (Streamed<Messages>, CacheHeaders);

    async fn run(&self, attempt: &Attempt) -> Result<Served<Self::Outcome>, Error> {
        let deployment = &attempt.deployment;
        let call = MessagesCall {
            timeout: routing::timeout(attempt, routing::streamed(&self.body)),
            ..project(
                deployment,
                self.gateway.shaping(&deployment.id),
                routing::attempt_body(&self.body, attempt),
                self.headers,
            )?
        };
        let machine = self.route.clone().machine(call, self.policy);
        let cache = CacheHeaders::default();
        let ran = litellm_host_http::run(machine, (), cache.clone(), None).await?;
        Ok(match ran {
            CallOutput::Complete(body) => {
                let total_tokens = routing::total_tokens(match &body {
                    MessagesBody::Message(message) => message.usage.as_ref(),
                    MessagesBody::Unchecked(fields) => fields.get("usage"),
                });
                Served::new((CallOutput::Complete(body), cache), total_tokens)
            }
            CallOutput::Stream { head, chunks } => {
                let chunks = stream_outcome::reporting_outcome(
                    self.gateway,
                    deployment,
                    chunks,
                    stream_event::messages_event,
                );
                Served {
                    deferred: true,
                    ..Served::new((CallOutput::Stream { head, chunks }, cache), 0)
                }
            }
        })
    }

    fn held(outcome: &mut Self::Outcome) -> Option<(&mut RouteChunks, Read)> {
        match &mut outcome.0 {
            CallOutput::Stream { chunks, .. } => Some((chunks, stream_event::messages_event)),
            CallOutput::Complete(_) => None,
        }
    }
}

fn project(
    deployment: &Deployment,
    shaping: MessagesShaping,
    body: Map<String, Value>,
    headers: &HeaderMap,
) -> Result<MessagesCall, Error> {
    let tools = merged_tools(deployment, body.get(TOOLS)).map(|tools| (TOOLS.to_owned(), tools));
    let params = &deployment.params;
    let client_params = body.into_iter().map(|(name, value)| match name.as_str() {
        "model" => (name, Value::from(params.model.as_str())),
        _ => (name, value),
    });
    let body = deployment_default_params(deployment)
        .chain(client_params)
        .chain(tools)
        .collect();
    Ok(MessagesCall {
        body: messages_body(body)?,
        api_key: routing::api_key(deployment),
        api_base: params.api_base.clone(),
        custom_llm_provider: params.custom_llm_provider.clone(),
        litellm_params: params.clone(),
        extra_headers: None,
        provider_specific_header: anthropic_api_headers(headers),
        timeout: None,
        shaping,
    })
}

fn deployment_default_params(
    deployment: &Deployment,
) -> impl Iterator<Item = (String, Value)> + '_ {
    MessagesOptionalParams::fields()
        .filter(|name| *name != TOOLS)
        .filter_map(|name| {
            let value = deployment.params.extra.get(name)?;
            Some((name.to_owned(), value.clone()))
        })
}

fn merged_tools(deployment: &Deployment, client_tools: Option<&Value>) -> Option<Value> {
    let deployment_tools = deployment.params.extra.get(TOOLS)?.as_array()?;
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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::message_start(
        r#"{"type":"message_start","index":0,"message":{"id":"msg_1","model":"provider-model","role":"assistant"}}"#,
        Some(Some(
            r#"{"type":"message_start","index":0,"message":{"id":"msg_1","model":"public","role":"assistant"}}"#
        ))
    )]
    #[case::message_start_naming_no_model(
        r#"{"type":"message_start","message":{"id":"msg_1"}}"#,
        Some(Some(r#"{"type":"message_start","message":{"id":"msg_1","model":"public"}}"#))
    )]
    #[case::message_start_already_naming_it(
        r#"{"type":"message_start","message":{"model":"public"}}"#,
        Some(None)
    )]
    #[case::message_start_without_a_message(r#"{"type":"message_start"}"#, Some(None))]
    #[case::another_event(
        r#"{"type":"content_block_delta","delta":{"text":"message_start"}}"#,
        Some(None)
    )]
    #[case::not_json("[DONE]", Some(None))]
    #[case::a_ping(r#"{"type":"ping"}"#, None)]
    fn the_message_start_event_names_a_streams_model_and_nothing_else_changes(
        #[case] event: &str,
        #[case] expected: Option<Option<&str>>,
    ) {
        let read = match message_start(event, "public") {
            Stamp::Later => None,
            Stamp::Never => Some(None),
            Stamp::Now(data) => Some(Some(data)),
        };

        assert_eq!(read.as_ref().map(|named| named.as_deref()), expected);
    }
}
