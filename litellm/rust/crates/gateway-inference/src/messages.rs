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
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    Ended, Error, Gateway, JsonObject, RequestId,
    caching::CacheHeaders,
    headers::Routing,
    model_restamp::{ReportedModel, Stamp},
    request,
    routing::{self, Host, RouteAttempt},
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

/// The events that open a Messages stream, up to the one that names its model.
#[derive(Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Opening {
    Ping,
    MessageStart {
        message: Named,
        #[serde(flatten)]
        rest: Map<String, Value>,
    },
    #[serde(other)]
    Other,
}

/// The part of a `message_start` that names the model.
#[derive(Deserialize, Serialize)]
struct Named {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(flatten)]
    rest: Map<String, Value>,
}

/// Where a Messages stream names its model: the `message_start` event, the first one that is
/// not a `ping`, as `AnthropicStreamModelRestamper` reads it.
fn message_start(data: &str, model: &str) -> Stamp {
    match serde_json::from_str(data) {
        Ok(Opening::Ping) => Stamp::Later,
        Ok(Opening::MessageStart { message, rest }) if message.model.as_deref() != Some(model) => {
            let named = Opening::MessageStart {
                message: Named {
                    model: Some(model.to_owned()),
                    ..message
                },
                rest,
            };
            serde_json::to_string(&named).map_or(Stamp::Never, Stamp::Now)
        }
        _ => Stamp::Never,
    }
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
                    stream_outcome::messages_event,
                );
                Served {
                    deferred: true,
                    ..Served::new((CallOutput::Stream { head, chunks }, cache), 0)
                }
            }
        })
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
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::message_start(
        json!({"type": "message_start", "message": {"id": "msg_1", "model": "provider-model"}, "index": 0}),
        Some(Some(json!({"type": "message_start", "message": {"id": "msg_1", "model": "public"}, "index": 0})))
    )]
    #[case::message_start_naming_no_model(
        json!({"type": "message_start", "message": {"id": "msg_1"}}),
        Some(Some(json!({"type": "message_start", "message": {"id": "msg_1", "model": "public"}})))
    )]
    #[case::message_start_already_naming_it(
        json!({"type": "message_start", "message": {"model": "public"}}),
        Some(None)
    )]
    #[case::message_start_without_a_message(json!({"type": "message_start"}), Some(None))]
    #[case::another_event(
        json!({"type": "content_block_delta", "delta": {"text": "message_start"}}),
        Some(None)
    )]
    #[case::a_ping(json!({"type": "ping"}), None)]
    fn the_message_start_event_names_a_streams_model(
        #[case] event: Value,
        #[case] expected: Option<Option<Value>>,
    ) {
        let read = match message_start(&event.to_string(), "public") {
            Stamp::Later => None,
            Stamp::Never => Some(None),
            Stamp::Now(data) => Some(Some(serde_json::from_str::<Value>(&data).unwrap())),
        };

        assert_eq!(read, expected);
    }
}
