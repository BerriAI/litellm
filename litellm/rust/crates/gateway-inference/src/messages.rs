//! `POST /v1/messages`, as the Python proxy's `anthropic_response` serves it.

use axum::{Json, body::Bytes, http::HeaderMap, response::Response};
use futures_util::StreamExt;
use litellm_cache_response::CachePolicy;
use litellm_framer::sse;
use litellm_host::call::CallOutput;
use litellm_host_http::Sse;
use litellm_inference_messages::{
    MessagesBody, MessagesCall, MessagesRoute, MessagesSettings, MessagesShaping, messages_body,
    route::Messages,
};
use litellm_llms_types::formats::chat_completions::ChatCompletionsUsage;
use litellm_llms_types::formats::messages::{
    MessagesOptionalParams,
    streaming::{MessagesStreamEvent, MessagesStreamUsage},
};
use litellm_llms_types::headers::{ProviderSpecificHeader, ProviderSpecificHeaders};
use litellm_python_compat::{flags::lax_bool, json::from_json};
use litellm_router_core::{Attempt, Deployment, Op};
use serde_json::{Map, Value};

use crate::{
    Error, Gateway, JsonObject,
    caching::CacheHeaders,
    error::ErrorFormat,
    hooks::CallType,
    model_restamp::{ReportedModel, Stamp},
    routing::{self, Host, RouteAttempt, RouteChunks},
    serve::{self, Call, Endpoint},
    stream_event::{Counts, Kind, Read, Reading},
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
/// The keys a request sets its own messages settings by. Python reads them from the call's kwargs
/// and never sends them on.
pub const REQUEST_SETTINGS: [&str; 2] = ["drop_params", "additional_drop_params"];

pub async fn create<H: Host>(call: Call<H>, body: Result<JsonObject, Error>) -> Response {
    serve::run(&call, body.map(|JsonObject(body)| body), |body| {
        MessagesAttempt::new(&call, body)
    })
    .await
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

impl<'a, H: Host> MessagesAttempt<'a, H> {
    fn new(call: &'a Call<H>, body: Map<String, Value>) -> Result<Self, Error> {
        let gateway = &*call.gateway;
        let (body, cache_options) = crate::caching::prepare(&call.identity, body)?;
        let route = gateway.messages.clone();
        let route = match &gateway.cache {
            Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
                cache.clone(),
                cache_options.scope,
            )),
            None => route,
        };
        Ok(Self {
            gateway,
            route,
            body,
            headers: &call.headers,
            policy: cache_options.policy,
        })
    }
}

impl<H: Host> Endpoint for MessagesAttempt<'_, H> {
    const CALL_TYPE: CallType = CallType::AnthropicMessages;
    const OP: Op = Op::ANTHROPIC_MESSAGES;
    const PATH: &'static str = "/v1/messages";
    const ERRORS: ErrorFormat = ErrorFormat::Anthropic;

    fn counts((ran, _): &Self::Outcome) -> Option<Counts> {
        let usage = match ran {
            CallOutput::Complete(MessagesBody::Message(message)) => message.usage.as_ref(),
            CallOutput::Complete(MessagesBody::Unchecked(fields)) => fields.get("usage"),
            CallOutput::Stream { .. } => None,
        }?;
        serde_json::from_value::<MessagesStreamUsage>(usage.clone())
            .ok()
            .map(Counts::from)
    }

    fn usage(counts: Counts) -> Option<ChatCompletionsUsage> {
        Some(counts.messages_usage())
    }

    fn render((ran, cache): Self::Outcome, reported: ReportedModel<'_>) -> Result<Response, Error> {
        let stream =
            Sse::<Messages, _, _>::new(Json, |error| Bytes::from(Error::from(error).sse_frame()));
        let ran = reported.answer(ran, message_start);
        let response = litellm_host_http::render(stream, stream_outcome::bytes(ran))?;
        Ok(cache.apply(response))
    }
}

impl<H: Host> RouteAttempt for MessagesAttempt<'_, H> {
    type Outcome = (Streamed<Messages>, CacheHeaders);

    async fn run(&self, attempt: &Attempt) -> Result<Self::Outcome, Error> {
        let deployment = &attempt.deployment;
        let shaping = self.gateway.shaping(&deployment.id);
        let call = MessagesCall {
            timeout: routing::timeout(
                attempt,
                routing::streamed(&self.body),
                shaping.settings.request_timeout(),
            ),
            ..project(
                deployment,
                shaping,
                routing::attempt_body(&self.body, attempt),
                self.headers,
            )?
        };
        let machine = self.route.clone().machine(call, self.policy);
        let cache = CacheHeaders::default();
        let ran = litellm_host_http::run(machine, (), cache.clone(), None).await?;
        let ran = match ran {
            CallOutput::Complete(body) => CallOutput::Complete(body),
            CallOutput::Stream { head, chunks } => CallOutput::Stream {
                head,
                chunks: sse::blocks(chunks).boxed(),
            },
        };
        Ok((ran, cache))
    }

    fn total_tokens((ran, _): &Self::Outcome) -> u64 {
        let usage = match ran {
            CallOutput::Complete(MessagesBody::Message(message)) => message.usage.as_ref(),
            CallOutput::Complete(MessagesBody::Unchecked(fields)) => fields.get("usage"),
            CallOutput::Stream { .. } => None,
        };
        routing::total_tokens(usage)
    }

    fn held(outcome: &mut Self::Outcome) -> Option<(&mut RouteChunks, Read)> {
        match &mut outcome.0 {
            CallOutput::Stream { chunks, .. } => Some((chunks, read_event)),
            CallOutput::Complete(_) => None,
        }
    }
}

fn project(
    deployment: &Deployment,
    deployment_shaping: MessagesShaping,
    body: Map<String, Value>,
    headers: &HeaderMap,
) -> Result<MessagesCall, Error> {
    let shaping = requested(deployment_shaping, &body);
    let tools = merged_tools(deployment, body.get(TOOLS)).map(|tools| (TOOLS.to_owned(), tools));
    let params = &deployment.params;
    let client_params = body
        .into_iter()
        .filter(|(name, _)| !REQUEST_SETTINGS.contains(&name.as_str()))
        .map(|(name, value)| match name.as_str() {
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

/// The deployment's settings with the request's own laid over them, as Python lays a call's
/// kwargs over the deployment's (`{**deployment, **request}`). The request's `drop_params` turns
/// dropping on or off for its call, and `litellm_settings.drop_params` keeps it on whatever the
/// request says (`litellm.drop_params or drop_params`).
fn requested(shaping: MessagesShaping, body: &Map<String, Value>) -> MessagesShaping {
    let settings = shaping.settings;
    let drops = body
        .get("drop_params")
        .map(|requested| settings.global_drop_params || is_on(requested));
    let additional_drop_params = body.get("additional_drop_params").map(drop_paths);
    MessagesShaping {
        settings: MessagesSettings {
            drop_params: drops.unwrap_or(settings.drop_params),
            drops_unsupported_params: drops.unwrap_or(settings.drops_unsupported_params),
            additional_drop_params: additional_drop_params
                .unwrap_or(settings.additional_drop_params),
            ..settings
        },
        ..shaping
    }
}

/// `lax_bool(value) is True`, the rule Python reads a call's `drop_params` by.
fn is_on(value: &Value) -> bool {
    lax_bool(&from_json(value.clone())) == Some(true)
}

/// `additional_drop_paths`: the text entries of a list, and none for any other value.
pub fn drop_paths(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|paths| {
            paths
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
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

/// An Anthropic Messages event. `message_start` counts the input and `message_delta` the output,
/// only a `content_block_delta` carries generated output (`is_anthropic_content_delta_chunk`),
/// and an `error` event is answered as its error type is before a stream, with the event's own
/// data as the body (`parse_anthropic_error_event`).
pub(crate) fn read_event(data: &str) -> Reading {
    match serde_json::from_str(data) {
        Ok(MessagesStreamEvent::MessageStart { message }) => Reading::lifecycle(message.usage),
        Ok(
            MessagesStreamEvent::MessageDelta { usage, .. }
            | MessagesStreamEvent::MessageStop { usage, .. },
        ) => Reading::lifecycle(usage),
        Ok(MessagesStreamEvent::ContentBlockDelta { .. }) => Reading::of(Kind::Content),
        Ok(MessagesStreamEvent::Error { error }) => Reading::of(Kind::Failed {
            status: error.error_type.status(),
            answer: data.to_owned(),
        }),
        Ok(
            MessagesStreamEvent::ContentBlockStart { .. }
            | MessagesStreamEvent::ContentBlockStop { .. }
            | MessagesStreamEvent::Ping
            | MessagesStreamEvent::Other(_),
        )
        | Err(_) => Reading::of(Kind::Lifecycle),
    }
}

impl From<MessagesStreamUsage> for Counts {
    fn from(usage: MessagesStreamUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cache_creation_input_tokens: usage.cache_creation_input_tokens,
            cache_read_input_tokens: usage.cache_read_input_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn deployment_shaping(global: bool, drop_params: bool) -> MessagesShaping {
        MessagesShaping {
            settings: MessagesSettings {
                global_drop_params: global,
                drop_params,
                drops_unsupported_params: drop_params,
                additional_drop_params: vec!["deployment".to_owned()],
                ..MessagesSettings::default()
            },
            ..MessagesShaping::default()
        }
    }

    #[rstest]
    #[case::the_deployments_when_the_request_names_none(json!({}), false, true, true)]
    #[case::a_request_turning_it_on(json!({"drop_params": true}), false, false, true)]
    #[case::a_flag_word(json!({"drop_params": " yes "}), false, false, true)]
    #[case::a_number_that_is_no_flag(json!({"drop_params": 2}), false, true, false)]
    #[case::a_request_turning_it_off(json!({"drop_params": false}), false, true, false)]
    #[case::null_turning_it_off(json!({"drop_params": null}), false, true, false)]
    #[case::the_config_over_a_request_turning_it_off(
        json!({"drop_params": false}),
        true,
        true,
        true
    )]
    fn a_requests_drop_params_lays_over_the_deployments(
        #[case] body: Value,
        #[case] global: bool,
        #[case] deployment: bool,
        #[case] expected: bool,
    ) {
        let settings = requested(
            deployment_shaping(global, deployment),
            body.as_object().unwrap(),
        )
        .settings;

        assert_eq!(
            (settings.drop_params, settings.drops_unsupported_params),
            (expected, expected)
        );
    }

    #[rstest]
    #[case::the_deployments_when_the_request_names_none(json!({}), &["deployment"])]
    #[case::the_requests_in_its_place(
        json!({"additional_drop_params": ["request", 1, "nested.path"]}),
        &["request", "nested.path"]
    )]
    #[case::none_for_null(json!({"additional_drop_params": null}), &[])]
    #[case::none_for_text(json!({"additional_drop_params": "request"}), &[])]
    fn a_requests_additional_drop_params_replace_the_deployments(
        #[case] body: Value,
        #[case] expected: &[&str],
    ) {
        let settings =
            requested(deployment_shaping(false, false), body.as_object().unwrap()).settings;

        assert_eq!(settings.additional_drop_params, expected);
    }

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

    /// A provider's stream goes out as it was sent, so the reader takes events Anthropic itself
    /// would never send: a strict reading would hold back content or lose the usage they carry.
    #[rstest]
    #[case::a_start_counts_its_input(
        r#"{"type":"message_start","message":{"usage":{"input_tokens":7}}}"#,
        Kind::Lifecycle,
        Some(7)
    )]
    #[case::a_closing_delta_counts_its_output(
        r#"{"type":"message_delta","delta":{},"usage":{"output_tokens":5}}"#,
        Kind::Lifecycle,
        Some(5)
    )]
    #[case::a_ping_says_nothing(r#"{"type":"ping"}"#, Kind::Lifecycle, None)]
    #[case::a_block_opening_is_not_content_yet(
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        Kind::Lifecycle,
        None
    )]
    #[case::a_text_delta_is_content(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        Kind::Content,
        None
    )]
    #[case::a_delta_of_a_type_not_named_yet_is_content(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"future_delta","payload":1}}"#,
        Kind::Content,
        None
    )]
    #[case::a_known_delta_with_a_field_of_another_shape_is_content(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":["hi"]}}"#,
        Kind::Content,
        None
    )]
    #[case::an_error_without_a_type_is_no_failure(r#"{"type":"error"}"#, Kind::Lifecycle, None)]
    #[case::data_that_is_no_event_says_nothing("[DONE]", Kind::Lifecycle, None)]
    fn a_messages_event_that_is_no_error_says_whether_content_began_and_what_it_counts(
        #[case] data: &str,
        #[case] kind: Kind,
        #[case] total_tokens: Option<u64>,
    ) {
        let reading = read_event(data);

        assert_eq!(
            (reading.kind, reading.usage.map(Counts::total)),
            (kind, total_tokens)
        );
    }

    #[rstest]
    #[case::overloaded(
        r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#,
        529
    )]
    #[case::without_a_message(r#"{"type":"error","error":{"type":"overloaded_error"}}"#, 529)]
    #[case::refused(
        r#"{"type":"error","error":{"type":"invalid_request_error","message":"no"}}"#,
        400
    )]
    #[case::of_a_type_not_named_yet(
        r#"{"type":"error","error":{"type":"future_error","message":"busy"}}"#,
        500
    )]
    fn a_messages_error_event_is_answered_as_its_type_is_before_a_stream_with_its_own_data(
        #[case] data: &str,
        #[case] status: u16,
    ) {
        assert_eq!(
            read_event(data),
            Reading::of(Kind::Failed {
                status,
                answer: data.to_owned()
            })
        );
    }
}
