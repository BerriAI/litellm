//! `POST /v1/messages`, as the Python proxy's `anthropic_response` serves it.

use std::time::Duration;

use axum::{Json, body::Bytes, http::HeaderMap, response::Response};
use futures_util::StreamExt;
use litellm_cache_response::CachePolicy;
use litellm_framer::sse;
use litellm_host::call::CallOutput;
use litellm_host_http::Sse;
use litellm_inference_messages::{
    MessagesBody, MessagesCall, MessagesRoute, MessagesSettings, MessagesShaping, VertexShaping,
    messages_body, route::Messages,
};
use litellm_llms::anthropic::model_capabilities::messages_route_capabilities;
use litellm_llms::vertex_ai::common_utils::supported_regions;
use litellm_llms_types::formats::chat_completions::ChatCompletionsUsage;
use litellm_llms_types::formats::messages::{
    MessagesOptionalParams,
    streaming::{MessagesStreamEvent, MessagesStreamUsage},
};
use litellm_llms_types::headers::{ProviderSpecificHeader, ProviderSpecificHeaders};
use litellm_model_catalog::Catalog;
use litellm_router_core::{Attempt, Deployment, Op};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{
    Checked, Error, Gateway, JsonObject,
    coding_agent::CodingAgent,
    drop_params::{Sources, drops, written},
    error::ErrorFormat,
    hooks::CallType,
    model_restamp::{ReportedModel, Stamp},
    pricing,
    record::{self, Recorder},
    routing::{self, Host, Prompt, RouteAttempt, RouteChunks},
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
const VERTEX_AI: &str = "vertex_ai";
/// The keys a request sets its own messages settings by. Python reads them from the call's kwargs
/// and never sends them on.
pub const REQUEST_SETTINGS: [&str; 2] = ["drop_params", "additional_drop_params"];

/// What `litellm_settings` sets for a messages call to any deployment.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct Configured {
    /// `litellm_settings.drop_params`, unset when neither the config nor its variable wrote a
    /// flag.
    pub drop_params: Option<bool>,
    pub reasoning_auto_summary: bool,
    pub request_timeout: Option<f64>,
    pub detached_drain_timeout: Option<f64>,
    pub detached_drain_limit: Option<usize>,
}

impl Configured {
    fn request_timeout(&self) -> Option<Duration> {
        self.request_timeout
            .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
            .filter(|timeout| !timeout.is_zero())
    }
}

pub async fn create<H: Host>(Checked { call, body }: Checked<H, JsonObject>) -> Response {
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
    client: Option<CodingAgent>,
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
            client: CodingAgent::sent(&call.headers),
            policy: cache_options.policy,
        })
    }
}

impl<H: Host> Endpoint for MessagesAttempt<'_, H> {
    const CALL_TYPE: CallType = CallType::AnthropicMessages;
    const OP: Op = Op::ANTHROPIC_MESSAGES;
    const PATH: &'static str = "/v1/messages";
    const ERRORS: ErrorFormat = ErrorFormat::Anthropic;

    fn usage(counts: Counts) -> Option<ChatCompletionsUsage> {
        Some(counts.messages_usage())
    }

    fn response_id(ran: &Self::Output) -> Option<String> {
        match ran {
            CallOutput::Complete(MessagesBody::Message(message)) => Some(message.id.clone()),
            CallOutput::Complete(MessagesBody::Unchecked(fields)) => {
                fields.get("id").and_then(Value::as_str).map(str::to_owned)
            }
            CallOutput::Stream { .. } => None,
        }
    }

    fn stream_response_id(data: &str) -> Option<String> {
        match serde_json::from_str(data) {
            Ok(MessagesStreamEvent::MessageStart { message }) => message.id,
            _ => None,
        }
    }

    fn configured_timeout<G: Host>(gateway: &Gateway<G>, _: &str) -> Option<Duration> {
        gateway.configured.request_timeout()
    }

    fn render(ran: Self::Output, reported: ReportedModel<'_>) -> Result<Response, Error> {
        let stream =
            Sse::<Messages, _, _>::new(Json, |error| Bytes::from(Error::from(error).sse_frame()));
        let ran = reported.answer(ran, message_start);
        Ok(litellm_host_http::render(
            stream,
            stream_outcome::bytes(ran),
        )?)
    }
}

impl<H: Host> RouteAttempt for MessagesAttempt<'_, H> {
    type Output = Streamed<Messages>;

    fn prompt(&self) -> Option<Prompt<'_>> {
        Some(Prompt::Messages(&self.body))
    }

    async fn run(&self, attempt: &Attempt) -> Result<Self::Output, Error> {
        let deployment = &attempt.deployment;
        let cost_map = self.gateway.cost_map();
        let body = routing::attempt_body(&self.body, attempt);
        let mut shaping = shaping(
            deployment,
            cost_map.catalog(),
            &self.gateway.configured,
            &body,
            self.client,
        );
        shaping.settings.pricing = pricing::priced_as(
            deployment,
            cost_map.catalog(),
            self.gateway.secrets.as_ref(),
        )
        .await?;
        let call = MessagesCall {
            timeout: routing::timeout(
                deployment,
                attempt.timeout,
                <Self as Endpoint>::streamed(&self.body),
                shaping.settings.request_timeout(),
            ),
            ..project(deployment, shaping, body, self.headers)?
        };
        let machine = self.route.clone().machine(call, self.policy);
        let ran = litellm_host_http::run(machine, (), Recorder, None).await?;
        Ok(match ran {
            CallOutput::Complete(body) => CallOutput::Complete(body),
            CallOutput::Stream { head, chunks } => {
                record::stream_head(head.headers.clone());
                CallOutput::Stream {
                    head,
                    chunks: sse::blocks(chunks).boxed(),
                }
            }
        })
    }

    fn counts(ran: &Self::Output) -> Option<Counts> {
        counts(ran)
    }

    fn held(ran: &mut Self::Output) -> Option<(&mut RouteChunks, Read)> {
        match ran {
            CallOutput::Stream { chunks, .. } => Some((chunks, read_event)),
            CallOutput::Complete(_) => None,
        }
    }
}

/// How the messages route shapes a call from `client` to `deployment` with `body`: the
/// `configured` settings, the capabilities `catalog` and its `model_info` declare, whether the
/// call [`drops`] what the provider rejects, the deployment's `additional_drop_params` with the
/// request's own in their place, as Python lays a call's kwargs over the deployment's
/// (`{**deployment, **request}`), except that a request's `null` is not written and leaves the
/// deployment's, and for Claude on Vertex AI the model's `supported_regions`, which route the call
/// as they route it in Python.
fn shaping(
    deployment: &Deployment,
    catalog: &Catalog,
    configured: &Configured,
    body: &Map<String, Value>,
    client: Option<CodingAgent>,
) -> MessagesShaping {
    let params = &deployment.params;
    let base_model = params.extra.get("base_model").and_then(Value::as_str);
    MessagesShaping {
        settings: MessagesSettings {
            drop_params: drops(&Sources {
                request: body.get("drop_params").and_then(written),
                deployment: params.drop_params,
                global: configured.drop_params,
                client,
            }),
            reasoning_auto_summary: configured.reasoning_auto_summary,
            request_timeout: configured.request_timeout,
            detached_drain_timeout: configured.detached_drain_timeout,
            detached_drain_limit: configured.detached_drain_limit,
            additional_drop_params: body
                .get("additional_drop_params")
                .filter(|sent| !sent.is_null())
                .or_else(|| params.extra.get("additional_drop_params"))
                .map(drop_paths)
                .unwrap_or_default(),
            base_model: base_model.map(str::to_owned),
            ..MessagesSettings::default()
        },
        capabilities: messages_route_capabilities(
            catalog,
            &params.model,
            params.custom_llm_provider.as_deref(),
            base_model,
            &|capability| {
                deployment
                    .model_info
                    .raw
                    .get(capability)
                    .and_then(Value::as_bool)
            },
        ),
        vertex: match deployment.provider() {
            VERTEX_AI => VertexShaping {
                supported_regions: supported_regions(catalog, &params.model),
            },
            _ => VertexShaping::default(),
        },
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

/// `additional_drop_paths`: the text entries of a list, and none for any other value.
fn drop_paths(value: &Value) -> Vec<String> {
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

/// What a complete answer's `usage` counts. A stream's is read from its events.
fn counts(ran: &Streamed<Messages>) -> Option<Counts> {
    let usage = match ran {
        CallOutput::Complete(MessagesBody::Message(message)) => message.usage.as_ref(),
        CallOutput::Complete(MessagesBody::Unchecked(fields)) => fields.get("usage"),
        CallOutput::Stream { .. } => None,
    }?;
    MessagesStreamUsage::deserialize(usage)
        .ok()
        .map(Counts::from)
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
    use crate::test_support::deployment;

    /// What each source of `drop_params` holds for one call.
    #[derive(Default)]
    struct Written {
        configured: Option<bool>,
        deployment: Option<Value>,
        request: Option<Value>,
        client: Option<CodingAgent>,
    }

    fn shaped(sources: Written) -> MessagesSettings {
        let params = json!({"model": "provider/model", "additional_drop_params": ["deployment"]})
            .as_object()
            .cloned()
            .into_iter()
            .flatten()
            .chain(
                sources
                    .deployment
                    .map(|flag| ("drop_params".to_owned(), flag)),
            )
            .collect();
        let body = sources
            .request
            .and_then(|body| body.as_object().cloned())
            .unwrap_or_default();
        let configured = Configured {
            drop_params: sources.configured,
            ..Configured::default()
        };

        shaping(
            &deployment(Value::Object(params), json!({})),
            &catalog(),
            &configured,
            &body,
            sources.client,
        )
        .settings
    }

    #[rstest]
    #[case::nothing_written(Written::default(), false)]
    #[case::the_setting(Written { configured: Some(true), ..Written::default() }, true)]
    #[case::the_deployment(Written { deployment: Some(json!(true)), ..Written::default() }, true)]
    #[case::the_deployments_flag_word(
        Written { deployment: Some(json!(" True ")), ..Written::default() },
        true
    )]
    #[case::the_deployments_word_that_is_no_flag(
        Written { deployment: Some(json!("maybe")), configured: Some(true), ..Written::default() },
        true
    )]
    #[case::the_request(
        Written { request: Some(json!({"drop_params": "yes"})), ..Written::default() },
        true
    )]
    #[case::the_request_over_the_deployment(
        Written {
            deployment: Some(json!(true)),
            request: Some(json!({"drop_params": false})),
            ..Written::default()
        },
        false
    )]
    #[case::a_request_null_under_the_deployment(
        Written {
            deployment: Some(json!(true)),
            request: Some(json!({"drop_params": null})),
            ..Written::default()
        },
        true
    )]
    #[case::the_deployment_over_the_setting(
        Written {
            configured: Some(true),
            deployment: Some(json!(false)),
            ..Written::default()
        },
        false
    )]
    #[case::a_coding_agent(Written { client: Some(CodingAgent::ClaudeCode), ..Written::default() }, true)]
    #[case::a_coding_agent_to_a_deployment_that_turns_it_off(
        Written {
            deployment: Some(json!(false)),
            client: Some(CodingAgent::ClaudeCode),
            ..Written::default()
        },
        false
    )]
    fn every_source_of_drop_params_reaches_the_call(
        #[case] sources: Written,
        #[case] expected: bool,
    ) {
        assert_eq!(shaped(sources).drop_params, expected);
    }

    #[rstest]
    #[case::the_deployments_when_the_request_names_none(json!({}), &["deployment"])]
    #[case::the_requests_in_its_place(
        json!({"additional_drop_params": ["request", 1, "nested.path"]}),
        &["request", "nested.path"]
    )]
    #[case::the_deployments_when_the_request_sends_null(
        json!({"additional_drop_params": null}),
        &["deployment"]
    )]
    #[case::none_for_text(json!({"additional_drop_params": "request"}), &[])]
    fn a_requests_additional_drop_params_replace_the_deployments(
        #[case] body: Value,
        #[case] expected: &[&str],
    ) {
        let sources = Written {
            request: Some(body),
            ..Written::default()
        };

        assert_eq!(shaped(sources).additional_drop_params, expected);
    }

    const CACHED_USAGE: &str = r#""usage":{"input_tokens":7,"output_tokens":5,"cache_creation_input_tokens":100,"cache_read_input_tokens":20}"#;

    #[rstest]
    #[case::a_message(
        r#"{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":"end_turn","stop_sequence":null,USAGE}"#
    )]
    #[case::a_body_that_is_not_a_message(r#"{"type":"message",USAGE}"#)]
    fn a_complete_answer_counts_the_cache_it_wrote_and_read(#[case] body: &str) {
        let body: MessagesBody =
            serde_json::from_str(&body.replace("USAGE", CACHED_USAGE)).unwrap();
        assert_eq!(
            counts(&CallOutput::Complete(body)).map(Counts::total),
            Some(132)
        );
    }

    fn catalog() -> Catalog {
        Catalog::parse(br#"{"provider-model": {"litellm_provider": "provider"}}"#)
            .unwrap()
            .value
    }

    #[rstest]
    #[case::declared(json!({"supports_reasoning": true}), true)]
    #[case::declared_off(json!({"supports_reasoning": false}), false)]
    #[case::undeclared(json!({}), false)]
    fn model_info_flags_reach_the_deployment(#[case] model_info: Value, #[case] expected: bool) {
        let deployment = deployment(json!({"model": "anthropic/claude-unlisted-9"}), model_info);

        let capabilities = shaping(
            &deployment,
            &catalog(),
            &Configured::default(),
            &Map::new(),
            None,
        )
        .capabilities;

        assert_eq!(capabilities.supports_reasoning, expected);
    }

    #[test]
    fn configured_settings_reach_the_deployment() {
        let deployment = deployment(json!({"model": "provider/model"}), json!({}));
        let configured = Configured {
            reasoning_auto_summary: true,
            request_timeout: Some(30.0),
            ..Configured::default()
        };

        let settings = shaping(&deployment, &catalog(), &configured, &Map::new(), None).settings;

        assert_eq!(
            (settings.reasoning_auto_summary, settings.request_timeout),
            (true, Some(30.0))
        );
    }

    #[rstest]
    #[case::a_list(json!(["metadata.user_id", 1, "thinking"]), &["metadata.user_id", "thinking"])]
    #[case::text(json!("metadata.user_id"), &[])]
    #[case::a_map(json!({"metadata": "user_id"}), &[])]
    fn a_deployments_additional_drop_params_are_the_text_entries_of_its_list(
        #[case] written: Value,
        #[case] expected: &[&str],
    ) {
        let deployment = deployment(
            json!({"model": "provider/model", "additional_drop_params": written}),
            json!({}),
        );

        let settings = shaping(
            &deployment,
            &catalog(),
            &Configured::default(),
            &Map::new(),
            None,
        )
        .settings;

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
    #[case::a_start_counts_the_cache_it_wrote_and_read(
        r#"{"type":"message_start","message":{"usage":{"input_tokens":7,"cache_creation_input_tokens":100,"cache_read_input_tokens":20}}}"#,
        Kind::Lifecycle,
        Some(127)
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
