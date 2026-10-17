//! `stream: true` for the providers chat translates (Anthropic, Bedrock Converse): the
//! provider's stream goes through its config's decoder and chunk parser, then through
//! [`StreamWrapper`], Python's `CustomStreamWrapper`, which decides what the client sees,
//! and goes out as OpenAI `chat.completion.chunk` SSE frames closed by `data: [DONE]`.

use std::{
    collections::VecDeque,
    convert::Infallible,
    io,
    sync::{Arc, Mutex},
    time::Duration,
};

use bytes::Bytes;
use futures_util::{StreamExt, stream::BoxStream};
use litellm_core_utils::core_helpers::{finish_reason_for, unix_now};
use litellm_host::{
    call::{CallOutput, HostedMachine, hosted_call},
    interceptors::{
        Cost, ExecutionFacts, Interceptors, ProviderIdentity, RequestContext, ResultSource,
        WireRequest,
    },
    observation::ObservationSender,
    protocol::Protocol,
};
use litellm_http::{request::truncate_error_body, transport};
use litellm_inference::{CallOptions, context::CallContext};
use litellm_llms::base_llm::{
    auth::{Authenticated, resolve_auth},
    base_model_iterator::ByteStream,
    chat::streaming::{ChatChunkStream, ChatStream, encode},
    sse::DONE_FRAME,
};
use litellm_llms_types::formats::chat_completions::{
    ChatCompletionChunk, ChatCompletionDelta, ChatCompletionStreamingChoice, ChatCompletionsUsage,
};
use serde_json::{Map, Value};

use crate::{
    ChatCompletionsRoute, Error,
    constants::CHAT_COMPLETIONS_TIMEOUT_SECS,
    openai::OpenAiChatOutput,
    prepare::{prepare_provider_request, resolve_request},
    types::{ChatCompletionsCall, ChatCompletionsRequest},
};

const CHUNK_OBJECT: &str = "chat.completion.chunk";
const STREAM_PARAM: &str = "stream";
const STREAM_OPTIONS_PARAM: &str = "stream_options";

/// A translated stream: its chunks are the OpenAI SSE frames the client gets.
pub struct ChatCompletionsStream;

impl Protocol for ChatCompletionsStream {
    type Request = ChatStreamCall;
    /// Never produced: the call always streams.
    type Response = Bytes;
    type Error = Error;
    type HostCall = Infallible;
    type Chunk = Bytes;
    type StreamHead = ();
}

/// The usage a translated stream ends with, whether or not the client asked to see it, so
/// the host can account for the call.
#[derive(Clone, Debug, Default)]
pub struct ReportedUsage(Arc<Mutex<Option<ChatCompletionsUsage>>>);

impl ReportedUsage {
    pub fn get(&self) -> Option<ChatCompletionsUsage> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn set(&self, usage: ChatCompletionsUsage) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(usage);
    }
}

/// A streaming chat call to a provider chat translates.
pub struct ChatStreamCall {
    pub call: ChatCompletionsCall,
    /// The model every chunk names: the one the client asked for, as the proxy stamps it.
    pub response_model: String,
    pub usage: ReportedUsage,
}

impl ChatCompletionsRoute {
    /// Whether a chat body asks for a stream.
    pub fn is_stream(optional_params: &Map<String, Value>) -> bool {
        optional_params
            .get(STREAM_PARAM)
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// The call as a stream of OpenAI SSE frames.
    pub fn stream_machine(
        self,
        call: ChatStreamCall,
        observers: Option<ObservationSender>,
    ) -> HostedMachine<ChatCompletionsStream> {
        hosted_call(
            call,
            observers,
            move |call, _, interceptors, observers| async move {
                let context = CallContext::new(
                    &interceptors,
                    CallOptions {
                        cache: None,
                        observers,
                    },
                );
                Box::pin(self.run_stream(call, context)).await
            },
        )
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "chat_completions",
        model = %call.call.model,
        provider,
        resolved_model,
        stream = true,
        outcome
    ))]
    pub(crate) async fn run_stream(
        &self,
        call: ChatStreamCall,
        context: CallContext<'_, impl Interceptors<Error>>,
    ) -> Result<OpenAiChatOutput, Error> {
        let ChatStreamCall {
            call,
            response_model,
            usage,
        } = call;
        litellm_inference::diagnostic::call(async {
            let mut optional_params = call.optional_params;
            optional_params.remove(STREAM_PARAM);
            let include_usage = include_usage(optional_params.remove(STREAM_OPTIONS_PARAM))?;
            let timeout = call
                .timeout
                .unwrap_or(Duration::from_secs(CHAT_COMPLETIONS_TIMEOUT_SECS));
            let resolved = resolve_request(ChatCompletionsRequest {
                model: &call.model,
                messages: call.messages,
                optional_params,
                api_key: call.api_key.as_deref(),
                api_base: call.api_base.as_deref(),
                custom_llm_provider: call.custom_llm_provider.as_deref(),
                extra_headers: call.extra_headers,
                timeout: call.timeout,
            })?;
            let snapshot = self
                .secrets
                .resolve(&resolved.config.secret_names())
                .await?;
            let prepared = prepare_provider_request(resolved, snapshot)?;
            litellm_inference::diagnostic::provider(&prepared.model, &prepared.custom_llm_provider);
            let decoder = prepared
                .config
                .model_response_iterator(prepared.stream_shape.clone())
                .ok_or(Error::Unsupported("streaming"))?;
            let (url, body) = prepared
                .config
                .streaming_request(prepared.url, prepared.body)?;
            let request_context = RequestContext {
                model: prepared.model.clone(),
                custom_llm_provider: prepared.custom_llm_provider.clone(),
                optional_params: Value::Object(prepared.optional_params),
                secret_fields: Vec::new(),
                api_key: prepared.api_key,
            };
            let secrets = prepared.secrets;
            let authenticated =
                resolve_auth(&self.auth, prepared.environment, &|key| secrets.get(key)).await?;
            let wire = context
                .interceptors
                .before_provider_request(
                    WireRequest {
                        url,
                        headers: authenticated.headers,
                        body,
                    },
                    request_context,
                )
                .await?;
            let response = send(
                &self.http,
                Authenticated {
                    headers: wire.headers,
                    signer: authenticated.signer,
                },
                wire.url,
                &wire.body,
                timeout,
            )
            .await?;
            let wrapper = StreamWrapper::new(response_model, include_usage, usage);
            let chunks = openai_frames(response_bytes(response, timeout), decoder, wrapper);
            // An error before anything reaches the client still fails the attempt, so the
            // router can retry it elsewhere, as Python's pre-first-chunk fallback does.
            let chunks = first_frame(chunks).await?;
            context
                .result_ready(ExecutionFacts {
                    provider: ProviderIdentity {
                        model: prepared.model,
                        provider: prepared.custom_llm_provider,
                    },
                    source: ResultSource::Provider,
                    cost: Cost::Deferred,
                })
                .await?;
            Ok(CallOutput::Stream { head: (), chunks })
        })
        .await
    }
}

/// `stream_options.include_usage`, which only decides whether the client sees the usage.
fn include_usage(options: Option<Value>) -> Result<bool, Error> {
    match options {
        None | Some(Value::Null) => Ok(false),
        Some(Value::Object(options)) => {
            Ok(options.get("include_usage") == Some(&Value::Bool(true)))
        }
        Some(_) => Err(Error::InvalidRequest(
            "stream_options must be an object".into(),
        )),
    }
}

/// Waits for the stream's first frame: an error then fails the call itself.
async fn first_frame(
    mut chunks: BoxStream<'static, Result<Bytes, Error>>,
) -> Result<BoxStream<'static, Result<Bytes, Error>>, Error> {
    match chunks.next().await {
        None => Ok(futures_util::stream::empty().boxed()),
        Some(Err(error)) => Err(error),
        Some(Ok(first)) => Ok(futures_util::stream::once(async move { Ok(first) })
            .chain(chunks)
            .boxed()),
    }
}

/// The response head is awaited for at most `timeout`; a stream has no total limit, so a long
/// generation is not cut off, and `timeout` bounds each read after the head instead.
async fn send(
    http: &litellm_http::Client,
    authenticated: Authenticated,
    url: String,
    body: &Value,
    timeout: Duration,
) -> Result<reqwest::Response, Error> {
    let request = litellm_inference::outbound::outbound_request(authenticated, url, body, None)
        .map_err(|error| match error {
            litellm_http::Error::ComputedHeader(_) => {
                Error::Unsupported("request forwards a header AWS SigV4 computes")
            }
            other => Error::Http(other),
        })?;
    let response = tokio::time::timeout(timeout, litellm_inference::outbound::send(request, http))
        .await
        .map_err(|_| silent_provider(timeout))?
        .map_err(|error| Error::Transport(transport::Error::from_reqwest_before_dispatch(error)))?;
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let text = response
        .text()
        .await
        .map_err(|error| Error::Transport(transport::Error::from(error)))?;
    Err(Error::Transport(transport::Error::Http {
        status: status.as_u16(),
        body: truncate_error_body(&text),
    }))
}

fn silent_provider(timeout: Duration) -> Error {
    Error::Transport(transport::Error::Network(format!(
        "the provider sent nothing for {}s",
        timeout.as_secs_f64()
    )))
}

/// The provider's body, with each read bounded by `idle`. A read that fails is kept in full
/// beside the stream, since the decoder only sees an I/O error.
struct ResponseBytes {
    bytes: ByteStream,
    failure: Arc<Mutex<Option<Error>>>,
}

fn response_bytes(response: reqwest::Response, idle: Duration) -> ResponseBytes {
    let failure = Arc::new(Mutex::new(None));
    let recorded = failure.clone();
    let bytes = futures_util::stream::try_unfold(response, move |mut response| {
        let recorded = recorded.clone();
        async move {
            let read = tokio::time::timeout(idle, response.chunk())
                .await
                .map_err(|_| silent_provider(idle))
                .and_then(|read| {
                    read.map_err(|error| Error::Transport(transport::Error::from(error)))
                });
            match read {
                Ok(chunk) => Ok(chunk.map(|chunk| (chunk, response))),
                Err(error) => {
                    let message = error.to_string();
                    *recorded.lock().unwrap_or_else(|e| e.into_inner()) = Some(error);
                    Err(io::Error::other(message))
                }
            }
        }
    })
    .boxed();
    ResponseBytes { bytes, failure }
}

struct Frames {
    chunks: ChatChunkStream,
    failure: Arc<Mutex<Option<Error>>>,
    wrapper: StreamWrapper,
    ready: VecDeque<Bytes>,
    ended: bool,
}

/// Decodes the provider's bytes into chunks and sends what the wrapper keeps as SSE frames,
/// then `data: [DONE]`. The first error ends the stream.
fn openai_frames(
    bytes: ResponseBytes,
    decoder: ChatStream,
    wrapper: StreamWrapper,
) -> BoxStream<'static, Result<Bytes, Error>> {
    let ResponseBytes { bytes, failure } = bytes;
    let frames = Frames {
        chunks: decoder.run(bytes),
        failure,
        wrapper,
        ready: VecDeque::new(),
        ended: false,
    };
    futures_util::stream::unfold(frames, |mut frames| async move {
        loop {
            if let Some(frame) = frames.ready.pop_front() {
                return Some((Ok(frame), frames));
            }
            if frames.ended {
                return None;
            }
            match frames.chunks.next().await {
                Some(Ok(chunk)) => {
                    if let Some(chunk) = frames.wrapper.process(chunk) {
                        match encode(&chunk) {
                            Ok(frame) => frames.ready.push_back(frame),
                            Err(error) => {
                                frames.ended = true;
                                return Some((Err(error.into()), frames));
                            }
                        }
                    }
                }
                Some(Err(error)) => {
                    frames.ended = true;
                    let transport = frames
                        .failure
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .take();
                    return Some((Err(transport.unwrap_or_else(|| error.into())), frames));
                }
                None => {
                    frames.ended = true;
                    for chunk in frames.wrapper.finish() {
                        match encode(&chunk) {
                            Ok(frame) => frames.ready.push_back(frame),
                            Err(error) => return Some((Err(error.into()), frames)),
                        }
                    }
                    frames.ready.push_back(Bytes::from_static(DONE_FRAME));
                }
            }
        }
    })
    .boxed()
}

/// Python's `CustomStreamWrapper` over provider chunks: the client gets content chunks with
/// the role on the first, one empty chunk with the finish reason, and, when it asked for it,
/// a last chunk with the usage. Usage is never sent on the other chunks.
pub(crate) struct StreamWrapper {
    model: String,
    id: Option<String>,
    created: u64,
    include_usage: bool,
    reported: ReportedUsage,
    sent_first_chunk: bool,
    sent_last_chunk: bool,
    received_finish_reason: Option<String>,
    tool_call: bool,
    usage: Option<ChatCompletionsUsage>,
}

impl StreamWrapper {
    pub(crate) fn new(model: String, include_usage: bool, reported: ReportedUsage) -> Self {
        Self {
            model,
            id: None,
            created: unix_now(),
            include_usage,
            reported,
            sent_first_chunk: false,
            sent_last_chunk: false,
            received_finish_reason: None,
            tool_call: false,
            usage: None,
        }
    }

    fn id(&mut self) -> String {
        self.id
            .get_or_insert_with(|| format!("chatcmpl-{}", uuid::Uuid::new_v4()))
            .clone()
    }

    fn chunk(
        &mut self,
        delta: ChatCompletionDelta,
        finish_reason: Option<String>,
        provider_specific_fields: Option<Map<String, Value>>,
    ) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: Some(self.id()),
            created: Some(self.created),
            model: Some(self.model.clone()),
            object: Some(CHUNK_OBJECT.into()),
            choices: vec![ChatCompletionStreamingChoice {
                index: 0,
                delta,
                finish_reason,
                logprobs: None,
            }],
            usage: None,
            provider_specific_fields,
            extra: Map::new(),
        }
    }

    /// The chunk the client gets for one provider chunk, if any.
    pub(crate) fn process(&mut self, chunk: ChatCompletionChunk) -> Option<ChatCompletionChunk> {
        let ChatCompletionChunk {
            id,
            usage,
            choices,
            provider_specific_fields,
            ..
        } = chunk;
        if let Some(id) = id.filter(|id| !id.trim().is_empty()) {
            self.id = Some(id);
        }
        if let Some(usage) = usage {
            self.usage = Some(usage);
        }
        let ChatCompletionStreamingChoice {
            mut delta,
            finish_reason,
            ..
        } = choices.into_iter().next()?;
        if let Some(reason) = finish_reason.filter(|reason| !reason.is_empty()) {
            self.received_finish_reason = Some(reason);
        }
        let has_tool_calls = delta
            .tool_calls
            .as_ref()
            .is_some_and(|tools| !tools.is_empty());
        if has_tool_calls {
            self.tool_call = true;
        }
        let non_empty = delta.content.as_ref().is_some_and(|text| !text.is_empty())
            || has_tool_calls
            || delta.reasoning_content.is_some()
            || delta.provider_specific_fields.is_some()
            || (!self.sent_first_chunk && delta.role.is_some());
        if non_empty {
            delta.role = (!self.sent_first_chunk).then(|| "assistant".to_owned());
            self.sent_first_chunk = true;
            return Some(self.chunk(delta, None, provider_specific_fields));
        }
        let received = self.received_finish_reason.clone()?;
        if self.sent_last_chunk || delta.tool_calls.is_some() {
            return None;
        }
        self.sent_last_chunk = true;
        Some(self.chunk(
            ChatCompletionDelta::default(),
            Some(finish_reason_for(&received).to_owned()),
            provider_specific_fields,
        ))
    }

    /// The chunks after the provider's last: the finish reason if none went out yet, then the
    /// usage when the client asked for it.
    pub(crate) fn finish(&mut self) -> Vec<ChatCompletionChunk> {
        let mut chunks = Vec::new();
        if !self.sent_last_chunk {
            self.sent_last_chunk = true;
            let reason = match self.received_finish_reason.as_deref().unwrap_or("stop") {
                "stop" if self.tool_call => "tool_calls".to_owned(),
                reason => reason.to_owned(),
            };
            chunks.push(self.chunk(ChatCompletionDelta::default(), Some(reason), None));
        }
        if let Some(usage) = self.usage.clone() {
            self.reported.set(usage.clone());
            if self.include_usage {
                let mut chunk = self.chunk(ChatCompletionDelta::default(), None, None);
                chunk.usage = Some(usage);
                chunks.push(chunk);
            }
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use futures_util::{TryStreamExt, stream};
    use litellm_llms::{
        anthropic::chat::transformation::ANTHROPIC_CHAT_COMPLETIONS_CONFIG,
        base_llm::chat::{streaming::StreamShape, transformation::BaseConfig},
        bedrock::chat::converse_transformation::BEDROCK_CHAT_COMPLETIONS_CONFIG,
    };
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn in_pieces(wire: Vec<u8>, size: usize) -> ResponseBytes {
        let pieces: Vec<Bytes> = wire.chunks(size).map(Bytes::copy_from_slice).collect();
        ResponseBytes {
            bytes: stream::iter(pieces.into_iter().map(Ok)).boxed(),
            failure: Arc::default(),
        }
    }

    fn anthropic_wire(events: Value) -> Vec<u8> {
        let Value::Array(events) = events else {
            panic!("events are a list");
        };
        events
            .iter()
            .map(|event| {
                format!(
                    "event: {}\ndata: {event}\n\n",
                    event["type"].as_str().unwrap()
                )
            })
            .collect::<String>()
            .into_bytes()
    }

    fn bedrock_wire(events: Value) -> Vec<u8> {
        use litellm_framer::aws_event_stream::{AwsEventStreamCodec, Header, HeaderValue, Message};
        use tokio_util::codec::Encoder;
        let Value::Array(events) = events else {
            panic!("events are a list");
        };
        let mut wire = bytes::BytesMut::new();
        for event in events {
            let (kind, name, payload) = (
                event[0].as_str().unwrap(),
                event[1].as_str().unwrap(),
                &event[2],
            );
            let type_header = if kind == "exception" {
                ":exception-type"
            } else {
                ":event-type"
            };
            let message = Message::new(Bytes::from(payload.to_string()))
                .add_header(Header::new(
                    ":message-type",
                    HeaderValue::String(kind.to_owned().into()),
                ))
                .add_header(Header::new(
                    type_header,
                    HeaderValue::String(name.to_owned().into()),
                ));
            AwsEventStreamCodec.encode(message, &mut wire).unwrap();
        }
        wire.to_vec()
    }

    struct Translated {
        frames: Vec<Result<Value, String>>,
        done: bool,
        reported: Option<ChatCompletionsUsage>,
    }

    async fn translate(
        config: &dyn BaseConfig,
        wire: Vec<u8>,
        include_usage: bool,
        size: usize,
    ) -> Translated {
        let reported = ReportedUsage::default();
        let wrapper = StreamWrapper::new("public/model".into(), include_usage, reported.clone());
        let decoder = config
            .model_response_iterator(StreamShape::default())
            .unwrap();
        let items: Vec<Result<Bytes, Error>> =
            openai_frames(in_pieces(wire, size), decoder, wrapper)
                .collect()
                .await;
        let mut done = false;
        let frames = items
            .into_iter()
            .filter_map(|item| match item {
                Ok(frame) if frame.as_ref() == DONE_FRAME => {
                    done = true;
                    None
                }
                Ok(frame) => {
                    let text = std::str::from_utf8(&frame).unwrap();
                    let data = text
                        .strip_prefix("data: ")
                        .unwrap()
                        .strip_suffix("\n\n")
                        .unwrap();
                    Some(Ok(serde_json::from_str(data).unwrap()))
                }
                Err(error) => Some(Err(error.to_string())),
            })
            .collect();
        Translated {
            frames,
            done,
            reported: reported.get(),
        }
    }

    /// The parts of a chunk Python's proxy sends that do not vary per call.
    fn shape(chunk: &Value) -> Value {
        let mut chunk = chunk.clone();
        let object = chunk.as_object_mut().unwrap();
        assert!(
            object
                .remove("id")
                .unwrap()
                .as_str()
                .unwrap()
                .starts_with("chatcmpl-")
        );
        assert!(object.remove("created").unwrap().is_u64());
        assert_eq!(
            object.remove("object"),
            Some(json!("chat.completion.chunk"))
        );
        assert_eq!(object.remove("model"), Some(json!("public/model")));
        chunk
    }

    fn shapes(translated: &Translated) -> Vec<Value> {
        translated
            .frames
            .iter()
            .map(|frame| shape(frame.as_ref().unwrap()))
            .collect()
    }

    fn anthropic_text() -> Value {
        json!([
            {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x-20250101", "content": [], "stop_reason": null, "stop_sequence": null,
             "usage": {"input_tokens": 10, "cache_creation_input_tokens": 6, "cache_read_input_tokens": 20, "output_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "ping"},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 12}},
            {"type": "message_stop"}
        ])
    }

    fn usage_36_12() -> Value {
        json!({"completion_tokens": 12, "prompt_tokens": 36, "total_tokens": 48,
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 12},
            "prompt_tokens_details": {"cached_tokens": 20, "text_tokens": 10, "cache_write_tokens": 6, "cache_creation_tokens": 6},
            "cache_creation_input_tokens": 6, "cache_read_input_tokens": 20})
    }

    // Expected chunks are the ones Python's proxy sends for the same provider stream.
    #[rstest]
    #[tokio::test]
    async fn anthropic_text_matches_python(
        #[values(true, false)] include_usage: bool,
        #[values(1, 7, 4096)] size: usize,
    ) {
        let translated = translate(
            &ANTHROPIC_CHAT_COMPLETIONS_CONFIG,
            anthropic_wire(anthropic_text()),
            include_usage,
            size,
        )
        .await;
        let mut expected = vec![
            json!({"choices": [{"index": 0, "delta": {"content": "Hello", "role": "assistant"}}]}),
            json!({"choices": [{"index": 0, "delta": {"content": " world"}}]}),
            json!({"choices": [{"finish_reason": "stop", "index": 0, "delta": {}}]}),
        ];
        if include_usage {
            expected.push(json!({"choices": [{"index": 0, "delta": {}}], "usage": usage_36_12()}));
        }
        assert_eq!(shapes(&translated), expected);
        assert!(translated.done);
        // The host gets the usage either way.
        assert_eq!(
            serde_json::to_value(translated.reported.unwrap()).unwrap(),
            usage_36_12()
        );
        let ids: Vec<_> = translated
            .frames
            .iter()
            .map(|f| f.as_ref().unwrap()["id"].clone())
            .collect();
        assert!(
            ids.windows(2).all(|pair| pair[0] == pair[1]),
            "one id per stream"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn anthropic_tool_calls_match_python() {
        let wire = anthropic_wire(json!([
            {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 10, "output_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me check"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\": \"Pa"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "ris\"}"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_2", "name": "now", "input": {}}},
            {"type": "content_block_stop", "index": 2},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null}, "usage": {"output_tokens": 30}},
            {"type": "message_stop"}
        ]));
        let translated = translate(&ANTHROPIC_CHAT_COMPLETIONS_CONFIG, wire, true, 5).await;
        let tool = |call: Value| json!({"choices": [{"index": 0, "delta": {"content": "", "tool_calls": [call]}}]});
        assert_eq!(
            shapes(&translated),
            vec![
                json!({"choices": [{"index": 0, "delta": {"content": "Let me check", "role": "assistant"}}]}),
                tool(
                    json!({"id": "toolu_1", "function": {"arguments": "", "name": "get_weather"}, "type": "function", "index": 0})
                ),
                tool(json!({"function": {"arguments": ""}, "type": "function", "index": 0})),
                tool(
                    json!({"function": {"arguments": "{\"city\": \"Pa"}, "type": "function", "index": 0})
                ),
                tool(json!({"function": {"arguments": "ris\"}"}, "type": "function", "index": 0})),
                tool(
                    json!({"id": "toolu_2", "function": {"arguments": "", "name": "now"}, "type": "function", "index": 1})
                ),
                json!({"choices": [{"finish_reason": "tool_calls", "index": 0, "delta": {}}]}),
                json!({"choices": [{"index": 0, "delta": {}}], "usage": {"completion_tokens": 30, "prompt_tokens": 10, "total_tokens": 40,
                    "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 30},
                    "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 10, "cache_write_tokens": 0, "cache_creation_tokens": 0},
                    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}),
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn anthropic_thinking_matches_python() {
        let wire = anthropic_wire(json!([
            {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 10, "output_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me think"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig123"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "redacted_thinking", "data": "enc"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "Answer"}},
            {"type": "content_block_stop", "index": 2},
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens", "stop_sequence": null}, "usage": {"output_tokens": 50}},
            {"type": "message_stop"}
        ]));
        let translated = translate(&ANTHROPIC_CHAT_COMPLETIONS_CONFIG, wire, false, 4096).await;
        let thinking = json!([{"type": "thinking", "thinking": "Let me think", "index": 0}]);
        let signature =
            json!([{"type": "thinking", "thinking": "", "signature": "sig123", "index": 0}]);
        let redacted = json!([{"type": "redacted_thinking", "data": "enc", "index": 1}]);
        assert_eq!(
            shapes(&translated),
            vec![
                json!({"choices": [{"index": 0, "delta": {"reasoning_content": "Let me think", "thinking_blocks": thinking, "provider_specific_fields": {"thinking_blocks": thinking}, "content": "", "role": "assistant"}}]}),
                json!({"choices": [{"index": 0, "delta": {"reasoning_content": "", "thinking_blocks": signature, "provider_specific_fields": {"thinking_blocks": signature}, "content": ""}}]}),
                json!({"choices": [{"index": 0, "delta": {"thinking_blocks": redacted, "provider_specific_fields": {"thinking_blocks": redacted}, "content": ""}}]}),
                json!({"choices": [{"index": 0, "delta": {"content": "Answer"}}]}),
                json!({"choices": [{"finish_reason": "length", "index": 0, "delta": {}}]}),
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_anthropic_error_event_ends_the_stream_after_what_was_sent() {
        let wire = anthropic_wire(json!([
            {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x", "content": [], "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 10, "output_tokens": 1}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        ]));
        let translated = translate(&ANTHROPIC_CHAT_COMPLETIONS_CONFIG, wire, true, 4096).await;
        assert_eq!(translated.frames.len(), 2);
        assert_eq!(
            translated.frames[0].as_ref().unwrap()["choices"][0]["delta"]["content"],
            "Hi"
        );
        assert!(
            translated.frames[1]
                .as_ref()
                .unwrap_err()
                .contains("Overloaded")
        );
        assert!(!translated.done);
        assert_eq!(translated.reported, None);
    }

    #[rstest]
    #[tokio::test]
    async fn bedrock_text_matches_python(#[values(true, false)] include_usage: bool) {
        let wire = bedrock_wire(json!([
            ["event", "messageStart", {"role": "assistant"}],
            ["event", "contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hello"}}],
            ["event", "contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": " world"}}],
            ["event", "contentBlockStop", {"contentBlockIndex": 0}],
            ["event", "messageStop", {"stopReason": "end_turn"}],
            ["event", "metadata", {"usage": {"inputTokens": 10, "outputTokens": 12, "totalTokens": 48, "cacheReadInputTokens": 20, "cacheWriteInputTokens": 6}, "metrics": {"latencyMs": 100}}]
        ]));
        let translated = translate(&BEDROCK_CHAT_COMPLETIONS_CONFIG, wire, include_usage, 3).await;
        let mut expected = vec![
            json!({"choices": [{"index": 0, "delta": {"content": "Hello", "role": "assistant"}}], "provider_specific_fields": {}}),
            json!({"choices": [{"index": 0, "delta": {"content": " world"}}], "provider_specific_fields": {}}),
            json!({"choices": [{"finish_reason": "stop", "index": 0, "delta": {}}], "provider_specific_fields": {}}),
        ];
        if include_usage {
            expected.push(json!({"choices": [{"index": 0, "delta": {}}], "usage": usage_36_12()}));
        }
        assert_eq!(shapes(&translated), expected);
        assert!(translated.done);
        assert_eq!(
            serde_json::to_value(translated.reported.unwrap()).unwrap(),
            usage_36_12()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn bedrock_tool_calls_match_python() {
        let wire = bedrock_wire(json!([
            ["event", "messageStart", {"role": "assistant"}],
            ["event", "contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Let me check"}}],
            ["event", "contentBlockStop", {"contentBlockIndex": 0}],
            ["event", "contentBlockStart", {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "tooluse_1", "name": "get_weather"}}}],
            ["event", "contentBlockDelta", {"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{\"city\": \"Pa"}}}],
            ["event", "contentBlockDelta", {"contentBlockIndex": 1, "delta": {"toolUse": {"input": "ris\"}"}}}],
            ["event", "contentBlockStop", {"contentBlockIndex": 1}],
            ["event", "contentBlockStart", {"contentBlockIndex": 2, "start": {"toolUse": {"toolUseId": "tooluse_2", "name": "now"}}}],
            ["event", "contentBlockDelta", {"contentBlockIndex": 2, "delta": {"toolUse": {"input": ""}}}],
            ["event", "contentBlockStop", {"contentBlockIndex": 2}],
            ["event", "messageStop", {"stopReason": "tool_use"}],
            ["event", "metadata", {"usage": {"inputTokens": 10, "outputTokens": 30, "totalTokens": 40}, "metrics": {"latencyMs": 100}}]
        ]));
        let translated = translate(&BEDROCK_CHAT_COMPLETIONS_CONFIG, wire, false, 4096).await;
        let tool = |call: Value| json!({"choices": [{"index": 0, "delta": {"content": "", "tool_calls": [call]}}], "provider_specific_fields": {}});
        assert_eq!(
            shapes(&translated),
            vec![
                json!({"choices": [{"index": 0, "delta": {"content": "Let me check", "role": "assistant"}}], "provider_specific_fields": {}}),
                tool(
                    json!({"id": "tooluse_1", "function": {"arguments": "", "name": "get_weather"}, "type": "function", "index": 0})
                ),
                tool(
                    json!({"function": {"arguments": "{\"city\": \"Pa"}, "type": "function", "index": 0})
                ),
                tool(json!({"function": {"arguments": "ris\"}"}, "type": "function", "index": 0})),
                tool(
                    json!({"id": "tooluse_2", "function": {"arguments": "", "name": "now"}, "type": "function", "index": 1})
                ),
                tool(json!({"function": {"arguments": ""}, "type": "function", "index": 1})),
                tool(json!({"function": {"arguments": "{}"}, "type": "function", "index": 1})),
                json!({"choices": [{"finish_reason": "tool_calls", "index": 0, "delta": {}}], "provider_specific_fields": {}}),
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn bedrock_reasoning_matches_python() {
        let wire = bedrock_wire(json!([
            ["event", "messageStart", {"role": "assistant"}],
            ["event", "contentBlockDelta", {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "Let me think"}}}],
            ["event", "contentBlockDelta", {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig123"}}}],
            ["event", "contentBlockStop", {"contentBlockIndex": 0}],
            ["event", "contentBlockDelta", {"contentBlockIndex": 1, "delta": {"text": "Answer"}}],
            ["event", "contentBlockStop", {"contentBlockIndex": 1}],
            ["event", "messageStop", {"stopReason": "max_tokens"}]
        ]));
        let translated = translate(&BEDROCK_CHAT_COMPLETIONS_CONFIG, wire, false, 4096).await;
        assert_eq!(
            shapes(&translated),
            vec![
                json!({"choices": [{"index": 0, "delta": {"reasoning_content": "Let me think", "thinking_blocks": [{"type": "thinking", "index": 0, "thinking": "Let me think"}], "provider_specific_fields": {"reasoningContent": {"text": "Let me think"}}, "content": "", "role": "assistant"}}], "provider_specific_fields": {}}),
                json!({"choices": [{"index": 0, "delta": {"reasoning_content": "", "thinking_blocks": [{"type": "thinking", "index": 0, "signature": "sig123", "thinking": ""}], "provider_specific_fields": {"reasoningContent": {"signature": "sig123"}}, "content": ""}}], "provider_specific_fields": {}}),
                json!({"choices": [{"index": 0, "delta": {"content": "Answer"}}], "provider_specific_fields": {}}),
                json!({"choices": [{"finish_reason": "length", "index": 0, "delta": {}}], "provider_specific_fields": {}}),
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_bedrock_exception_ends_the_stream_with_its_status() {
        let wire = bedrock_wire(json!([
            ["event", "messageStart", {"role": "assistant"}],
            ["event", "contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hi"}}],
            ["exception", "throttlingException", {"message": "Too many requests"}]
        ]));
        let reported = ReportedUsage::default();
        let decoder = BEDROCK_CHAT_COMPLETIONS_CONFIG
            .model_response_iterator(StreamShape::default())
            .unwrap();
        let mut frames = openai_frames(
            in_pieces(wire, 4096),
            decoder,
            StreamWrapper::new("m".into(), true, reported),
        );
        assert!(frames.try_next().await.unwrap().is_some());
        let error = frames.try_next().await.unwrap_err();
        assert!(
            matches!(&error, Error::Transport(transport::Error::Http { status: 429, body }) if body.contains("Too many requests")),
            "{error:?}"
        );
        assert!(frames.next().await.is_none());
    }

    #[rstest]
    #[tokio::test]
    async fn an_error_before_the_first_frame_fails_the_call() {
        let wire = bedrock_wire(json!([
            ["event", "messageStart", {"role": "assistant"}],
            ["exception", "serviceUnavailableException", {"message": "busy"}]
        ]));
        let decoder = BEDROCK_CHAT_COMPLETIONS_CONFIG
            .model_response_iterator(StreamShape::default())
            .unwrap();
        let frames = openai_frames(
            in_pieces(wire, 4096),
            decoder,
            StreamWrapper::new("m".into(), false, ReportedUsage::default()),
        );
        assert!(matches!(
            first_frame(frames).await.err(),
            Some(Error::Transport(transport::Error::Http { status: 503, .. }))
        ));
    }

    #[rstest]
    fn a_stream_that_ends_without_a_finish_reason_stops_or_calls_tools() {
        let mut wrapper = StreamWrapper::new("m".into(), false, ReportedUsage::default());
        let tool: ChatCompletionChunk = serde_json::from_value(json!({"choices": [{"index": 0, "delta": {"content": "", "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": ""}, "index": 0}]}}]})).unwrap();
        assert!(wrapper.process(tool).is_some());
        let finish = wrapper.finish();
        assert_eq!(finish.len(), 1);
        assert_eq!(
            finish[0].choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[rstest]
    #[case::absent(None, Ok(false))]
    #[case::null(Some(Value::Null), Ok(false))]
    #[case::on(Some(json!({"include_usage": true})), Ok(true))]
    #[case::off(Some(json!({"include_usage": false})), Ok(false))]
    #[case::not_an_object(Some(json!(true)), Err(()))]
    fn include_usage_reads_stream_options(
        #[case] options: Option<Value>,
        #[case] expected: Result<bool, ()>,
    ) {
        assert_eq!(include_usage(options).map_err(|_| ()), expected);
    }

    mod calls {
        use std::time::Duration;

        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{header, method, path},
        };

        use super::*;
        use crate::test_support::{Recording, route};

        fn call(model: &str, api_base: &str, params: Value) -> ChatStreamCall {
            ChatStreamCall {
                call: ChatCompletionsCall {
                    model: model.into(),
                    messages: json!([{"role": "user", "content": "hi"}]),
                    optional_params: params.as_object().unwrap().clone(),
                    api_key: Some("sk-test".into()),
                    api_base: Some(api_base.into()),
                    custom_llm_provider: None,
                    extra_headers: None,
                    timeout: Some(Duration::from_secs(5)),
                },
                response_model: "public/model".into(),
                usage: ReportedUsage::default(),
            }
        }

        async fn run(
            route: &ChatCompletionsRoute,
            call: ChatStreamCall,
            interceptors: &Recording,
        ) -> Result<OpenAiChatOutput, Error> {
            route
                .run_stream(call, CallContext::new(interceptors, CallOptions::default()))
                .await
        }

        async fn body(output: OpenAiChatOutput) -> String {
            let CallOutput::Stream { chunks, .. } = output else {
                panic!("a translated call streams");
            };
            let frames: Vec<Bytes> = chunks.try_collect().await.unwrap();
            String::from_utf8(frames.concat()).unwrap()
        }

        #[rstest]
        #[tokio::test]
        async fn anthropic_gets_stream_true_and_never_the_stream_options(
            route: ChatCompletionsRoute,
        ) {
            let upstream = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/messages"))
                .and(header("x-api-key", "sk-test"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_raw(anthropic_wire(anthropic_text()), "text/event-stream"),
                )
                .expect(1)
                .mount(&upstream)
                .await;
            let interceptors = Recording::default();
            let streaming = call(
                "anthropic/claude-x",
                &upstream.uri(),
                json!({"max_tokens": 16, "stream": true, "stream_options": {"include_usage": true}}),
            );
            let reported = streaming.usage.clone();

            let body = body(run(&route, streaming, &interceptors).await.unwrap()).await;

            let sent: Value =
                serde_json::from_slice(&upstream.received_requests().await.unwrap()[0].body)
                    .unwrap();
            assert_eq!(
                sent,
                json!({"model": "claude-x", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "max_tokens": 16, "stream": true})
            );
            assert!(body.ends_with("data: [DONE]\n\n"));
            assert_eq!(body.matches("data: ").count(), 5, "{body}");
            assert!(body.contains("\"usage\":{\"prompt_tokens\":36"), "{body}");
            assert_eq!(reported.get().unwrap().completion_tokens, 12);
            assert_eq!(
                interceptors.contexts.lock().unwrap()[0].optional_params,
                json!({"max_tokens": 16})
            );
            assert_eq!(interceptors.results.lock().unwrap().len(), 1);
        }

        #[rstest]
        #[tokio::test]
        async fn bedrock_streams_from_converse_stream(route: ChatCompletionsRoute) {
            let upstream = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/model/anthropic.claude-x-v1:0/converse-stream"))
                .and(header("authorization", "Bearer sk-test"))
                .respond_with(ResponseTemplate::new(200).set_body_raw(
                    bedrock_wire(json!([
                        ["event", "messageStart", {"role": "assistant"}],
                        ["event", "contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hi"}}],
                        ["event", "messageStop", {"stopReason": "end_turn"}],
                        ["event", "metadata", {"usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4}, "metrics": {}}]
                    ])),
                    "application/vnd.amazon.eventstream",
                ))
                .expect(1)
                .mount(&upstream)
                .await;
            let streaming = call(
                "bedrock/anthropic.claude-x-v1:0",
                &upstream.uri(),
                json!({"maxTokens": 16, "stream": true}),
            );
            let reported = streaming.usage.clone();

            let body = body(run(&route, streaming, &Recording::default()).await.unwrap()).await;

            let sent: Value =
                serde_json::from_slice(&upstream.received_requests().await.unwrap()[0].body)
                    .unwrap();
            assert_eq!(
                sent,
                json!({"inferenceConfig": {"maxTokens": 16}, "messages": [{"role": "user", "content": [{"text": "hi"}]}]})
            );
            assert_eq!(body.matches("data: ").count(), 3, "{body}");
            assert!(!body.contains("usage"), "{body}");
            assert_eq!(reported.get().unwrap().total_tokens, 4);
        }

        #[rstest]
        #[tokio::test]
        async fn an_upstream_error_fails_before_the_stream_opens(route: ChatCompletionsRoute) {
            let upstream = MockServer::start().await;
            let error =
                r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(429).set_body_raw(error, "application/json"))
                .mount(&upstream)
                .await;
            let result = run(
                &route,
                call(
                    "anthropic/claude-x",
                    &upstream.uri(),
                    json!({"stream": true}),
                ),
                &Recording::default(),
            )
            .await;
            assert_eq!(
                result.err(),
                Some(Error::Transport(transport::Error::Http {
                    status: 429,
                    body: error.into()
                }))
            );
        }

        #[rstest]
        #[tokio::test]
        async fn a_provider_without_a_stream_parser_is_refused(route: ChatCompletionsRoute) {
            let result = run(
                &route,
                call(
                    "openai_like/gpt",
                    "http://127.0.0.1:1",
                    json!({"stream": true}),
                ),
                &Recording::default(),
            )
            .await;
            assert_eq!(result.err(), Some(Error::Unsupported("streaming")));
        }

        /// Streams the head and the first text, then reports when the client hangs up.
        async fn hanging_upstream(closed: tokio::sync::oneshot::Sender<()>) -> String {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = vec![0; 64 * 1024];
                let _ = socket.read(&mut request).await;
                socket
                    .write_all(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
                    .await
                    .unwrap();
                let events = String::from_utf8(anthropic_wire(json!([
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}
                ])))
                .unwrap();
                socket
                    .write_all(format!("{:x}\r\n{events}\r\n", events.len()).as_bytes())
                    .await
                    .unwrap();
                let mut rest = [0; 1024];
                while matches!(socket.read(&mut rest).await, Ok(read) if read > 0) {}
                let _ = closed.send(());
            });
            base
        }

        #[rstest]
        #[tokio::test]
        async fn dropping_the_stream_drops_the_upstream_request(route: ChatCompletionsRoute) {
            let (closed, hung_up) = tokio::sync::oneshot::channel();
            let base = hanging_upstream(closed).await;
            let output = run(
                &route,
                call("anthropic/claude-x", &base, json!({"stream": true})),
                &Recording::default(),
            )
            .await
            .unwrap();
            let CallOutput::Stream { mut chunks, .. } = output else {
                panic!("a translated call streams");
            };
            let first = chunks.try_next().await.unwrap().unwrap();
            assert!(std::str::from_utf8(&first).unwrap().contains("\"Hi\""));

            drop(chunks);

            tokio::time::timeout(Duration::from_secs(2), hung_up)
                .await
                .expect("the upstream connection closes once the client stream is dropped")
                .unwrap();
        }

        #[rstest]
        #[tokio::test]
        async fn a_stream_that_goes_silent_past_the_timeout_fails(route: ChatCompletionsRoute) {
            let (closed, _hung_up) = tokio::sync::oneshot::channel();
            let base = hanging_upstream(closed).await;
            let mut streaming = call("anthropic/claude-x", &base, json!({"stream": true}));
            streaming.call.timeout = Some(Duration::from_millis(200));
            let output = run(&route, streaming, &Recording::default()).await.unwrap();
            let CallOutput::Stream { mut chunks, .. } = output else {
                panic!("a translated call streams");
            };
            chunks.try_next().await.unwrap();
            assert!(matches!(
                chunks.try_next().await,
                Err(Error::Transport(transport::Error::Network(message))) if message.contains("sent nothing")
            ));
        }
    }
}
