//! A Messages stream as Python logs it: the Anthropic passthrough logging handler reads each
//! event as a chat chunk (`ModelResponseIterator.chunk_parser` in `llms/anthropic/chat/handler.py`)
//! and `stream_chunk_builder` joins the chunks into one `ModelResponse`
//! (`build_complete_streaming_response` in `llms/anthropic/messages/stream_assembly.py`). The
//! handler then names the response by the `message_start` id and the model it priced
//! (`_create_anthropic_response_logging_payload`).
//!
//! [`StreamedMessage`] keeps, event by event, what those chunks add up to, so the stream's
//! events need not be held.

use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{
    ASSISTANT, CHAT_COMPLETION, Error, FUNCTION, LoggedChoice, LoggedMessage, LoggedResponse,
    LoggedUsage, Null, ResponseContext, STOP_DETAILS, STOP_REASON, StreamedChatMessage,
    StreamedToolCall, ToolFunction, finish_reason, refusal, stop_reason_fields,
    usage::{Reasoning, ReportedUsage, StreamedUsage, calculate},
};

/// The events of a Messages stream, with the fields the chunk parser reads.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Event {
    MessageStart {
        message: StartMessage,
    },
    ContentBlockStart {
        content_block: Map<String, Value>,
    },
    ContentBlockDelta {
        delta: Map<String, Value>,
    },
    ContentBlockStop {},
    MessageDelta {
        #[serde(default)]
        delta: Map<String, Value>,
        #[serde(default)]
        usage: Option<Map<String, Value>>,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct StartMessage {
    #[serde(default)]
    id: Option<Value>,
    #[serde(default)]
    usage: Option<Map<String, Value>>,
    #[serde(default)]
    input_transformations: Option<Value>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct ToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// The cumulative usage a usage-bearing chunk counted, and how much reasoning text the stream
/// had sent by then.
#[derive(Clone, Debug, PartialEq)]
struct UsageChunk {
    usage: Map<String, Value>,
    reasoning_len: Option<usize>,
}

/// What a Messages stream's chunks add up to so far, built from its events as they pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamedMessage {
    /// The `message_start` id.
    id: Option<String>,
    /// The type of the content block open now.
    open_block: Option<String>,
    /// The thinking blocks started so far, one more than the index the chunk parser gives the
    /// open one.
    thinking_started: i64,
    text: String,
    tool_calls: Vec<ToolCall>,
    /// The thinking text each chunk reports, joined: the reasoning the usage estimate reads.
    reasoning: String,
    /// Whether a chunk reported reasoning, empty text included, so the message carries it.
    reasoned: bool,
    /// `get_combined_thinking_content` so far: the blocks it closed, and the text of the one
    /// it has not.
    thinking_blocks: Vec<Value>,
    open_thinking: String,
    citations: Vec<Value>,
    /// Each chunk's `provider_specific_fields` laid over the earlier ones' in stream order, as
    /// `stream_chunk_builder` merges them.
    provider_fields: Map<String, Value>,
    refusal: String,
    finish_reason: Option<&'static str>,
    cumulative_usage: Map<String, Value>,
    usage_chunks: Vec<UsageChunk>,
    /// The first block whose logged fields the Rust builder does not build.
    unported: Option<String>,
}

const TEXT: &str = "text";
const THINKING: &str = "thinking";
const SIGNATURE: &str = "signature";
const REDACTED_THINKING: &str = "redacted_thinking";
const TOOL_USE: &str = "tool_use";
const THINKING_BLOCKS: &str = "thinking_blocks";

fn non_empty_text<'a>(fields: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    fields
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

impl StreamedMessage {
    /// Reads one event's `data`. Data that is not a Messages event adds nothing, as Python
    /// skips a frame it cannot decode.
    pub fn see(&mut self, data: &str) {
        let Ok(event) = serde_json::from_str::<Event>(data) else {
            return;
        };
        match event {
            Event::MessageStart { message } => self.start(message),
            Event::ContentBlockStart { content_block } => self.start_block(&content_block),
            Event::ContentBlockDelta { delta } => self.delta(&delta),
            Event::ContentBlockStop {} => self.open_block = None,
            Event::MessageDelta { delta, usage } => self.message_delta(&delta, usage),
            Event::Other => {}
        }
    }

    fn start(&mut self, message: StartMessage) {
        if self.id.is_none() {
            self.id = message
                .id
                .as_ref()
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map(str::to_owned);
        }
        if let Some(input_transformations) = message.input_transformations.filter(|v| !v.is_null())
        {
            self.provider_field("input_transformations", input_transformations);
        }
        if let Some(usage) = message.usage {
            self.count(usage);
        }
    }

    fn start_block(&mut self, block: &Map<String, Value>) {
        let kind = block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        self.open_block = Some(kind.to_owned());
        if kind == THINKING || kind == REDACTED_THINKING {
            self.thinking_started += 1;
        }
        match kind {
            TEXT => {
                if let Some(text) = block.get(TEXT).and_then(Value::as_str) {
                    self.text.push_str(text);
                }
            }
            TOOL_USE => {
                self.tool_calls.push(ToolCall {
                    id: non_empty_text(block, "id").map(str::to_owned),
                    name: non_empty_text(block, "name").map(str::to_owned),
                    arguments: String::new(),
                });
            }
            REDACTED_THINKING => {
                let data = block.get("data").cloned().unwrap_or(Value::Null);
                self.provider_field(
                    THINKING_BLOCKS,
                    json!([{"type": REDACTED_THINKING, "data": data, "index": self.thinking_index()}]),
                );
                self.open_thinking.clear();
                if data.as_str().is_some_and(|data| !data.is_empty()) {
                    self.thinking_blocks
                        .push(json!({"type": REDACTED_THINKING, "data": data}));
                }
            }
            _ if kind == "server_tool_use"
                || kind.ends_with("_tool_result")
                || kind == "compaction" =>
            {
                self.unported.get_or_insert_with(|| kind.to_owned());
            }
            _ => {}
        }
    }

    fn delta(&mut self, delta: &Map<String, Value>) {
        if let Some(text) = delta.get(TEXT) {
            if let Some(text) = text.as_str() {
                self.text.push_str(text);
            }
            return;
        }
        if let Some(partial_json) = delta.get("partial_json") {
            if self.open_block.as_deref() == Some(TOOL_USE)
                && let (Some(partial_json), Some(call)) =
                    (partial_json.as_str(), self.tool_calls.last_mut())
            {
                call.arguments.push_str(partial_json);
            }
            return;
        }
        if let Some(citation) = delta.get("citation") {
            if !citation.is_null() {
                self.citations.push(citation.clone());
            }
            return;
        }
        if delta.contains_key(THINKING) || delta.contains_key(SIGNATURE) {
            self.thinking_delta(delta);
        }
    }

    /// A `thinking_delta` or `signature_delta`: the chunk reports its text as reasoning, and its
    /// signature closes the thinking block `get_combined_thinking_content` has open.
    fn thinking_delta(&mut self, delta: &Map<String, Value>) {
        let index = self.thinking_index();
        if let Some(thinking) = non_empty_text(delta, THINKING) {
            self.reasoning.push_str(thinking);
            self.reasoned = true;
            if non_empty_text(delta, SIGNATURE).is_none() {
                self.open_thinking.push_str(thinking);
                self.provider_field(
                    THINKING_BLOCKS,
                    json!([{"type": THINKING, "thinking": thinking, "index": index}]),
                );
            }
        }
        if let Some(signature) = non_empty_text(delta, SIGNATURE) {
            self.reasoned = true;
            self.provider_field(
                THINKING_BLOCKS,
                json!([{"type": THINKING, "thinking": "", "signature": signature, "index": index}]),
            );
            let thinking = std::mem::take(&mut self.open_thinking);
            self.thinking_blocks
                .push(json!({"type": THINKING, "thinking": thinking, "signature": signature}));
        }
    }

    fn message_delta(&mut self, delta: &Map<String, Value>, usage: Option<Map<String, Value>>) {
        let stop_reason = delta.get(STOP_REASON);
        self.finish_reason = Some(finish_reason(
            stop_reason
                .and_then(Value::as_str)
                .filter(|reason| !reason.is_empty()),
        ));
        if let Some(usage) = usage {
            self.count(usage);
        }
        let stop_details = delta.get(STOP_DETAILS);
        for (key, value) in stop_reason_fields(stop_reason, stop_details) {
            self.provider_fields.insert(key, value);
        }
        if let Some(container) = delta.get("container").filter(|container| truthy(container)) {
            self.provider_field("container", container.clone());
        }
        if let Some(refused) = refusal(stop_reason, stop_details) {
            self.refusal.push_str(&refused);
        }
    }

    /// The index the chunk parser gives the thinking block open now, `-1` before any.
    fn thinking_index(&self) -> i64 {
        self.thinking_started - 1
    }

    fn provider_field(&mut self, key: &str, value: Value) {
        self.provider_fields.insert(key.to_owned(), value);
    }

    /// `_handle_usage`: a chunk's usage over the ones before it, as Anthropic may leave out of
    /// `message_delta` what `message_start` reported.
    fn count(&mut self, usage: Map<String, Value>) {
        self.cumulative_usage.extend(usage);
        self.usage_chunks.push(UsageChunk {
            usage: self.cumulative_usage.clone(),
            reasoning_len: (!self.reasoning.is_empty()).then_some(self.reasoning.len()),
        });
    }

    /// The `ModelResponse` Python logs for the stream so far, `model` the one its handler priced
    /// the call with.
    pub fn response(
        &self,
        model: &str,
        context: &ResponseContext<'_>,
    ) -> Result<LoggedResponse, Error> {
        if let Some(kind) = &self.unported {
            return Err(Error::UnportedBlock(kind.clone()));
        }
        let counted = self
            .usage_chunks
            .iter()
            .map(|chunk| {
                Ok(calculate(
                    ReportedUsage::read(&chunk.usage)?,
                    Reasoning {
                        content: chunk.reasoning_len.map(|len| &self.reasoning[..len]),
                        has_thinking_block: false,
                    },
                    context.requested_speed,
                    context.count_tokens,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let usage = StreamedUsage::of(counted).ok_or(Error::NoUsage)?;
        Ok(LoggedResponse {
            id: self.id.as_deref().unwrap_or(context.fallback_id).to_owned(),
            created: context.created,
            model: model.to_owned(),
            object: CHAT_COMPLETION,
            system_fingerprint: Some(Null),
            choices: vec![LoggedChoice {
                finish_reason: self.finish_reason.unwrap_or(super::STOP),
                index: 0,
                message: LoggedMessage::Streamed(self.message()),
            }],
            usage: LoggedUsage::Streamed(usage),
        })
    }

    fn message(&self) -> StreamedChatMessage {
        let tool_calls = (!self.tool_calls.is_empty()).then(|| {
            self.tool_calls
                .iter()
                .filter_map(|call| {
                    Some(StreamedToolCall {
                        function: ToolFunction {
                            arguments: if call.arguments.is_empty() {
                                "{}".to_owned()
                            } else {
                                call.arguments.clone()
                            },
                            name: call.name.clone()?,
                        },
                        id: call.id.clone()?,
                        kind: FUNCTION,
                    })
                })
                .collect()
        });
        let citations =
            (!self.citations.is_empty()).then(|| ("citations".to_owned(), json!([self.citations])));
        let provider_specific_fields: Map<String, Value> = citations
            .into_iter()
            .chain(self.provider_fields.clone())
            .collect();
        StreamedChatMessage {
            content: self.text.clone(),
            role: ASSISTANT,
            tool_calls,
            function_call: Null,
            provider_specific_fields: (!provider_specific_fields.is_empty())
                .then_some(provider_specific_fields),
            thinking_blocks: (!self.thinking_blocks.is_empty())
                .then(|| self.thinking_blocks.clone()),
            reasoning_content: self.reasoned.then(|| self.reasoning.clone()),
            refusal: (!self.refusal.is_empty()).then(|| self.refusal.clone()),
        }
    }
}

/// Python's truthiness of a decoded JSON value, which decides whether a `container` is kept.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::super::tests::{CREATED, context, counter, fixture};
    use super::*;

    fn streamed(name: &str) -> (StreamedMessage, super::super::tests::Fixture) {
        let fixture = fixture(name);
        let mut message = StreamedMessage::default();
        for event in &fixture.events {
            message.see(&event.to_string());
        }
        (message, fixture)
    }

    #[rstest]
    #[case::text("text")]
    #[case::max_tokens("max_tokens")]
    #[case::stop_sequence("stop_sequence")]
    #[case::tool_use("tool_use")]
    #[case::two_tools("two_tools")]
    #[case::thinking("thinking")]
    #[case::redacted_thinking("redacted_thinking")]
    #[case::cache("cache")]
    #[case::empty("empty")]
    #[case::refusal("refusal")]
    #[case::pause_turn("pause_turn")]
    #[case::citations("citations")]
    #[case::speed("speed")]
    #[case::thinking_estimate("thinking_estimate")]
    #[case::thinking_tool("thinking_tool")]
    #[case::two_thinking("two_thinking")]
    #[case::server_tool_usage("server_tool_usage")]
    #[case::all_usage_fields("all_usage_fields")]
    #[case::stop_details_other("stop_details_other")]
    #[case::input_transformations("input_transformations")]
    #[case::context_management("context_management")]
    #[case::requested_speed("requested_speed")]
    #[case::cumulative_delta_usage("cumulative_delta_usage")]
    #[case::unicode_text("unicode_text")]
    #[case::tool_no_input("tool_no_input")]
    #[case::tool_caller("tool_caller")]
    #[case::two_citations("two_citations")]
    #[case::null_stop("null_stop")]
    fn a_stream_logs_what_python_logs(#[case] name: &str) {
        let (message, fixture) = streamed(name);
        let count_tokens = counter(&fixture);
        let model = fixture.streamed["model"].as_str().unwrap_or_default();
        let logged = message
            .response(model, &context(&fixture, &count_tokens))
            .expect("loggable");
        assert_eq!(
            serde_json::to_string(&logged).expect("serializable"),
            fixture.streamed.to_string()
        );
    }

    #[rstest]
    fn a_stream_without_an_id_logs_the_call_id() {
        let mut message = StreamedMessage::default();
        message.see(
            r#"{"type":"message_start","message":{"usage":{"input_tokens":1,"output_tokens":1}}}"#,
        );
        let fixture = fixture("text");
        let count_tokens = counter(&fixture);
        let logged = message
            .response("claude", &context(&fixture, &count_tokens))
            .expect("loggable");
        assert_eq!(
            (logged.id.as_str(), logged.created),
            ("chatcmpl-fallback", CREATED)
        );
    }

    #[rstest]
    #[case::a_hosted_tool_call(r#"{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}"#)]
    #[case::compaction(r#"{"type":"content_block_start","index":0,"content_block":{"type":"compaction","content":"summary"}}"#)]
    fn a_block_the_rust_builder_does_not_port_is_refused(#[case] event: &str) {
        let (mut message, fixture) = streamed("text");
        message.see(event);
        let count_tokens = counter(&fixture);
        assert!(matches!(
            message.response("claude", &context(&fixture, &count_tokens)),
            Err(Error::UnportedBlock(_))
        ));
    }

    #[rstest]
    #[case::no_events(None)]
    #[case::undecodable(Some("not json"))]
    #[case::a_ping(Some(r#"{"type":"ping"}"#))]
    fn a_stream_that_counted_nothing_is_refused(#[case] event: Option<&str>) {
        let mut message = StreamedMessage::default();
        if let Some(event) = event {
            message.see(event);
        }
        let fixture = fixture("text");
        let count_tokens = counter(&fixture);
        assert!(matches!(
            message.response("claude", &context(&fixture, &count_tokens)),
            Err(Error::NoUsage)
        ));
    }
}
