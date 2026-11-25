//! A Messages stream as Python logs it: the Anthropic passthrough logging handler reads each
//! event as a chat chunk (`ModelResponseIterator.chunk_parser` in `llms/anthropic/chat/handler.py`)
//! and `stream_chunk_builder` joins the chunks into one `ModelResponse`
//! (`build_complete_streaming_response` in `llms/anthropic/messages/stream_assembly.py`). The
//! handler then names the response by the `message_start` id and the model it priced
//! (`_create_anthropic_response_logging_payload`).
//!
//! [`StreamedMessage`] keeps, event by event, what those chunks add up to, so the stream's
//! events need not be held.

use litellm_python_compat::{
    json::{dumps, from_json},
    truthy::json_truthy,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{
    ASSISTANT, CHAT_COMPLETION, Error, FUNCTION, LoggedChoice, LoggedMessage, LoggedResponse,
    LoggedUsage, Null, ResponseContext, STOP_DETAILS, STOP_REASON, StreamedChatMessage,
    StreamedToolCall, ToolFunction, finish_reason, refusal,
    server_tools::{
        CODE_EXECUTION_RESULT, TOOL_SEARCH_RESULT, code_interpreter_call, is_hosted_web_tool,
        is_web_result, server_tool_call, web_search_call,
    },
    stop_reason_fields,
    usage::{Reasoning, ReportedUsage, StreamedOutput, StreamedUsage, calculate},
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
    model: Option<Value>,
    #[serde(default)]
    usage: Option<Map<String, Value>>,
    #[serde(default)]
    input_transformations: Option<Value>,
}

/// The `server_tool_use` block open now: its id and tool.
#[derive(Clone, Debug, PartialEq)]
struct ServerToolCall {
    id: String,
    name: String,
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
    /// The model of the stream's first chunk, which `stream_chunk_builder` estimates an
    /// unreported output count with: the `message_start` model when the stream opens with one.
    /// `None` before the first event.
    first_model: Option<Option<String>>,
    /// Whether the last of the stream's `message_start`, `content_block_delta`,
    /// `content_block_stop` and `message_delta` events is a `message_delta`.
    ended_on_message_delta: bool,
    /// The `input_json_delta` text of the block open now.
    open_input: String,
    open_server_tool: Option<ServerToolCall>,
    /// The latest input of each server tool call, by id.
    server_tool_inputs: Map<String, Value>,
    /// Each hosted web tool call as it stands, by id.
    web_search_calls: Map<String, Value>,
    server_tool_calls: Vec<Value>,
    web_search_results: Vec<Value>,
    tool_results: Vec<Map<String, Value>>,
    compaction_blocks: Vec<Value>,
    container_id: Option<String>,
}

const TEXT: &str = "text";
const THINKING: &str = "thinking";
const SIGNATURE: &str = "signature";
const REDACTED_THINKING: &str = "redacted_thinking";
const TOOL_USE: &str = "tool_use";
const SERVER_TOOL_USE: &str = "server_tool_use";
const COMPACTION: &str = "compaction";
const CONTENT: &str = "content";
const WEB_SEARCH_CALLS: &str = "web_search_calls";
const CODE_INTERPRETER_RESULTS: &str = "code_interpreter_results";
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
        if self.first_model.is_none() {
            self.first_model = Some(match &event {
                Event::MessageStart { message } => message
                    .model
                    .as_ref()
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                _ => None,
            });
        }
        match event {
            Event::MessageStart { message } => {
                self.ended_on_message_delta = false;
                self.start(message);
            }
            Event::ContentBlockStart { content_block } => self.start_block(&content_block),
            Event::ContentBlockDelta { delta } => {
                self.ended_on_message_delta = false;
                self.delta(&delta);
            }
            Event::ContentBlockStop {} => {
                self.ended_on_message_delta = false;
                self.stop_block();
            }
            Event::MessageDelta { delta, usage } => {
                self.ended_on_message_delta = true;
                self.message_delta(&delta, usage);
            }
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
        self.open_input.clear();
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
            SERVER_TOOL_USE => self.start_server_tool(block),
            COMPACTION => {
                self.compaction_blocks.push(Value::Object(block.clone()));
                self.provider_field(
                    "compaction_blocks",
                    Value::Array(self.compaction_blocks.clone()),
                );
                let content = block.get(CONTENT).cloned().unwrap_or_else(|| json!(""));
                self.provider_field(
                    "compaction_start",
                    json!({"type": COMPACTION, "content": content}),
                );
            }
            _ if kind.ends_with("_tool_result") => self.tool_result(kind, block),
            _ => {}
        }
    }

    fn start_server_tool(&mut self, block: &Map<String, Value>) {
        let (Some(id), Some(name)) = (
            block.get("id").and_then(Value::as_str),
            block.get("name").and_then(Value::as_str),
        ) else {
            return;
        };
        let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
        self.server_tool_inputs.insert(id.to_owned(), input.clone());
        self.open_server_tool = Some(ServerToolCall {
            id: id.to_owned(),
            name: name.to_owned(),
        });
        if is_hosted_web_tool(name) {
            self.web_search_calls
                .insert(id.to_owned(), pending_web_search_call(id, &input, name));
            self.provider_field(
                WEB_SEARCH_CALLS,
                Value::Object(self.web_search_calls.clone()),
            );
        }
    }

    /// A `*_tool_result` block, whose whole result comes with its start.
    fn tool_result(&mut self, kind: &str, block: &Map<String, Value>) {
        if is_web_result(kind) {
            self.web_search_results.push(Value::Object(block.clone()));
            let tool_name = kind.trim_end_matches("_tool_result");
            if let Some(id) = block.get("tool_use_id").and_then(Value::as_str)
                && self.web_search_calls.contains_key(id)
            {
                let input = self.server_tool_input(id);
                let call = web_search_call(
                    id,
                    &input,
                    Some(&Value::Object(block.clone())),
                    None,
                    tool_name,
                );
                self.web_search_calls.insert(id.to_owned(), call);
            }
            self.provider_field(
                "web_search_results",
                Value::Array(self.web_search_results.clone()),
            );
            self.provider_field(
                WEB_SEARCH_CALLS,
                Value::Object(self.web_search_calls.clone()),
            );
            return;
        }
        self.tool_results.push(block.clone());
        let results = self
            .tool_results
            .iter()
            .cloned()
            .map(Value::Object)
            .collect();
        self.provider_field("tool_results", Value::Array(results));
        if kind != TOOL_SEARCH_RESULT {
            self.provider_field(CODE_INTERPRETER_RESULTS, self.code_interpreter_results());
        }
    }

    /// `content_block_stop`: a server tool call's input is whole, so its call is listed.
    fn stop_block(&mut self) {
        let open_block = self.open_block.take();
        let Some(ServerToolCall { id, name }) = self
            .open_server_tool
            .take()
            .filter(|_| open_block.as_deref() == Some(SERVER_TOOL_USE))
        else {
            return;
        };
        let arguments = std::mem::take(&mut self.open_input);
        if let Ok(input) = serde_json::from_str::<Value>(&arguments) {
            self.server_tool_inputs.insert(id.clone(), input);
        }
        let input = self.server_tool_input(&id);
        if self.web_search_calls.contains_key(&id) {
            self.web_search_calls
                .insert(id.clone(), pending_web_search_call(&id, &input, &name));
            self.provider_field(
                WEB_SEARCH_CALLS,
                Value::Object(self.web_search_calls.clone()),
            );
        }
        let arguments = match arguments.is_empty() {
            true => dumps(&from_json(input)).unwrap_or_default(),
            false => arguments,
        };
        let call = server_tool_call(&id, &name, &arguments, self.server_tool_calls.len(), None);
        self.server_tool_calls.push(call);
        self.provider_field(
            "server_tool_calls",
            Value::Array(self.server_tool_calls.clone()),
        );
    }

    fn server_tool_input(&self, id: &str) -> Value {
        self.server_tool_inputs
            .get(id)
            .cloned()
            .unwrap_or_else(|| json!({}))
    }

    /// `_build_code_interpreter_results` over the results so far.
    fn code_interpreter_results(&self) -> Value {
        let calls = self
            .tool_results
            .iter()
            .filter(|result| {
                result.get("type").and_then(Value::as_str) == Some(CODE_EXECUTION_RESULT)
            })
            .map(|result| {
                let call_id = result
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let code = self
                    .server_tool_inputs
                    .get(call_id)
                    .and_then(|input| input.get("command"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                code_interpreter_call(result, code, self.container_id.as_deref())
            })
            .collect();
        Value::Array(calls)
    }

    fn delta(&mut self, delta: &Map<String, Value>) {
        if let Some(text) = delta.get(TEXT) {
            if let Some(text) = text.as_str() {
                self.text.push_str(text);
            }
            return;
        }
        if delta.get("type").and_then(Value::as_str) == Some("input_json_delta")
            && let Some(partial_json) = delta.get("partial_json").and_then(Value::as_str)
        {
            self.open_input.push_str(partial_json);
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
            return;
        }
        if let Some(content) = delta.get(CONTENT)
            && delta.get("type").and_then(Value::as_str) == Some("compaction_delta")
        {
            self.compaction_delta(content);
        }
    }

    /// A `compaction_delta`, whose summary text joins the open compaction block's.
    fn compaction_delta(&mut self, content: &Value) {
        self.provider_field(
            "compaction_delta",
            json!({"type": "compaction_delta", "content": content}),
        );
        let (Some(Value::Object(block)), Some(content)) =
            (self.compaction_blocks.last_mut(), content.as_str())
        else {
            return;
        };
        let summary = block
            .get(CONTENT)
            .and_then(Value::as_str)
            .unwrap_or_default();
        let summary = format!("{summary}{content}");
        block.insert(CONTENT.to_owned(), Value::from(summary));
        self.provider_field(
            "compaction_blocks",
            Value::Array(self.compaction_blocks.clone()),
        );
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
        if let Some(container) = delta
            .get("container")
            .filter(|container| json_truthy(container))
        {
            self.provider_field("container", container.clone());
            if container.get("id").is_some_and(json_truthy) && !self.tool_results.is_empty() {
                self.container_id = container
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                self.provider_field(CODE_INTERPRETER_RESULTS, self.code_interpreter_results());
            }
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
        let estimated_model = self.first_model.clone().ok_or(Error::NoEvents)?;
        let counted = self
            .usage_chunks
            .iter()
            .map(|chunk| {
                calculate(
                    ReportedUsage::read(&chunk.usage),
                    Reasoning {
                        content: chunk.reasoning_len.map(|len| &self.reasoning[..len]),
                        has_thinking_block: false,
                    },
                    0,
                    context.requested_speed,
                    context.count_tokens,
                )
            })
            .collect();
        let message = self.message();
        let output_text = output_text(&message);
        let count_estimated = |text: &str| (context.count_tokens)(estimated_model.as_deref(), text);
        let count_recovered = |text: &str| (context.count_tokens)(Some(model), text);
        let usage = StreamedUsage::of(
            counted,
            &StreamedOutput {
                finished: self.finish_reason.is_some(),
                interrupted: !self.ended_on_message_delta,
                text: &output_text,
                reasoning_tokens: self.reasoning_tokens(context),
                count_estimated: &count_estimated,
                count_recovered: &count_recovered,
            },
        );
        Ok(LoggedResponse {
            id: self.id.as_deref().unwrap_or(context.fallback_id).to_owned(),
            created: context.created,
            model: model.to_owned(),
            object: CHAT_COMPLETION,
            system_fingerprint: Some(Null),
            choices: vec![LoggedChoice {
                finish_reason: self.finish_reason.unwrap_or(super::STOP),
                index: 0,
                message: LoggedMessage::Streamed(message),
            }],
            usage: LoggedUsage::Streamed(usage),
        })
    }

    /// The reasoning tokens `stream_chunk_builder` adds to an estimated output count: `0` when
    /// every chunk carried text alone, else `count_reasoning_tokens` over the joined reasoning.
    fn reasoning_tokens(&self, context: &ResponseContext<'_>) -> Option<u64> {
        let carries_more_than_text = !self.tool_calls.is_empty()
            || self.reasoned
            || !self.provider_fields.is_empty()
            || !self.citations.is_empty()
            || !self.refusal.is_empty();
        match carries_more_than_text {
            false => Some(0),
            true => {
                (!self.reasoning.is_empty()).then(|| (context.count_tokens)(None, &self.reasoning))
            }
        }
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
            thinking_blocks: self
                .provider_fields
                .contains_key(THINKING_BLOCKS)
                .then(|| (!self.thinking_blocks.is_empty()).then(|| self.thinking_blocks.clone())),
            reasoning_content: self.reasoned.then(|| self.reasoning.clone()),
            refusal: (!self.refusal.is_empty()).then(|| self.refusal.clone()),
        }
    }
}

fn pending_web_search_call(id: &str, input: &Value, tool_name: &str) -> Value {
    web_search_call(
        id,
        input,
        Some(&json!({"content": []})),
        Some("in_progress"),
        tool_name,
    )
}

fn text_of(value: Option<&Value>) -> &str {
    value.and_then(Value::as_str).unwrap_or_default()
}

/// A tool call as `ChatCompletionMessageToolCall.model_dump_json()` writes it.
fn dumped_tool_call(id: &str, name: &str, arguments: &str) -> String {
    json!({"function": {"arguments": arguments, "name": name}, "id": id, "type": FUNCTION})
        .to_string()
}

/// `get_content_from_model_response`: the text, then each server tool call and tool call as
/// its JSON.
fn output_text(message: &StreamedChatMessage) -> String {
    let server_tool_calls = message
        .provider_specific_fields
        .as_ref()
        .and_then(|fields| fields.get("server_tool_calls"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|call| {
            let function = call.get("function");
            dumped_tool_call(
                text_of(call.get("id")),
                text_of(function.and_then(|function| function.get("name"))),
                text_of(function.and_then(|function| function.get("arguments"))),
            )
        });
    let tool_calls = message
        .tool_calls
        .iter()
        .flatten()
        .map(|call| dumped_tool_call(&call.id, &call.function.name, &call.function.arguments));
    std::iter::once(message.content.clone())
        .chain(server_tool_calls)
        .chain(tool_calls)
        .collect()
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
    #[case::web_search("web_search")]
    #[case::web_fetch("web_fetch")]
    #[case::web_search_error("web_search_error")]
    #[case::code_execution("code_execution")]
    #[case::code_execution_stderr("code_execution_stderr")]
    #[case::tool_search("tool_search")]
    #[case::compaction("compaction")]
    #[case::two_compactions("two_compactions")]
    #[case::iterations("iterations")]
    #[case::iterations_thinking("iterations_thinking")]
    #[case::no_usage("no_usage")]
    #[case::no_usage_no_delta("no_usage_no_delta")]
    #[case::no_usage_tool("no_usage_tool")]
    #[case::cut_before_delta("cut_before_delta")]
    #[case::cut_before_delta_stopped("cut_before_delta_stopped")]
    #[case::cut_thinking("cut_thinking")]
    #[case::cut_thinking_only("cut_thinking_only")]
    #[case::cut_short_text("cut_short_text")]
    #[case::cut_tool("cut_tool")]
    #[case::cut_tool_empty("cut_tool_empty")]
    #[case::cut_server_tool("cut_server_tool")]
    #[case::cut_citation("cut_citation")]
    #[case::cut_ends_on_start("cut_ends_on_start")]
    #[case::cut_after_stop_event("cut_after_stop_event")]
    #[case::delta_without_usage("delta_without_usage")]
    #[case::only_start("only_start")]
    #[case::no_message_start_model("no_message_start_model")]
    #[case::ping_first("ping_first")]
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
    #[case::no_events(None)]
    #[case::undecodable(Some("not json"))]
    fn a_stream_python_assembles_nothing_from_is_refused(#[case] event: Option<&str>) {
        let mut message = StreamedMessage::default();
        if let Some(event) = event {
            message.see(event);
        }
        let fixture = fixture("text");
        let count_tokens = counter(&fixture);
        assert!(matches!(
            message.response("claude", &context(&fixture, &count_tokens)),
            Err(Error::NoEvents)
        ));
    }

    #[rstest]
    fn a_placeholder_output_count_is_estimated_with_the_message_start_model() {
        let mut message = StreamedMessage::default();
        for event in [
            r#"{"type":"message_start","message":{"id":"msg_1","model":"claude-start","usage":{"input_tokens":10,"output_tokens":1}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"a long answer"}}"#,
        ] {
            message.see(event);
        }
        let fixture = fixture("text");
        let count_tokens = |model: Option<&str>, _: &str| match model {
            Some("claude-start") => 30,
            _ => 20,
        };
        let logged = message
            .response("claude-logged", &context(&fixture, &count_tokens))
            .expect("loggable");
        assert_eq!(
            serde_json::to_value(&logged.usage).expect("serializable")["completion_tokens"],
            30
        );
    }
}
