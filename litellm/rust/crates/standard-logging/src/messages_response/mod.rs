//! The `response` of a served `/v1/messages` call's logging payload. Python logs an
//! `anthropic_messages` call's answer as a chat `ModelResponse` (`_handle_anthropic_messages_response_logging`
//! in `litellm_logging.py`) and the payload carries its `model_dump()`
//! (`StandardLoggingPayloadSetup.get_final_response_obj`). The two ways Python builds that
//! `ModelResponse` give it two shapes, so each has its own builder here:
//!
//! - a complete answer goes through `AnthropicConfig.transform_response` ([`complete_response`])
//! - a stream goes through the Anthropic passthrough logging handler, which reads its events as
//!   chat chunks and joins them with `stream_chunk_builder` ([`StreamedMessage`])
//!
//! Both shapes are pinned against `model_dump()` output captured from Python in
//! `fixtures/messages_response.json`.

mod complete;
mod stream;
mod usage;

use serde::{Serialize, Serializer};
use serde_json::{Map, Value};

pub use complete::complete_response;
pub use stream::StreamedMessage;
pub use usage::{CompleteUsage, StreamedUsage};

/// What a logged response is built with besides the answer.
#[derive(Clone, Copy)]
pub struct ResponseContext<'a> {
    /// The `created` second, which Python stamps with `time.time()` as it builds the response.
    pub created: u64,
    /// The id Python logs when the answer names none: the `chatcmpl-` id `ModelResponse` mints
    /// for a complete answer, the call's `litellm_call_id` for a stream.
    pub fallback_id: &'a str,
    /// The request's `speed`, which the usage names when the answer's own usage does not.
    pub requested_speed: Option<&'a str>,
    /// Python's `token_counter(text=..., count_response_tokens=True)`, which estimates the
    /// reasoning tokens of an answer whose usage does not report them.
    pub count_tokens: &'a dyn Fn(&str) -> u64,
}

/// An answer the Rust builders do not log as Python would.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The answer is not a message Python's transform reads.
    #[error("the answer is not a readable Messages response")]
    Unreadable(#[source] serde_json::Error),
    /// The answer is Anthropic's error body, which Python's transform raises on.
    #[error("the answer is an error body")]
    ErrorBody,
    /// A content block whose logged fields (server tool calls, tool results, compaction) the
    /// Rust builders do not build.
    #[error("the answer carries a {0} block, which is not logged by the Rust builders")]
    UnportedBlock(String),
    /// A usage block that sums per-iteration usage, which the Rust builders do not port.
    #[error("the answer's usage reports iterations")]
    UnportedIterations,
    /// A stream that carried no usage, whose tokens Python counts with its tokenizer.
    #[error("the stream carried no usage")]
    NoUsage,
    /// A tool call's input Python's `json.dumps` would not write.
    #[error("a tool call's input could not be written as Python writes it")]
    ToolInput(#[source] litellm_python_compat::Error),
}

/// A key Python's `model_dump()` always writes as `null` for this answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Null;

impl Serialize for Null {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_none()
    }
}

/// `ModelResponse.model_dump()` of a served Messages answer.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LoggedResponse {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub object: &'static str,
    /// Written, as `null`, only by a stream, whose builder passes it to `ModelResponse`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<Null>,
    pub choices: Vec<LoggedChoice>,
    pub usage: LoggedUsage,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LoggedChoice {
    pub finish_reason: &'static str,
    pub index: u64,
    pub message: LoggedMessage,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum LoggedMessage {
    Complete(CompleteMessage),
    Streamed(StreamedChatMessage),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum LoggedUsage {
    Complete(CompleteUsage),
    Streamed(StreamedUsage),
}

/// The `Message` `AnthropicConfig.transform_parsed_response` builds, its keys in the order its
/// `model_dump()` writes them.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CompleteMessage {
    pub content: Option<String>,
    pub role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    pub tool_calls: Option<Vec<CompleteToolCall>>,
    pub function_call: Null,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_blocks: Option<Vec<Map<String, Value>>>,
    pub provider_specific_fields: Map<String, Value>,
}

/// A `tool_use` block as `convert_tool_use_to_openai_format` turns it into a tool call.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CompleteToolCall {
    /// The block's position among the answer's content blocks.
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caller: Option<Value>,
    pub function: ToolFunction,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolFunction {
    pub arguments: String,
    pub name: String,
}

/// The `Message` `stream_chunk_builder` joins a stream's chunks into, its keys in the order its
/// `model_dump()` writes them.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StreamedChatMessage {
    pub content: String,
    pub role: &'static str,
    pub tool_calls: Option<Vec<StreamedToolCall>>,
    pub function_call: Null,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_specific_fields: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_blocks: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StreamedToolCall {
    pub function: ToolFunction,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
}

const CHAT_COMPLETION: &str = "chat.completion";
const ASSISTANT: &str = "assistant";
const FUNCTION: &str = "function";
const STOP: &str = "stop";
const STOP_REASON: &str = "stop_reason";
const STOP_DETAILS: &str = "stop_details";
const REFUSAL: &str = "refusal";

/// Python's `map_finish_reason` for the reasons an Anthropic answer stops on. Python warns and
/// answers `stop` for any other.
fn finish_reason(stop_reason: Option<&str>) -> &'static str {
    match stop_reason {
        Some("max_tokens" | "compaction" | "pause_turn" | "model_context_window_exceeded") => {
            "length"
        }
        Some("tool_use") => "tool_calls",
        Some(REFUSAL) => "content_filter",
        _ => STOP,
    }
}

/// `stop_reason_provider_specific_fields`: the stop reason and its details when the answer
/// names details, else the stop reason alone for one Python keeps raw.
fn stop_reason_fields(
    stop_reason: Option<&Value>,
    stop_details: Option<&Value>,
) -> Map<String, Value> {
    let named_reason = || {
        (
            STOP_REASON.to_owned(),
            stop_reason.cloned().unwrap_or(Value::Null),
        )
    };
    match stop_details.filter(|details| !details.is_null()) {
        Some(details) => {
            Map::from_iter([named_reason(), (STOP_DETAILS.to_owned(), details.clone())])
        }
        None if matches!(
            stop_reason.and_then(Value::as_str),
            Some(REFUSAL | "pause_turn")
        ) =>
        {
            Map::from_iter([named_reason()])
        }
        None => Map::new(),
    }
}

/// `refusal_params`: a refusal's explanation, when its stop details give a non-empty one.
fn refusal(stop_reason: Option<&Value>, stop_details: Option<&Value>) -> Option<String> {
    if stop_reason.and_then(Value::as_str) != Some(REFUSAL) {
        return None;
    }
    stop_details?
        .get("explanation")?
        .as_str()
        .filter(|explanation| !explanation.is_empty())
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde::Deserialize;

    use super::*;

    /// One answer as `fixtures/messages_response.json` holds it: the complete body, the same
    /// answer as stream events, and what Python logged for each, with `created` set to
    /// [`CREATED`] and the stream's id and model as the passthrough handler stamps them.
    #[derive(Deserialize)]
    pub(super) struct Fixture {
        pub(super) body: Map<String, Value>,
        pub(super) events: Vec<Value>,
        pub(super) requested_speed: Option<String>,
        /// What Python's `token_counter` counted for each reasoning text.
        pub(super) token_counts: HashMap<String, u64>,
        pub(super) complete: Value,
        pub(super) streamed: Value,
    }

    pub(super) const CREATED: u64 = 1_700_000_000;

    pub(super) fn fixture(name: &str) -> Fixture {
        let mut all: HashMap<String, Fixture> =
            serde_json::from_str(include_str!("../../fixtures/messages_response.json"))
                .expect("fixture file parses");
        all.remove(name).expect("fixture exists")
    }

    pub(super) fn context<'a>(
        fixture: &'a Fixture,
        count_tokens: &'a dyn Fn(&str) -> u64,
    ) -> ResponseContext<'a> {
        ResponseContext {
            created: CREATED,
            fallback_id: "chatcmpl-fallback",
            requested_speed: fixture.requested_speed.as_deref(),
            count_tokens,
        }
    }

    pub(super) fn counter(fixture: &Fixture) -> impl Fn(&str) -> u64 + '_ {
        |text| fixture.token_counts.get(text).copied().unwrap_or(0)
    }
}
