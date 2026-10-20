//! `litellm/llms/anthropic/pass_through/adapters/streaming_iterator.py` (`AnthropicStreamWrapper`):
//! a Chat Completions stream re-emitted as the Messages event sequence.

use futures_util::{StreamExt, TryStreamExt};
use litellm_llms_types::formats::messages::streaming::{
    MessagesContentBlock, MessagesContentBlockDelta, MessagesDelta, MessagesStreamEvent,
    MessagesStreamMessage, MessagesStreamUsage,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::transformation::{ChatToolCall, ChatUsage, ToolNames, message_id, stop_reason};
use crate::{
    Error, ErrorDetail,
    anthropic::common_utils::normalize_anthropic_tool_use_id,
    base_llm::{
        base_model_iterator::{ByteStream, StreamError, StreamTransformer, transform_stream},
        messages::streaming::EventStream,
        sse,
    },
};

/// A Chat Completions SSE body as Messages events, `model` naming the message they open.
pub fn anthropic_events(bytes: ByteStream, model: String, tool_names: ToolNames) -> EventStream {
    transform_stream(
        sse::event_stream::<ChatChunk>(bytes, sse::is_done),
        AnthropicStreamWrapper::new(model, tool_names),
    )
    .map_err(|error| match error {
        StreamError::Decode(error) | StreamError::Transform(error) => error,
    })
    .boxed()
}

#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    usage: Option<ChatUsage>,
    error: Option<Value>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct Delta {
    content: Option<String>,
    reasoning_content: Option<String>,
    refusal: Option<String>,
    tool_calls: Option<Vec<ChatToolCall>>,
}

/// The content block being streamed. A tool call is told apart by its index in the chat
/// stream and, for hosts that resend it on every fragment, its id.
#[derive(PartialEq)]
enum Block {
    Text,
    Thinking,
    ToolUse { call: u64, id: Option<String> },
}

/// Opens the message on the first chunk, a content block whenever the kind of content
/// changes, and holds the stop reason and usage for `message_delta` once the stream ends,
/// since a host sends its usage after the chunk that carries the finish reason.
struct AnthropicStreamWrapper {
    id: String,
    model: String,
    tool_names: ToolNames,
    started: bool,
    open: Option<Block>,
    opened: u64,
    finish_reason: Option<String>,
    refused: bool,
    usage: Option<ChatUsage>,
}

impl AnthropicStreamWrapper {
    fn new(model: String, tool_names: ToolNames) -> Self {
        Self {
            id: message_id(),
            model,
            tool_names,
            started: false,
            open: None,
            opened: 0,
            finish_reason: None,
            refused: false,
            usage: None,
        }
    }

    fn start(&mut self, events: &mut Vec<MessagesStreamEvent>) {
        if std::mem::replace(&mut self.started, true) {
            return;
        }
        events.push(MessagesStreamEvent::MessageStart {
            message: MessagesStreamMessage {
                id: self.id.clone(),
                message_type: "message".into(),
                role: "assistant".into(),
                model: self.model.clone(),
                content: Vec::new(),
                stop_reason: None,
                stop_sequence: None,
                usage: MessagesStreamUsage {
                    input_tokens: Some(0),
                    output_tokens: Some(0),
                    ..MessagesStreamUsage::default()
                },
                extra: Map::new(),
            },
        });
    }

    fn begin(
        &mut self,
        block: Block,
        content_block: MessagesContentBlock,
        events: &mut Vec<MessagesStreamEvent>,
    ) {
        self.close(events);
        events.push(MessagesStreamEvent::ContentBlockStart {
            index: self.opened,
            content_block,
        });
        self.open = Some(block);
        self.opened += 1;
    }

    fn close(&mut self, events: &mut Vec<MessagesStreamEvent>) {
        if self.open.take().is_some() {
            events.push(MessagesStreamEvent::ContentBlockStop {
                index: self.index(),
            });
        }
    }

    fn index(&self) -> u64 {
        self.opened.saturating_sub(1)
    }

    fn delta(&self, delta: MessagesContentBlockDelta) -> MessagesStreamEvent {
        MessagesStreamEvent::ContentBlockDelta {
            index: self.index(),
            delta,
        }
    }

    fn thinking(&mut self, thinking: String, events: &mut Vec<MessagesStreamEvent>) {
        if self.open != Some(Block::Thinking) {
            let start = MessagesContentBlock {
                thinking: Some(String::new()),
                signature: Some(String::new()),
                ..content_block("thinking")
            };
            self.begin(Block::Thinking, start, events);
        }
        events.push(self.delta(MessagesContentBlockDelta::ThinkingDelta { thinking }));
    }

    fn text(&mut self, text: String, events: &mut Vec<MessagesStreamEvent>) {
        if self.open != Some(Block::Text) {
            let start = MessagesContentBlock {
                text: Some(String::new()),
                ..content_block("text")
            };
            self.begin(Block::Text, start, events);
        }
        events.push(self.delta(MessagesContentBlockDelta::TextDelta { text }));
    }

    fn tool_call(&mut self, call: ChatToolCall, events: &mut Vec<MessagesStreamEvent>) {
        let index = call.index.unwrap_or(0);
        let continues = matches!(
            &self.open,
            Some(Block::ToolUse { call: open, id })
                if *open == index && (call.id.is_none() || call.id == *id)
        );
        if !continues {
            let id = call
                .id
                .as_deref()
                .map_or_else(tool_use_id, normalize_anthropic_tool_use_id);
            let name = call.function.name.unwrap_or_default();
            let start = MessagesContentBlock {
                id: Some(id),
                name: Some(self.tool_names.original(&name).to_owned()),
                input: Some(json!({})),
                ..content_block("tool_use")
            };
            let block = Block::ToolUse {
                call: index,
                id: call.id,
            };
            self.begin(block, start, events);
        }
        if let Some(partial_json) = call
            .function
            .arguments
            .filter(|arguments| !arguments.is_empty())
        {
            events.push(self.delta(MessagesContentBlockDelta::InputJsonDelta { partial_json }));
        }
    }
}

impl StreamTransformer for AnthropicStreamWrapper {
    type Input = ChatChunk;
    type Output = MessagesStreamEvent;
    type Error = Error;

    fn transform(&mut self, chunk: ChatChunk) -> Result<Vec<MessagesStreamEvent>, Error> {
        if let Some(detail) = chunk.error {
            return Err(Error::InvalidResponse(ErrorDetail::RemoteFailure {
                operation: "chat completions stream",
                detail,
            }));
        }
        let mut events = Vec::new();
        self.start(&mut events);
        if let Some(usage) = chunk.usage {
            self.usage = Some(usage);
        }
        for choice in chunk.choices {
            let Delta {
                content,
                reasoning_content,
                refusal,
                tool_calls,
            } = choice.delta;
            if let Some(thinking) = reasoning_content.filter(|text| !text.is_empty()) {
                self.thinking(thinking, &mut events);
            }
            self.refused |= refusal.as_deref().is_some_and(|text| !text.is_empty());
            for text in [content, refusal]
                .into_iter()
                .flatten()
                .filter(|text| !text.is_empty())
            {
                self.text(text, &mut events);
            }
            for call in tool_calls.into_iter().flatten() {
                self.tool_call(call, &mut events);
            }
            if choice.finish_reason.is_some() {
                self.finish_reason = choice.finish_reason;
            }
        }
        Ok(events)
    }

    fn finish(&mut self) -> Result<Vec<MessagesStreamEvent>, Error> {
        let mut events = Vec::new();
        self.start(&mut events);
        self.close(&mut events);
        events.push(MessagesStreamEvent::MessageDelta {
            delta: MessagesDelta {
                stop_reason: Some(stop_reason(self.finish_reason.as_deref(), self.refused).into()),
                ..MessagesDelta::default()
            },
            usage: Some(self.usage.take().unwrap_or_default().anthropic()),
            context_management: None,
        });
        events.push(MessagesStreamEvent::MessageStop { usage: None });
        Ok(events)
    }
}

fn content_block(block_type: &str) -> MessagesContentBlock {
    MessagesContentBlock {
        block_type: block_type.into(),
        id: None,
        name: None,
        text: None,
        input: None,
        thinking: None,
        signature: None,
        data: None,
        content: None,
        caller: None,
        extra: Map::new(),
    }
}

fn tool_use_id() -> String {
    format!("toolu_{}", uuid::Uuid::new_v4().simple())
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use futures_util::stream;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::anthropic::adapters::transformation::{
        translate_anthropic_to_openai, truncate_tool_name,
    };

    fn wire(chunks: &[Value]) -> String {
        chunks
            .iter()
            .map(|chunk| format!("data: {chunk}\n\n"))
            .chain(["data: [DONE]\n\n".to_owned()])
            .collect()
    }

    /// The events as `(type, the rest)`, the message id replaced since it is generated.
    async fn events(wire: &str, size: usize, tool_names: ToolNames) -> Result<Vec<Value>, Error> {
        let pieces: Vec<Result<Bytes, std::io::Error>> = wire
            .as_bytes()
            .chunks(size)
            .map(|piece| Ok(Bytes::copy_from_slice(piece)))
            .collect();
        anthropic_events(
            stream::iter(pieces).boxed(),
            "gpt-6-astra".into(),
            tool_names,
        )
        .map_ok(|event| {
            let mut event = serde_json::to_value(event).unwrap();
            if let Some(id) = event.pointer_mut("/message/id") {
                assert!(id.as_str().unwrap().starts_with("msg_"));
                *id = json!("msg");
            }
            event
        })
        .try_collect()
        .await
    }

    fn message_start() -> Value {
        json!({"type": "message_start", "message": {
            "id": "msg", "type": "message", "role": "assistant", "model": "gpt-6-astra",
            "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }})
    }

    fn delta(index: u64, delta: Value) -> Value {
        json!({"type": "content_block_delta", "index": index, "delta": delta})
    }

    fn start(index: u64, block: Value) -> Value {
        json!({"type": "content_block_start", "index": index, "content_block": block})
    }

    fn stop(index: u64) -> Value {
        json!({"type": "content_block_stop", "index": index})
    }

    #[rstest]
    #[tokio::test]
    async fn text_streams_as_one_block_with_usage_from_the_trailing_chunk(
        #[values(1, 7, 4096)] size: usize,
    ) {
        let wire = wire(&[
            json!({"id": "c", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}),
            json!({"id": "c", "choices": [{"index": 0, "delta": {"content": "Hel"}}]}),
            json!({"id": "c", "choices": [{"index": 0, "delta": {"content": "lo"}}]}),
            json!({"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            json!({"id": "c", "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2,
                "total_tokens": 14, "prompt_tokens_details": {"cached_tokens": 10}}}),
        ]);
        assert_eq!(
            events(&wire, size, ToolNames::default()).await.unwrap(),
            [
                message_start(),
                start(0, json!({"type": "text", "text": ""})),
                delta(0, json!({"type": "text_delta", "text": "Hel"})),
                delta(0, json!({"type": "text_delta", "text": "lo"})),
                stop(0),
                json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                    "usage": {"input_tokens": 2, "output_tokens": 2, "cache_read_input_tokens": 10}}),
                json!({"type": "message_stop"}),
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn reasoning_text_and_tool_calls_each_get_their_own_block() {
        let request = json!({"model": "m", "messages": [], "tools": [
            {"name": format!("{}_tool", "x".repeat(70)), "input_schema": {"type": "object"}},
        ]});
        let tool_names = translate_anthropic_to_openai(serde_json::from_value(request).unwrap())
            .unwrap()
            .tool_names;
        let long_name = format!("{}_tool", "x".repeat(70));
        let wire = wire(&[
            json!({"choices": [{"index": 0, "delta": {"reasoning_content": "think"}}]}),
            json!({"choices": [{"index": 0, "delta": {"content": "Calling."}}]}),
            json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_a",
                "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}}]}),
            json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0,
                "function": {"arguments": "{\"city\":"}}]}}]}),
            json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0,
                "function": {"arguments": "\"Oslo\"}"}}]}}]}),
            json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "id": "call_b",
                "type": "function", "function": {"name": truncate_tool_name(&long_name),
                "arguments": "{}"}}]}}]}),
            json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}}),
        ]);
        assert_eq!(
            events(&wire, 4096, tool_names).await.unwrap(),
            [
                message_start(),
                start(
                    0,
                    json!({"type": "thinking", "thinking": "", "signature": ""})
                ),
                delta(0, json!({"type": "thinking_delta", "thinking": "think"})),
                stop(0),
                start(1, json!({"type": "text", "text": ""})),
                delta(1, json!({"type": "text_delta", "text": "Calling."})),
                stop(1),
                start(
                    2,
                    json!({"type": "tool_use", "id": "call_a", "name": "get_weather", "input": {}})
                ),
                delta(
                    2,
                    json!({"type": "input_json_delta", "partial_json": "{\"city\":"})
                ),
                delta(
                    2,
                    json!({"type": "input_json_delta", "partial_json": "\"Oslo\"}"})
                ),
                stop(2),
                start(
                    3,
                    json!({"type": "tool_use", "id": "call_b", "name": long_name, "input": {}})
                ),
                delta(3, json!({"type": "input_json_delta", "partial_json": "{}"})),
                stop(3),
                json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                    "usage": {"input_tokens": 5, "output_tokens": 9}}),
                json!({"type": "message_stop"}),
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn complete_tool_calls_sharing_an_index_stay_apart_by_id() {
        let call = |id: &str| {
            json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": id,
                "type": "function", "function": {"name": "f", "arguments": "{}"}}]}}]})
        };
        let events = events(&wire(&[call("a"), call("b")]), 4096, ToolNames::default())
            .await
            .unwrap();
        let starts: Vec<&Value> = events
            .iter()
            .filter(|event| event["type"] == "content_block_start")
            .map(|event| &event["content_block"]["id"])
            .collect();
        assert_eq!(starts, [&json!("a"), &json!("b")]);
    }

    #[rstest]
    #[tokio::test]
    async fn an_empty_stream_still_opens_and_closes_the_message() {
        assert_eq!(
            events("data: [DONE]\n\n", 4096, ToolNames::default())
                .await
                .unwrap(),
            [
                message_start(),
                json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                    "usage": {"input_tokens": 0, "output_tokens": 0}}),
                json!({"type": "message_stop"}),
            ]
        );
    }

    #[rstest]
    #[case::error_chunk(wire(&[json!({"error": {"message": "overloaded"}})]))]
    #[case::malformed_chunk("data: {not json}\n\n".to_owned())]
    #[tokio::test]
    async fn a_failed_stream_ends_with_an_error_and_no_message_stop(#[case] wire: String) {
        assert!(matches!(
            events(&wire, 4096, ToolNames::default()).await,
            Err(Error::InvalidResponse(_))
        ));
    }
}
