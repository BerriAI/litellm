use std::collections::BTreeMap;

use bytes::Bytes;
use futures_util::{StreamExt, stream::BoxStream};
use litellm_framer::{
    frames,
    sse::{SseCodec, decode_all},
};
use litellm_llms_types::formats::messages::{
    MessagesResponse,
    streaming::{
        MessagesContentBlock, MessagesContentBlockDelta, MessagesDelta, MessagesStreamEvent,
        MessagesStreamMessage, MessagesStreamUsage,
    },
};
use serde_json::{Map, Value, json};

use crate::Error;
pub use crate::base_llm::base_model_iterator::ByteStream;

pub type EventStream = BoxStream<'static, Result<MessagesStreamEvent, Error>>;
pub type StreamDecoder = fn(ByteStream) -> EventStream;

pub fn anthropic_sse_event_stream(bytes: ByteStream) -> EventStream {
    Box::pin(frames(bytes, SseCodec::default()).map(|event| {
        let event = event.map_err(|error| {
            Error::InvalidResponse(crate::ErrorDetail::failed("stream framing", error))
        })?;
        serde_json::from_str(&event.data).map_err(|error| {
            Error::InvalidResponse(crate::ErrorDetail::invalid("Anthropic stream event", error))
        })
    }))
}

pub fn encode_anthropic_sse(event: &MessagesStreamEvent) -> Result<Bytes, Error> {
    let data = serde_json::to_value(event).map_err(|error| {
        Error::InvalidResponse(crate::ErrorDetail::invalid("Anthropic stream event", error))
    })?;
    let name = data
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            Error::InvalidResponse(
                "Anthropic stream event is invalid: stream event has no type".into(),
            )
        })?;
    Ok(Bytes::from(format!("event: {name}\ndata: {data}\n\n")))
}

fn invalid(error: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::InvalidResponse(crate::ErrorDetail::invalid("Anthropic stream event", error))
}

/// The typed events in a complete Anthropic SSE wire, however the bytes were chunked.
pub fn decode_anthropic_sse(wire: &[u8]) -> Result<Vec<MessagesStreamEvent>, Error> {
    decode_all(wire)
        .map_err(|error| {
            Error::InvalidResponse(crate::ErrorDetail::failed("stream framing", error))
        })?
        .iter()
        .map(|event| serde_json::from_str(&event.data).map_err(invalid))
        .collect()
}

/// A content block as the stream builds it: the typed start plus whatever the deltas add.
struct OpenBlock {
    block: Map<String, Value>,
    partial_json: String,
}

impl OpenBlock {
    fn start(block: MessagesContentBlock) -> Result<Self, Error> {
        let mut value = match serde_json::to_value(&block).map_err(invalid)? {
            Value::Object(map) => map,
            _ => Map::new(),
        };
        match block.block_type.as_str() {
            "text" => {
                value.entry("text").or_insert_with(|| json!(""));
            }
            "thinking" => {
                value.entry("thinking").or_insert_with(|| json!(""));
                value.entry("signature").or_insert_with(|| json!(""));
            }
            "tool_use" => {
                value.insert("input".into(), json!({}));
            }
            _ => {}
        }
        Ok(Self {
            block: value,
            partial_json: String::new(),
        })
    }

    fn append(&mut self, field: &str, piece: &str) {
        let current = self.block.entry(field).or_insert_with(|| json!(""));
        let joined = format!("{}{piece}", current.as_str().unwrap_or_default());
        *current = Value::String(joined);
    }

    fn apply(&mut self, delta: MessagesContentBlockDelta) {
        match delta {
            MessagesContentBlockDelta::TextDelta { text } => self.append("text", &text),
            MessagesContentBlockDelta::InputJsonDelta { partial_json } => {
                self.partial_json.push_str(&partial_json)
            }
            MessagesContentBlockDelta::ThinkingDelta { thinking } => {
                self.append("thinking", &thinking)
            }
            MessagesContentBlockDelta::SignatureDelta { signature } => {
                self.block
                    .insert("signature".into(), Value::String(signature));
            }
            MessagesContentBlockDelta::Citations { citation } => {
                let citations = self
                    .block
                    .entry("citations")
                    .or_insert_with(|| Value::Array(Vec::new()));
                if let Value::Array(citations) = citations {
                    citations.push(citation);
                }
            }
            MessagesContentBlockDelta::CompactionDelta { content } => {
                self.append("content", &content)
            }
        }
    }

    fn close(mut self) -> Result<Value, Error> {
        if !self.partial_json.is_empty() {
            let input: Value = serde_json::from_str(&self.partial_json).map_err(invalid)?;
            self.block.insert("input".into(), input);
        }
        Ok(Value::Object(self.block))
    }
}

fn merge_usage(into: &mut MessagesStreamUsage, usage: MessagesStreamUsage) {
    into.input_tokens = usage.input_tokens.or(into.input_tokens);
    into.output_tokens = usage.output_tokens.or(into.output_tokens);
    into.cache_creation_input_tokens = usage
        .cache_creation_input_tokens
        .or(into.cache_creation_input_tokens);
    into.cache_read_input_tokens = usage
        .cache_read_input_tokens
        .or(into.cache_read_input_tokens);
    into.server_tool_use = usage.server_tool_use.or(into.server_tool_use.take());
    into.extra.extend(usage.extra);
}

/// The complete message a finished Anthropic event stream described, or `None` when the
/// stream never carried a `message_start`.
pub fn assemble_message(
    events: impl IntoIterator<Item = MessagesStreamEvent>,
) -> Result<Option<MessagesResponse>, Error> {
    let mut message: Option<MessagesStreamMessage> = None;
    let mut blocks: BTreeMap<u64, OpenBlock> = BTreeMap::new();
    let mut closed: BTreeMap<u64, Value> = BTreeMap::new();
    let mut delta = MessagesDelta::default();
    let mut usage = MessagesStreamUsage::default();
    for event in events {
        match event {
            MessagesStreamEvent::MessageStart { message: started } => {
                usage = started.usage.clone();
                message = Some(started);
            }
            MessagesStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                blocks.insert(index, OpenBlock::start(content_block)?);
            }
            MessagesStreamEvent::ContentBlockDelta {
                index,
                delta: block_delta,
            } => {
                if let Some(block) = blocks.get_mut(&index) {
                    block.apply(block_delta);
                }
            }
            MessagesStreamEvent::ContentBlockStop { index } => {
                if let Some(block) = blocks.remove(&index) {
                    closed.insert(index, block.close()?);
                }
            }
            MessagesStreamEvent::MessageDelta {
                delta: message_delta,
                usage: delta_usage,
                ..
            } => {
                delta = message_delta;
                if let Some(delta_usage) = delta_usage {
                    merge_usage(&mut usage, delta_usage);
                }
            }
            MessagesStreamEvent::MessageStop { usage: stop_usage } => {
                if let Some(stop_usage) = stop_usage {
                    merge_usage(&mut usage, stop_usage);
                }
            }
            MessagesStreamEvent::Ping | MessagesStreamEvent::Error { .. } => {}
        }
    }
    let Some(message) = message else {
        return Ok(None);
    };
    for (index, block) in blocks {
        closed.insert(index, block.close()?);
    }
    Ok(Some(MessagesResponse {
        id: message.id,
        message_type: message.message_type,
        role: message.role,
        model: message.model,
        content: closed.into_values().collect(),
        stop_reason: delta.stop_reason,
        stop_sequence: delta.stop_sequence,
        usage: Some(serde_json::to_value(usage).map_err(invalid)?),
        container: delta.container,
        extra: message.extra,
    }))
}

fn block_events(index: u64, block: &Value) -> Vec<MessagesStreamEvent> {
    let object = block.as_object().cloned().unwrap_or_default();
    let field = |name: &str| object.get(name).cloned();
    let text = |name: &str| field(name).and_then(|value| value.as_str().map(str::to_owned));
    let start = |content_block: Value| MessagesStreamEvent::ContentBlockStart {
        index,
        content_block: serde_json::from_value(content_block)
            .unwrap_or_else(|_| serde_json::from_value(json!({"type": "text"})).unwrap()),
    };
    let delta =
        |delta: MessagesContentBlockDelta| MessagesStreamEvent::ContentBlockDelta { index, delta };
    let mut events = match text("type").as_deref() {
        Some("text") => vec![
            start(json!({"type": "text", "text": ""})),
            delta(MessagesContentBlockDelta::TextDelta {
                text: text("text").unwrap_or_default(),
            }),
        ],
        Some("thinking") => {
            let mut events = vec![start(
                json!({"type": "thinking", "thinking": "", "signature": ""}),
            )];
            if let Some(thinking) = text("thinking").filter(|value| !value.is_empty()) {
                events.push(delta(MessagesContentBlockDelta::ThinkingDelta { thinking }));
            }
            if let Some(signature) = text("signature").filter(|value| !value.is_empty()) {
                events.push(delta(MessagesContentBlockDelta::SignatureDelta {
                    signature,
                }));
            }
            events
        }
        Some("tool_use") => vec![
            start(
                json!({"type": "tool_use", "id": field("id"), "name": field("name"), "input": {}}),
            ),
            delta(MessagesContentBlockDelta::InputJsonDelta {
                partial_json: field("input").unwrap_or_else(|| json!({})).to_string(),
            }),
        ],
        _ => vec![start(block.clone())],
    };
    events.push(MessagesStreamEvent::ContentBlockStop { index });
    events
}

/// The event stream a client would have seen for a complete message, the way the Python
/// fake stream replays one: a start with empty content, one block at a time, then the
/// stop reason and usage.
pub fn disassemble_message(response: &MessagesResponse) -> Result<Vec<MessagesStreamEvent>, Error> {
    let usage: MessagesStreamUsage = match &response.usage {
        Some(usage) => serde_json::from_value(usage.clone()).map_err(invalid)?,
        None => MessagesStreamUsage::default(),
    };
    let start = MessagesStreamEvent::MessageStart {
        message: MessagesStreamMessage {
            id: response.id.clone(),
            message_type: response.message_type.clone(),
            role: response.role.clone(),
            model: response.model.clone(),
            content: Vec::new(),
            stop_reason: None,
            stop_sequence: None,
            usage: MessagesStreamUsage {
                input_tokens: Some(usage.input_tokens.unwrap_or(0)),
                output_tokens: Some(0),
                ..MessagesStreamUsage::default()
            },
            extra: response.extra.clone(),
        },
    };
    let blocks = response
        .content
        .iter()
        .enumerate()
        .flat_map(|(index, block)| block_events(index as u64, block));
    let delta = MessagesStreamEvent::MessageDelta {
        delta: MessagesDelta {
            stop_reason: response.stop_reason.clone(),
            stop_sequence: response.stop_sequence.clone(),
            container: response.container.clone(),
            ..MessagesDelta::default()
        },
        usage: Some(MessagesStreamUsage {
            output_tokens: Some(usage.output_tokens.unwrap_or(0)),
            input_tokens: usage.input_tokens,
            cache_creation_input_tokens: usage.cache_creation_input_tokens,
            cache_read_input_tokens: usage.cache_read_input_tokens,
            ..MessagesStreamUsage::default()
        }),
        context_management: None,
    };
    let stop = MessagesStreamEvent::MessageStop { usage: Some(usage) };
    Ok(std::iter::once(start)
        .chain(blocks)
        .chain([delta, stop])
        .collect())
}

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt, TryStreamExt, stream};
    use litellm_llms_types::formats::messages::streaming::{
        MessagesContentBlockDelta, MessagesStreamUsage,
    };
    use serde_json::json;

    use super::*;

    const TEXT_DELTA: &str =
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#;

    fn in_pieces(wire: &[u8]) -> ByteStream {
        let pieces: Vec<Bytes> = wire.chunks(3).map(Bytes::copy_from_slice).collect();
        stream::iter(pieces.into_iter().map(Ok)).boxed()
    }

    #[tokio::test]
    async fn sse_frames_split_anywhere_decode_into_typed_events() {
        let wire = format!("event: content_block_delta\ndata: {TEXT_DELTA}\n\n");
        let events = anthropic_sse_event_stream(in_pieces(wire.as_bytes()))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(
            events,
            vec![MessagesStreamEvent::ContentBlockDelta {
                index: 0,
                delta: MessagesContentBlockDelta::TextDelta {
                    text: "hello".into(),
                },
            }]
        );
    }

    #[tokio::test]
    async fn decodes_citations_delta_events() {
        let wire = concat!(
            "event: content_block_delta\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"char_location"}}}"#,
            "\n\n",
        );
        let events = anthropic_sse_event_stream(in_pieces(wire.as_bytes()))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert!(matches!(
            events.as_slice(),
            [MessagesStreamEvent::ContentBlockDelta {
                delta: MessagesContentBlockDelta::Citations { .. },
                ..
            }]
        ));
    }

    fn events() -> Vec<MessagesStreamEvent> {
        vec![
            MessagesStreamEvent::Ping,
            MessagesStreamEvent::ContentBlockDelta {
                index: 1,
                delta: MessagesContentBlockDelta::TextDelta { text: "hi".into() },
            },
            MessagesStreamEvent::ContentBlockStop { index: 1 },
            MessagesStreamEvent::MessageStop {
                usage: Some(MessagesStreamUsage {
                    output_tokens: Some(7),
                    ..MessagesStreamUsage::default()
                }),
            },
        ]
    }

    #[tokio::test]
    async fn encoded_events_decode_back_to_themselves() {
        let wire = events()
            .iter()
            .map(encode_anthropic_sse)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let decoded = anthropic_sse_event_stream(stream::iter(wire.into_iter().map(Ok)).boxed())
            .try_collect::<Vec<_>>()
            .await
            .unwrap();

        assert_eq!(decoded, events());
    }

    #[test]
    fn an_event_is_named_by_its_type() {
        let encoded =
            encode_anthropic_sse(&MessagesStreamEvent::MessageStop { usage: None }).unwrap();

        assert_eq!(
            encoded,
            Bytes::from(format!(
                "event: message_stop\ndata: {}\n\n",
                json!({"type": "message_stop"})
            ))
        );
    }

    mod assembly {
        use litellm_llms_types::formats::messages::MessagesResponse;
        use rstest::rstest;
        use serde_json::{Value, json};

        use super::super::{
            assemble_message, decode_anthropic_sse, disassemble_message, encode_anthropic_sse,
        };

        fn message(content: Vec<Value>, usage: Value) -> MessagesResponse {
            MessagesResponse {
                id: "msg_1".into(),
                message_type: "message".into(),
                role: "assistant".into(),
                model: "claude".into(),
                content,
                stop_reason: Some("end_turn".into()),
                stop_sequence: None,
                usage: Some(usage),
                container: None,
                extra: serde_json::Map::new(),
            }
        }

        /// Replaying a complete message as events and assembling those events gives the
        /// message back, for every block shape the Python fake stream knows.
        #[rstest]
        #[case::text(json!({"type": "text", "text": "hello"}))]
        #[case::tool_use(json!({"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}))]
        #[case::thinking(json!({"type": "thinking", "thinking": "hmm", "signature": "sig"}))]
        #[case::redacted_thinking(json!({"type": "redacted_thinking", "data": "opaque"}))]
        #[case::passthrough(json!({"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "x"}}))]
        fn a_message_survives_the_event_round_trip(#[case] block: Value) {
            let original = message(
                vec![block, json!({"type": "text", "text": "and then"})],
                json!({"input_tokens": 5, "output_tokens": 9, "cache_read_input_tokens": 2}),
            );
            let events = disassemble_message(&original).unwrap();
            let wire: Vec<u8> = events
                .iter()
                .map(|event| encode_anthropic_sse(event).unwrap())
                .flat_map(|bytes| bytes.to_vec())
                .collect();
            let decoded = decode_anthropic_sse(&wire).unwrap();
            let rebuilt = assemble_message(decoded).unwrap().unwrap();
            assert_eq!(
                serde_json::to_value(&rebuilt).unwrap(),
                serde_json::to_value(&original).unwrap()
            );
        }

        #[rstest]
        fn the_replay_starts_empty_and_ends_with_the_stop_reason_and_usage() {
            let events = disassemble_message(&message(
                vec![json!({"type": "text", "text": "hi"})],
                json!({"input_tokens": 5, "output_tokens": 9}),
            ))
            .unwrap();
            let shapes: Vec<Value> = events
                .iter()
                .map(|event| serde_json::to_value(event).unwrap())
                .collect();
            assert_eq!(shapes[0]["type"], "message_start");
            assert_eq!(shapes[0]["message"]["content"], json!([]));
            assert_eq!(
                shapes[0]["message"]["usage"],
                json!({"input_tokens": 5, "output_tokens": 0})
            );
            assert_eq!(shapes[shapes.len() - 2]["type"], "message_delta");
            assert_eq!(shapes[shapes.len() - 2]["delta"]["stop_reason"], "end_turn");
            assert_eq!(shapes[shapes.len() - 2]["usage"]["output_tokens"], 9);
            assert_eq!(shapes[shapes.len() - 1]["type"], "message_stop");
        }

        /// A provider stream splits deltas freely and interleaves pings; assembly joins the
        /// pieces and parses the tool input once the block closes.
        #[rstest]
        fn provider_events_assemble_across_split_deltas_and_pings() {
            let wire = concat!(
                "event: message_start
data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_p\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":7}}}\n\n",
                "event: ping\ndata: {\"type\":\"ping\"}\n\n",
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\": \"}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"Paris\\\"}\"}}\n\n",
                "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":12}}\n\n",
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
            );
            let message = assemble_message(decode_anthropic_sse(wire.as_bytes()).unwrap())
                .unwrap()
                .unwrap();
            assert_eq!(message.id, "msg_p");
            assert_eq!(message.stop_reason.as_deref(), Some("tool_use"));
            assert_eq!(
                message.content,
                vec![
                    json!({"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}})
                ]
            );
            assert_eq!(
                message.usage,
                Some(json!({"input_tokens": 7, "output_tokens": 12}))
            );
        }

        #[rstest]
        fn a_stream_without_a_message_start_assembles_to_nothing() {
            let wire = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
            assert!(
                assemble_message(decode_anthropic_sse(wire.as_bytes()).unwrap())
                    .unwrap()
                    .is_none()
            );
        }

        #[rstest]
        fn malformed_tool_input_is_a_response_error_not_a_guess() {
            let wire = concat!(
                "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"c\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{}}}\n\n",
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t\",\"name\":\"f\",\"input\":{}}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{not json\"}}\n\n",
                "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            );
            assert!(assemble_message(decode_anthropic_sse(wire.as_bytes()).unwrap()).is_err());
        }
    }
}
