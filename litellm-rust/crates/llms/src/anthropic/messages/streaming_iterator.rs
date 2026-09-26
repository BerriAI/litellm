use base64::Engine;
use bytes::Buf;
use futures_util::{Stream, StreamExt};
use litellm_framing::{
    aws_event_stream::{AwsEventStreamCodec, Message},
    frames,
    sse::{SseCodec, SseEvent},
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("stream framing failed: {0}")]
    StreamFraming(String),
    #[error("Anthropic stream event is invalid: {0}")]
    InvalidStreamEvent(String),
    #[error("Bedrock event payload is invalid: {0}")]
    InvalidBedrockPayload(String),
    #[error("Bedrock event payload has invalid base64: {0}")]
    InvalidBedrockBase64(String),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AnthropicStreamUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnthropicStreamMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<Value>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicStreamUsage,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlockDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    #[serde(rename = "citations_delta")]
    Citations {
        citation: Value,
    },
    ThinkingDelta {
        thinking: String,
    },
    SignatureDelta {
        signature: String,
    },
    CompactionDelta {
        content: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnthropicContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub caller: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AnthropicMessageDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_details: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnthropicStreamError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicMessagesStreamEvent {
    MessageStart {
        message: AnthropicStreamMessage,
    },
    ContentBlockStart {
        index: u64,
        content_block: AnthropicContentBlock,
    },
    ContentBlockDelta {
        index: u64,
        delta: AnthropicContentBlockDelta,
    },
    ContentBlockStop {
        index: u64,
    },
    MessageDelta {
        delta: AnthropicMessageDelta,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<AnthropicStreamUsage>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context_management: Option<Value>,
    },
    MessageStop,
    Ping,
    Error {
        error: AnthropicStreamError,
    },
}

#[derive(Deserialize)]
struct BedrockChunkPayload {
    bytes: String,
}

pub fn decode_anthropic_sse_frame(event: SseEvent) -> Result<AnthropicMessagesStreamEvent, Error> {
    serde_json::from_str(&event.data).map_err(|error| Error::InvalidStreamEvent(error.to_string()))
}

pub fn decode_bedrock_anthropic_frame(
    message: Message,
) -> Result<AnthropicMessagesStreamEvent, Error> {
    let payload: BedrockChunkPayload = serde_json::from_slice(message.payload())
        .map_err(|error| Error::InvalidBedrockPayload(error.to_string()))?;
    let event = base64::engine::general_purpose::STANDARD
        .decode(payload.bytes)
        .map_err(|error| Error::InvalidBedrockBase64(error.to_string()))?;
    serde_json::from_slice(&event).map_err(|error| Error::InvalidStreamEvent(error.to_string()))
}

pub fn direct_anthropic_event_stream<S, B, E>(
    input: S,
) -> impl Stream<Item = Result<AnthropicMessagesStreamEvent, Error>> + Send
where
    S: Stream<Item = Result<B, E>> + Send,
    B: Buf + Send,
    E: std::error::Error + Send + Sync + 'static,
{
    frames(input, SseCodec::default()).map(|event| {
        decode_anthropic_sse_frame(event.map_err(|error| Error::StreamFraming(error.to_string()))?)
    })
}

pub fn bedrock_anthropic_event_stream<S, B, E>(
    input: S,
) -> impl Stream<Item = Result<AnthropicMessagesStreamEvent, Error>> + Send
where
    S: Stream<Item = Result<B, E>> + Send,
    B: Buf + Send,
    E: std::error::Error + Send + Sync + 'static,
{
    frames(input, AwsEventStreamCodec).map(|message| {
        decode_bedrock_anthropic_frame(
            message.map_err(|error| Error::StreamFraming(error.to_string()))?,
        )
    })
}

#[cfg(test)]
mod tests {
    use std::io;

    use aws_smithy_eventstream::frame::write_message_to;
    use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
    use base64::engine::general_purpose::STANDARD;
    use bytes::Bytes;
    use futures_util::TryStreamExt;

    use super::*;

    const TEXT_DELTA: &str =
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#;

    #[tokio::test]
    async fn direct_anthropic_sse_frames_into_typed_events() {
        let wire = format!("event: content_block_delta\ndata: {TEXT_DELTA}\n\n");
        let events = direct_anthropic_event_stream(futures_util::stream::iter(
            wire.as_bytes().chunks(3).map(Ok::<_, io::Error>),
        ))
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        assert_eq!(
            events,
            vec![AnthropicMessagesStreamEvent::ContentBlockDelta {
                index: 0,
                delta: AnthropicContentBlockDelta::TextDelta {
                    text: "hello".into(),
                },
            }]
        );
    }

    #[test]
    fn decodes_citations_delta_events() {
        let event = decode_anthropic_sse_frame(SseEvent {
            event: Some("content_block_delta".into()),
            data: r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"char_location"}}}"#
                .into(),
            id: None,
            retry: None,
        })
        .unwrap();

        assert!(matches!(
            event,
            AnthropicMessagesStreamEvent::ContentBlockDelta {
                delta: AnthropicContentBlockDelta::Citations { .. },
                ..
            }
        ));
    }

    #[tokio::test]
    async fn bedrock_aws_frames_into_the_same_typed_events() {
        let payload = serde_json::json!({"bytes": STANDARD.encode(TEXT_DELTA)});
        let message = Message::new(Bytes::from(serde_json::to_vec(&payload).unwrap())).add_header(
            Header::new(":event-type", HeaderValue::String("chunk".into())),
        );
        let mut wire = Vec::new();
        write_message_to(&message, &mut wire).unwrap();

        let events = bedrock_anthropic_event_stream(futures_util::stream::iter(
            wire.chunks(3).map(Ok::<_, io::Error>),
        ))
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        assert_eq!(
            events,
            vec![AnthropicMessagesStreamEvent::ContentBlockDelta {
                index: 0,
                delta: AnthropicContentBlockDelta::TextDelta {
                    text: "hello".into(),
                },
            }]
        );
    }
}
