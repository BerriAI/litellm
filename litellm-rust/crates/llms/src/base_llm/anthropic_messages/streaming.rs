use bytes::Bytes;
use futures_util::{StreamExt, stream::BoxStream};
use litellm_framing::{frames, sse::SseCodec};

pub use crate::base_llm::base_model_iterator::ByteStream;
use crate::{Error, anthropic::messages::streaming_iterator::AnthropicMessagesStreamEvent};

pub type EventStream = BoxStream<'static, Result<AnthropicMessagesStreamEvent, Error>>;
pub type StreamDecoder = fn(ByteStream) -> EventStream;

pub fn anthropic_sse_event_stream(bytes: ByteStream) -> EventStream {
    Box::pin(frames(bytes, SseCodec::default()).map(|event| {
        let event = event
            .map_err(|error| Error::InvalidResponse(format!("stream framing failed: {error}")))?;
        serde_json::from_str(&event.data).map_err(|error| {
            Error::InvalidResponse(format!("Anthropic stream event is invalid: {error}"))
        })
    }))
}

pub fn encode_anthropic_sse(event: &AnthropicMessagesStreamEvent) -> Result<Bytes, Error> {
    let data = serde_json::to_value(event).map_err(|error| {
        Error::InvalidResponse(format!("Anthropic stream event is invalid: {error}"))
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

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt, TryStreamExt, stream};
    use serde_json::json;

    use super::*;
    use crate::anthropic::messages::streaming_iterator::{
        AnthropicContentBlockDelta, AnthropicStreamUsage,
    };

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
            vec![AnthropicMessagesStreamEvent::ContentBlockDelta {
                index: 0,
                delta: AnthropicContentBlockDelta::TextDelta {
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
            [AnthropicMessagesStreamEvent::ContentBlockDelta {
                delta: AnthropicContentBlockDelta::Citations { .. },
                ..
            }]
        ));
    }

    fn events() -> Vec<AnthropicMessagesStreamEvent> {
        vec![
            AnthropicMessagesStreamEvent::Ping,
            AnthropicMessagesStreamEvent::ContentBlockDelta {
                index: 1,
                delta: AnthropicContentBlockDelta::TextDelta { text: "hi".into() },
            },
            AnthropicMessagesStreamEvent::ContentBlockStop { index: 1 },
            AnthropicMessagesStreamEvent::MessageStop {
                usage: Some(AnthropicStreamUsage {
                    output_tokens: Some(7),
                    ..AnthropicStreamUsage::default()
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
            encode_anthropic_sse(&AnthropicMessagesStreamEvent::MessageStop { usage: None })
                .unwrap();

        assert_eq!(
            encoded,
            Bytes::from(format!(
                "event: message_stop\ndata: {}\n\n",
                json!({"type": "message_stop"})
            ))
        );
    }
}
