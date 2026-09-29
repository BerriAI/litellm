use bytes::Bytes;
use futures_util::{StreamExt, stream::BoxStream};
use litellm_framing::{frames, sse::SseCodec};
use litellm_llms_types::messages::streaming::MessagesStreamEvent;

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

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt, TryStreamExt, stream};
    use litellm_llms_types::messages::streaming::{MessagesContentBlockDelta, MessagesStreamUsage};
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
}
