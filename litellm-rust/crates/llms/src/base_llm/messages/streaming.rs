use bytes::Bytes;
use futures_util::{StreamExt, stream::BoxStream};
use litellm_framer::{frames, sse::SseCodec};
use litellm_llms_types::formats::messages::streaming::MessagesStreamEvent;

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
    use litellm_llms_types::formats::messages::MessagesUsage;
    use litellm_llms_types::formats::messages::streaming::MessagesContentBlockDelta;
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;

    const TEXT_DELTA: &str =
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#;

    #[fixture]
    fn unicode_delta() -> MessagesStreamEvent {
        MessagesStreamEvent::ContentBlockDelta {
            index: 2,
            delta: MessagesContentBlockDelta::TextDelta {
                text: "héllo 世界".into(),
                extra: Default::default(),
            },
            extra: Default::default(),
        }
    }

    #[rstest]
    #[case::one_byte(1)]
    #[case::inside_utf8(3)]
    #[case::inside_json(17)]
    #[case::whole_frame(1024)]
    #[tokio::test]
    async fn unicode_events_survive_byte_boundaries(
        unicode_delta: MessagesStreamEvent,
        #[case] chunk_size: usize,
    ) {
        let wire = concat!(
            "event: content_block_delta\r\n",
            "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"text_delta\",\"text\":\"héllo 世界\"}}\r\n\r\n"
        );
        let chunks: Vec<_> = wire
            .as_bytes()
            .chunks(chunk_size)
            .map(Bytes::copy_from_slice)
            .collect();
        let decoded = anthropic_sse_event_stream(stream::iter(chunks.into_iter().map(Ok)).boxed())
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(decoded, [unicode_delta]);
    }

    #[rstest]
    #[case::malformed_json(b"data: {\n\n")]
    #[case::missing_type(b"data: {\"index\":0}\n\n")]
    #[case::invalid_utf8(b"data: \xff\n\n")]
    #[case::unknown_event(b"data: {\"type\":\"future\"}\n\n")]
    #[tokio::test]
    async fn malformed_event_is_a_response_error(#[case] wire: &[u8]) {
        let result = anthropic_sse_event_stream(in_pieces(wire))
            .try_collect::<Vec<_>>()
            .await;
        assert!(
            matches!(result, Err(Error::InvalidResponse(_))),
            "{result:?}"
        );
    }

    #[rstest]
    #[case::multiline_data("data: {\"type\":\n data: ignored\ndata: \"ping\"}\n\n", json!({"type":"ping"}))]
    #[case::partial_tail("data: {\"type\":\"ping\"}\n\ndata: {\"type\":\"message_stop\"}", json!({"type":"ping"}))]
    #[tokio::test]
    async fn decoder_preserves_complete_events(
        #[case] wire: &str,
        #[case] expected: serde_json::Value,
    ) {
        let decoded = anthropic_sse_event_stream(in_pieces(wire.as_bytes()))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), json!([expected]));
    }

    #[rstest]
    #[tokio::test]
    async fn input_error_reaches_the_decoder_consumer() {
        let input: ByteStream = stream::iter([Err(std::io::Error::other("broken input"))]).boxed();
        let result = anthropic_sse_event_stream(input)
            .try_collect::<Vec<_>>()
            .await;
        assert!(
            matches!(result, Err(Error::InvalidResponse(_))),
            "{result:?}"
        );
    }

    #[rstest]
    #[case::message_start(json!({
        "type":"message_start","message":{
            "id":"msg_1","type":"message","role":"assistant","model":"claude","content":[],
            "stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0},
            "safeguard_results":[{"type":"dangerous_tool_use","status":{"type":"available","tool_uses":{"toolu_01":{"type":"evaluated","outcome":"not_flagged"}}}}]
        }
    }))]
    #[case::message_delta(json!({
        "type":"message_delta","delta":{
            "stop_reason":"end_turn","stop_sequence":null,
            "safeguard_results":[{"type":"dangerous_tool_use","status":{"type":"available","tool_uses":{"toolu_01":{"type":"evaluated","outcome":"not_flagged"}}}}]
        },"usage":{"output_tokens":1}
    }))]
    #[tokio::test]
    async fn native_messages_streaming_keeps_safeguard_results(#[case] wire: serde_json::Value) {
        let input = format!(
            "event: {}\ndata: {wire}\n\n",
            wire["type"].as_str().unwrap()
        );
        let decoded = anthropic_sse_event_stream(in_pieces(input.as_bytes()))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), json!([wire]));
    }

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
                    extra: Default::default(),
                },
                extra: Default::default(),
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
            MessagesStreamEvent::Ping {
                extra: Default::default(),
            },
            MessagesStreamEvent::ContentBlockDelta {
                index: 1,
                delta: MessagesContentBlockDelta::TextDelta {
                    text: "hi".into(),
                    extra: Default::default(),
                },
                extra: Default::default(),
            },
            MessagesStreamEvent::ContentBlockStop {
                index: 1,
                extra: Default::default(),
            },
            MessagesStreamEvent::MessageStop {
                usage: Some(Box::new(MessagesUsage {
                    output_tokens: Some(litellm_llms_types::serde_compat::Nullable::Value(7)),
                    ..MessagesUsage::default()
                })),
                extra: Default::default(),
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
        let encoded = encode_anthropic_sse(&MessagesStreamEvent::MessageStop {
            usage: None,
            extra: Default::default(),
        })
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
