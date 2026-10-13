use bytes::Bytes;
use litellm_llms_types::formats::messages::streaming::MessagesStreamEvent;

pub use crate::base_llm::base_model_iterator::ByteStream;
use crate::{Error, base_llm::sse};

pub type EventStream = sse::EventStream<MessagesStreamEvent>;
pub type StreamDecoder = fn(ByteStream) -> EventStream;

fn no_control_frames(_: &litellm_framer::sse::SseEvent) -> bool {
    false
}

pub fn event_stream(bytes: ByteStream) -> EventStream {
    sse::event_stream(bytes, no_control_frames)
}

pub fn decode(wire: &[u8]) -> Result<Vec<MessagesStreamEvent>, Error> {
    sse::decode(wire, no_control_frames)
}

/// Anthropic names every frame after the event's `type`, and LiteLLM's proxy does the same
/// when it re-emits translated events.
pub fn encode(event: &MessagesStreamEvent) -> Result<Bytes, Error> {
    let data = serde_json::to_string(event).map_err(sse::invalid)?;
    Ok(Bytes::from(format!(
        "event: {}\ndata: {data}\n\n",
        event.as_ref()
    )))
}

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt, TryStreamExt, stream};
    use litellm_llms_types::formats::messages::streaming::{
        MessagesContentBlockDelta, MessagesStreamUsage,
    };
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    const TEXT_DELTA: &str =
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#;

    fn in_pieces(wire: &[u8], size: usize) -> ByteStream {
        let pieces: Vec<Bytes> = wire.chunks(size).map(Bytes::copy_from_slice).collect();
        stream::iter(pieces.into_iter().map(Ok)).boxed()
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

    fn wire(newline: &str) -> String {
        format!(
            "event: content_block_delta{newline}data: {TEXT_DELTA}{newline}{newline}: keepalive{newline}{newline}event: message_stop{newline}data: {{\"type\":\"message_stop\"}}{newline}{newline}"
        )
    }

    #[rstest]
    #[case::lf("\n")]
    #[case::crlf("\r\n")]
    #[case::cr("\r")]
    #[tokio::test]
    async fn frames_split_anywhere_decode_into_typed_events(
        #[case] newline: &str,
        #[values(1, 3, 4096)] size: usize,
    ) {
        let wire = wire(newline);
        let expected = vec![
            MessagesStreamEvent::ContentBlockDelta {
                index: 0,
                delta: MessagesContentBlockDelta::TextDelta {
                    text: "hello".into(),
                },
            },
            MessagesStreamEvent::MessageStop { usage: None },
        ];
        let streamed = event_stream(in_pieces(wire.as_bytes(), size))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(streamed, expected);
        assert_eq!(decode(wire.as_bytes()).unwrap(), expected);
    }

    #[tokio::test]
    async fn decodes_citations_delta_events() {
        let wire = concat!(
            "event: content_block_delta\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"char_location"}}}"#,
            "\n\n",
        );
        let events = event_stream(in_pieces(wire.as_bytes(), 3))
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

    #[tokio::test]
    async fn encoded_events_decode_back_to_themselves() {
        let wire: Vec<u8> = events()
            .iter()
            .map(encode)
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        let streamed = event_stream(in_pieces(&wire, 3))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(streamed, events());
        assert_eq!(decode(&wire).unwrap(), events());
    }

    #[test]
    fn an_event_is_named_by_its_type() {
        let encoded = encode(&MessagesStreamEvent::MessageStop { usage: None }).unwrap();
        assert_eq!(
            encoded,
            Bytes::from(format!(
                "event: message_stop\ndata: {}\n\n",
                json!({"type": "message_stop"})
            ))
        );
    }

    #[rstest]
    #[case::json(b"data: {\n\n")]
    #[case::malformed_tool(b"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":3}}\n\n")]
    #[case::utf8(b"data: \xff\n\n")]
    #[case::done_is_not_an_anthropic_frame(b"data: [DONE]\n\n")]
    #[tokio::test]
    async fn malformed_streams_are_invalid_responses(#[case] wire: &[u8]) {
        assert!(matches!(decode(wire), Err(Error::InvalidResponse(_))));
        let streamed = event_stream(in_pieces(wire, 1)).collect::<Vec<_>>().await;
        assert!(matches!(
            streamed.last(),
            Some(Err(Error::InvalidResponse(_))) | None
        ));
    }

    #[rstest]
    #[tokio::test]
    async fn unknown_event_types_reach_the_client_as_sent(#[values(1, 3, 4096)] size: usize) {
        let frame = "event: future_event\ndata: {\"type\":\"future_event\",\"index\":2,\"payload\":{\"keep\":[1,null]}}\n\n";
        let streamed = event_stream(in_pieces(frame.as_bytes(), size))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(streamed, decode(frame.as_bytes()).unwrap());
        let [MessagesStreamEvent::Other(event)] = streamed.as_slice() else {
            panic!("expected one unknown event, got {streamed:?}");
        };
        assert_eq!(event.event_type, "future_event");
        assert_eq!(encode(&streamed[0]).unwrap(), Bytes::from(frame));
    }

    #[rstest]
    fn decoding_keeps_tool_deltas_and_block_extensions() {
        let wire = concat!(
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t\",\"name\":\"lookup\",\"input\":{},\"future\":null}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"x\\\":\"}}\n\n",
        );
        let decoded = decode(wire.as_bytes()).unwrap();
        let [
            MessagesStreamEvent::ContentBlockStart { content_block, .. },
            MessagesStreamEvent::ContentBlockDelta {
                delta: MessagesContentBlockDelta::InputJsonDelta { partial_json },
                ..
            },
        ] = decoded.as_slice()
        else {
            panic!("expected tool events");
        };
        assert_eq!(
            content_block.extra.get("future"),
            Some(&serde_json::Value::Null)
        );
        assert_eq!(partial_json, "{\"x\":");
    }

    #[rstest]
    #[tokio::test]
    async fn a_body_error_ends_the_stream_after_delivered_events() {
        let mut events = event_stream(
            stream::iter([
                Ok(Bytes::from_static(b"data: {\"type\":\"ping\"}\n\n")),
                Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionReset,
                    "reset",
                )),
            ])
            .boxed(),
        );
        assert_eq!(
            events.next().await.unwrap().unwrap(),
            MessagesStreamEvent::Ping
        );
        assert!(matches!(
            events.next().await,
            Some(Err(Error::InvalidResponse(_)))
        ));
        assert!(events.next().await.is_none());
    }
}
