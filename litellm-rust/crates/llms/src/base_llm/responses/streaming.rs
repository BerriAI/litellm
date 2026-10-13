use bytes::Bytes;
use litellm_llms_types::formats::responses::streaming::ResponsesStreamEvent;

pub use crate::base_llm::base_model_iterator::ByteStream;
use crate::{
    Error,
    base_llm::sse::{self, is_done},
};

pub type EventStream = sse::EventStream<ResponsesStreamEvent>;

pub fn event_stream(bytes: ByteStream) -> EventStream {
    sse::event_stream(bytes, is_done)
}

pub fn decode(wire: &[u8]) -> Result<Vec<ResponsesStreamEvent>, Error> {
    sse::decode(wire, is_done)
}

/// A bare `data:` frame. Python's proxy serializes every Responses event this way and never
/// writes an `event:` line, so clients see one shape whichever runtime served them.
pub fn encode(event: &ResponsesStreamEvent) -> Result<Bytes, Error> {
    sse::encode_data(event)
}

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt, TryStreamExt, stream};
    use litellm_llms_types::formats::responses::streaming::ResponsesEventType;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn in_pieces(wire: &[u8], size: usize) -> ByteStream {
        let pieces: Vec<Bytes> = wire.chunks(size).map(Bytes::copy_from_slice).collect();
        stream::iter(pieces.into_iter().map(Ok)).boxed()
    }

    #[rstest]
    #[case::created(json!({"type":"response.created","response":{"id":"r","output":[]}}))]
    #[case::text(json!({"type":"response.output_text.delta","delta":"hello\nworld","sequence_number":3}))]
    #[case::tool(json!({"type":"response.function_call_arguments.delta","delta":"{\"x\":","item_id":"fc_1"}))]
    #[case::unknown(json!({"type":"future.event","payload":[1,null],"future":{"nested":true}}))]
    #[case::completed(json!({"type":"response.completed","response":{"id":"r","usage":{"input_tokens":3}}}))]
    #[case::failed(json!({"type":"response.failed","response":{"error":{"code":"failure"}}}))]
    #[case::incomplete(json!({"type":"response.incomplete","response":{"incomplete_details":{"reason":"limit"}}}))]
    #[case::error(json!({"type":"error","error":{"message":"boom"}}))]
    fn encoded_events_are_bare_data_frames_that_decode_back(#[case] wire: Value) {
        let event: ResponsesStreamEvent = serde_json::from_value(wire.clone()).unwrap();
        let encoded = encode(&event).unwrap();
        assert!(encoded.starts_with(b"data: "));
        assert!(encoded.ends_with(b"\n\n"));
        assert_eq!(
            serde_json::from_slice::<Value>(&encoded[b"data: ".len()..encoded.len() - 2]).unwrap(),
            wire
        );
        assert_eq!(decode(&encoded).unwrap(), vec![event]);
    }

    #[rstest]
    #[case::lf("\n")]
    #[case::crlf("\r\n")]
    #[case::cr("\r")]
    #[tokio::test]
    async fn streams_split_anywhere_skip_comments_and_done(
        #[case] newline: &str,
        #[values(1, 5, 4096)] size: usize,
    ) {
        let wire = format!(
            "\u{feff}event: response.created{newline}data: {{\"type\":\"response.created\"}}{newline}{newline}: keepalive{newline}{newline}event: future.event{newline}data: {{\"type\":\"future.event\",\"payload\":null}}{newline}{newline}data: [DONE]{newline}{newline}"
        );
        let streamed = event_stream(in_pieces(wire.as_bytes(), size))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(streamed, decode(wire.as_bytes()).unwrap());
        assert_eq!(streamed.len(), 2);
        assert_eq!(streamed[0].event_type, ResponsesEventType::ResponseCreated);
        assert_eq!(
            streamed[1].event_type,
            ResponsesEventType::Other("future.event".into())
        );
        assert_eq!(streamed[1].data.get("payload"), Some(&Value::Null));
    }

    #[rstest]
    #[case::spaced(b"data: [DONE]\n\n")]
    #[case::unspaced(b"data:[DONE]\n\n")]
    #[case::trailing_space(b"data: [DONE] \n\n")]
    fn every_done_spelling_closes_the_stream_without_an_event(#[case] wire: &[u8]) {
        assert_eq!(decode(wire).unwrap(), vec![]);
    }

    #[rstest]
    #[case::invalid_json(b"data: {\n\n")]
    #[case::missing_type(b"data: {}\n\n")]
    #[case::numeric_type(b"data: {\"type\":3}\n\n")]
    #[case::invalid_utf8(b"data: \xff\n\n")]
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
    async fn a_body_error_ends_the_stream_after_delivered_events() {
        let mut events = event_stream(
            stream::iter([
                Ok(Bytes::from_static(
                    b"data: {\"type\":\"response.created\"}\n\n",
                )),
                Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionReset,
                    "reset",
                )),
            ])
            .boxed(),
        );
        assert_eq!(
            events.next().await.unwrap().unwrap().event_type,
            ResponsesEventType::ResponseCreated
        );
        assert!(matches!(
            events.next().await,
            Some(Err(Error::InvalidResponse(_)))
        ));
        assert!(events.next().await.is_none());
    }
}
