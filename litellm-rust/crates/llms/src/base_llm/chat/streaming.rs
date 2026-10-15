use std::collections::HashMap;

use bytes::Bytes;
use futures_util::{StreamExt, stream::BoxStream};
use litellm_llms_types::formats::chat_completions::ChatCompletionChunk;

use crate::{
    Error,
    base_llm::{
        base_model_iterator::{ByteStream, StreamError, StreamTransformer, transform_stream},
        sse::{self, is_done},
    },
};

pub type ChatChunkStream = BoxStream<'static, Result<ChatCompletionChunk, Error>>;

/// OpenAI-compatible SSE: bare `data:` frames closed by `data: [DONE]`.
pub fn event_stream(bytes: ByteStream) -> ChatChunkStream {
    sse::event_stream(bytes, is_done)
}

pub fn decode(wire: &[u8]) -> Result<Vec<ChatCompletionChunk>, Error> {
    sse::decode(wire, is_done)
}

pub fn encode(chunk: &ChatCompletionChunk) -> Result<Bytes, Error> {
    sse::encode_data(chunk)
}

/// What Python's `map_openai_params` decides about the stream and `completion`
/// hands to `ModelResponseIterator`: it is settled while the request is built,
/// never re-derived from the body.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamShape {
    pub json_mode: bool,
    pub speed: Option<String>,
    pub tool_name_reverse_map: HashMap<String, String>,
}

/// A wire decoder paired with the iterator that turns its events into chat chunks.
/// A config names both; the core runs the pair over the response bytes.
pub struct ChatStream {
    run: Box<dyn FnOnce(ByteStream) -> ChatChunkStream + Send>,
}

impl ChatStream {
    pub fn new<E, T>(
        decode: fn(ByteStream) -> BoxStream<'static, Result<E, Error>>,
        iterator: T,
    ) -> Self
    where
        E: Send + 'static,
        T: StreamTransformer<Input = E, Output = ChatCompletionChunk, Error = Error>
            + Send
            + 'static,
    {
        Self {
            run: Box::new(move |bytes| {
                Box::pin(transform_stream(decode(bytes), iterator).map(|item| {
                    item.map_err(|error| match error {
                        StreamError::Decode(error) | StreamError::Transform(error) => error,
                    })
                }))
            }),
        }
    }

    pub fn run(self, bytes: ByteStream) -> ChatChunkStream {
        (self.run)(bytes)
    }
}

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt, TryStreamExt, stream};
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn in_pieces(wire: &[u8], size: usize) -> ByteStream {
        let pieces: Vec<Bytes> = wire.chunks(size).map(Bytes::copy_from_slice).collect();
        stream::iter(pieces.into_iter().map(Ok)).boxed()
    }

    #[rstest]
    #[case::role(json!({"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}))]
    #[case::text(json!({"id":"c","choices":[{"index":0,"delta":{"content":"hi\nthere"}}]}))]
    #[case::tool(json!({"id":"c","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t","type":"function","function":{"name":"f","arguments":"{\"x\":"}}]}}]}))]
    #[case::finish(json!({"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}))]
    #[case::usage(json!({"id":"c","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3,"prompt_tokens_details":{"cached_tokens":0,"cache_creation_tokens":0,"text_tokens":1}}}))]
    #[case::host_extension(json!({"id":"c","choices":[],"citations":[{"url":"u"}]}))]
    fn encoded_chunks_are_bare_data_frames_that_decode_back(#[case] wire: Value) {
        let chunk: ChatCompletionChunk = serde_json::from_value(wire.clone()).unwrap();
        let encoded = encode(&chunk).unwrap();
        assert!(encoded.starts_with(b"data: "));
        assert!(encoded.ends_with(b"\n\n"));
        assert_eq!(
            serde_json::from_slice::<Value>(&encoded[b"data: ".len()..encoded.len() - 2]).unwrap(),
            wire
        );
        assert_eq!(decode(&encoded).unwrap(), vec![chunk]);
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
            ": keepalive{newline}{newline}data: {{\"id\":\"c\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"a\"}}}}]}}{newline}{newline}data:{{\"id\":\"c\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}{newline}{newline}data: [DONE]{newline}{newline}"
        );
        let streamed = event_stream(in_pieces(wire.as_bytes(), size))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(streamed, decode(wire.as_bytes()).unwrap());
        assert_eq!(streamed.len(), 2);
        assert_eq!(streamed[0].choices[0].delta.content.as_deref(), Some("a"));
        assert_eq!(
            streamed[1].choices[0].finish_reason.as_deref(),
            Some("stop")
        );
    }

    #[rstest]
    #[case::spaced(b"data: [DONE]\n\n")]
    #[case::unspaced(b"data:[DONE]\n\n")]
    #[case::trailing_space(b"data: [DONE] \n\n")]
    fn every_done_spelling_closes_the_stream_without_a_chunk(#[case] wire: &[u8]) {
        assert_eq!(decode(wire).unwrap(), vec![]);
    }

    #[rstest]
    #[case::invalid_json(b"data: {\n\n")]
    #[case::choices_not_a_list(b"data: {\"choices\":{}}\n\n")]
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
    async fn a_body_error_ends_the_stream_after_delivered_chunks() {
        let mut chunks = event_stream(
            stream::iter([
                Ok(Bytes::from_static(
                    b"data: {\"id\":\"c\",\"choices\":[]}\n\n",
                )),
                Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionReset,
                    "reset",
                )),
            ])
            .boxed(),
        );
        assert_eq!(
            chunks.next().await.unwrap().unwrap().id.as_deref(),
            Some("c")
        );
        assert!(matches!(
            chunks.next().await,
            Some(Err(Error::InvalidResponse(_)))
        ));
        assert!(chunks.next().await.is_none());
    }
}
