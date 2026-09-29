use std::collections::HashMap;

use futures_util::{StreamExt, stream::BoxStream};
use litellm_framing::{frames, sse::SseCodec};
use litellm_llms_types::formats::chat_completions::ChatCompletionChunk;

use crate::{
    Error,
    base_llm::base_model_iterator::{ByteStream, StreamError, StreamTransformer, transform_stream},
};

pub type ChatChunkStream = BoxStream<'static, Result<ChatCompletionChunk, Error>>;

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

enum OpenAiChatEvent {
    Chunk(ChatCompletionChunk),
    Done,
}

fn openai_chat_events(bytes: ByteStream) -> BoxStream<'static, Result<OpenAiChatEvent, Error>> {
    Box::pin(frames(bytes, SseCodec::default()).map(|event| {
        let event = event.map_err(|error| {
            Error::InvalidResponse(crate::ErrorDetail::failed("stream framing", error))
        })?;
        if event.data == "[DONE]" {
            return Ok(OpenAiChatEvent::Done);
        }
        serde_json::from_str(&event.data)
            .map(OpenAiChatEvent::Chunk)
            .map_err(|error| {
                Error::InvalidResponse(crate::ErrorDetail::invalid("Chat stream event", error))
            })
    }))
}

struct OpenAiChatIterator {
    done: bool,
    terminal_choice: bool,
}

impl StreamTransformer for OpenAiChatIterator {
    type Input = OpenAiChatEvent;
    type Output = ChatCompletionChunk;
    type Error = Error;

    fn transform(&mut self, input: Self::Input) -> Result<Vec<Self::Output>, Self::Error> {
        match input {
            OpenAiChatEvent::Done => {
                if self.done || !self.terminal_choice {
                    return Err(Error::InvalidResponse(
                        "premature Chat stream termination".into(),
                    ));
                }
                self.done = true;
                Ok(Vec::new())
            }
            OpenAiChatEvent::Chunk(chunk) if self.done => {
                let _ = chunk;
                Err(Error::InvalidResponse(
                    "Chat chunk after stream termination".into(),
                ))
            }
            OpenAiChatEvent::Chunk(chunk) => {
                self.terminal_choice |= chunk
                    .choices
                    .iter()
                    .any(|choice| choice.finish_reason.is_some());
                Ok(vec![chunk])
            }
        }
    }

    fn finish(&mut self) -> Result<Vec<Self::Output>, Self::Error> {
        if self.done {
            Ok(Vec::new())
        } else {
            Err(Error::InvalidResponse("truncated Chat stream".into()))
        }
    }
}

pub fn openai_chat_stream() -> ChatStream {
    ChatStream::new(
        openai_chat_events,
        OpenAiChatIterator {
            done: false,
            terminal_choice: false,
        },
    )
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use futures_util::{StreamExt, TryStreamExt, stream};
    use rstest::rstest;

    use super::*;

    fn pieces(wire: &'static str) -> ByteStream {
        stream::iter(
            wire.as_bytes()
                .chunks(3)
                .map(|piece| Ok(Bytes::copy_from_slice(piece)))
                .collect::<Vec<_>>(),
        )
        .boxed()
    }

    #[rstest]
    #[tokio::test]
    async fn fragmented_chat_events_retain_tool_argument_fragments() {
        let wire = concat!(
            "data: {\"id\":\"chat_1\",\"created\":1,\"model\":\"m\",\"object\":\"chat.completion.chunk\",",
            "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",",
            "\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"x\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chat_1\",\"created\":1,\"model\":\"m\",\"object\":\"chat.completion.chunk\",",
            "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"type\":\"function\",",
            "\"function\":{\"arguments\":\":1}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chat_1\",\"created\":1,\"model\":\"m\",\"object\":\"chat.completion.chunk\",",
            "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let chunks = openai_chat_stream()
            .run(pieces(wire))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks[0].choices[0].delta.tool_calls.as_ref().unwrap()[0]
                .function
                .arguments,
            "{\"x"
        );
        assert_eq!(
            chunks[1].choices[0].delta.tool_calls.as_ref().unwrap()[0]
                .function
                .arguments,
            ":1}"
        );
        assert_eq!(
            chunks[2].choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[rstest]
    #[tokio::test]
    async fn truncated_stream_fails_after_the_last_chunk() {
        let wire = concat!(
            "data: {\"id\":\"chat_1\",\"created\":1,\"model\":\"m\",\"object\":\"chat.completion.chunk\",",
            "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n"
        );
        let output = openai_chat_stream()
            .run(pieces(wire))
            .collect::<Vec<_>>()
            .await;
        assert_eq!(output.len(), 2);
        assert!(output[0].is_ok());
        assert!(matches!(output[1], Err(Error::InvalidResponse(_))));
    }
}
