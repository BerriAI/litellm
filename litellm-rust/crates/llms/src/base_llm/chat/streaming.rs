use std::collections::HashMap;

use futures_util::{StreamExt, stream::BoxStream};
use litellm_types::utils::ChatCompletionChunk;

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
