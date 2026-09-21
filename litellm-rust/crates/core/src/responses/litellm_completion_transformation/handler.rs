use futures_util::Stream;
use litellm_llms::base_llm::base_model_iterator::TransformedStream;
use litellm_types::utils::ChatCompletionChunk;

use super::streaming_iterator::LiteLlmCompletionStreamingIterator;
use crate::responses::streaming_iterator::{ResponsesApiStreamingIterator, ResponsesStreamError};

pub struct LiteLlmCompletionTransformationHandler;

impl LiteLlmCompletionTransformationHandler {
    pub fn streaming_response<S>(
        _chat_stream: S,
        _adapter: LiteLlmCompletionStreamingIterator,
    ) -> ResponsesApiStreamingIterator<TransformedStream<S, LiteLlmCompletionStreamingIterator>>
    where
        S: Stream<Item = Result<ChatCompletionChunk, ResponsesStreamError>>,
    {
        todo!(
            "Compose the transformed Chat source inside the same driver used for native Responses events"
        )
    }
}
