use litellm_types::utils::ChatCompletionChunk;

use crate::{
    anthropic::experimental_pass_through::messages::streaming_iterator::AnthropicMessagesStreamEvent,
    base_llm::{
        base_model_iterator::{AnthropicMessagesApi, ChatCompletions, StreamTransformation},
        chat::transformation::Error,
    },
};

pub struct AnthropicStreamWrapper {
    // TODO: Port block indexes, held stop/usage chunks, thinking signatures and compaction state from Python
    _state: (),
}

impl AnthropicStreamWrapper {
    pub fn new(_model: String) -> Self {
        todo!("Initialize one Chat-to-Messages conversion without owning upstream reads")
    }

    pub fn handle_choiceless_chunk(&mut self, _chunk: &ChatCompletionChunk) -> Result<(), Error> {
        todo!("Preserve late usage and provider fields even when choices are empty")
    }

    pub fn merge_usage_into_held_stop_reason_chunk(
        &mut self,
        _chunk: &ChatCompletionChunk,
    ) -> Result<AnthropicMessagesStreamEvent, Error> {
        todo!(
            "Merge final usage, cache usage, iterations and context management before message_delta and message_stop"
        )
    }
}

impl StreamTransformation for AnthropicStreamWrapper {
    type Caller = AnthropicMessagesApi;
    type Upstream = ChatCompletions;
    type Error = Error;

    fn transform_event(
        &mut self,
        _chunk: ChatCompletionChunk,
    ) -> Result<Vec<AnthropicMessagesStreamEvent>, Error> {
        todo!(
            "Port _CombinedChunkSplitter and AnthropicStreamWrapper: ordered thinking/text/tools, block starts/stops, signatures, refusal and compaction; retain usage exactly once"
        )
    }

    fn finish(self) -> Result<Vec<AnthropicMessagesStreamEvent>, Error> {
        todo!(
            "Validate upstream termination and close held blocks/stops after late usage; reject truncated streams"
        )
    }
}
