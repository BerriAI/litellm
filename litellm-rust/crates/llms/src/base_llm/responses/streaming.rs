use litellm_framing::sse::SseFrame;
use litellm_types::responses::streaming::{ResponsesEvent, ResponsesResponse};

use crate::base_llm::base_model_iterator::{StreamError, StreamOutcome, StreamPolicy};

pub struct ResponsesSseDecoder {
    _policy: StreamPolicy,
    _terminal: Option<StreamOutcome<ResponsesResponse>>,
}

impl ResponsesSseDecoder {
    pub fn new(_policy: StreamPolicy) -> Self {
        todo!(
            "Initialize semantic decoding downstream of SseFramer, which owns split UTF-8 and frame assembly"
        )
    }

    pub fn decode(&mut self, _frame: SseFrame) -> Result<Vec<ResponsesEvent>, StreamError> {
        todo!(
            "Decode canonical events, skip SSE control frames, handle DONE, preserve extensions and usage, and enforce event/buffer limits"
        )
    }

    pub fn finish(self) -> Result<StreamOutcome<ResponsesResponse>, StreamError> {
        todo!(
            "Require a terminal Responses event; EOF or DONE alone cannot fabricate completed output"
        )
    }
}

pub fn decode_response(_body: &[u8]) -> Result<ResponsesResponse, StreamError> {
    todo!("Validate the full wire schema and retain provider extensions in the typed response")
}

pub fn encode_event(_event: &ResponsesEvent) -> Result<String, StreamError> {
    todo!("Serialize canonical event names and payload fields for SSE or WebSocket delivery")
}
