use litellm_types::{
    llms::anthropic_messages::{
        anthropic_request::AnthropicMessagesRequest, anthropic_response::AnthropicMessagesResponse,
    },
    responses::streaming::{ResponsesRequest, ResponsesResponse},
};

use crate::base_llm::translation::TranslationError;

#[derive(Default)]
pub struct LiteLlmAnthropicToResponsesApiAdapter;

impl LiteLlmAnthropicToResponsesApiAdapter {
    pub fn translate_request(
        &self,
        _anthropic_request: AnthropicMessagesRequest,
        _include_encrypted_reasoning: bool,
    ) -> Result<ResponsesRequest, TranslationError> {
        todo!()
    }

    pub fn translate_response(
        &self,
        _response: ResponsesResponse,
    ) -> Result<AnthropicMessagesResponse, TranslationError> {
        todo!()
    }
}
