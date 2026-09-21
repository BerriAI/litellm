use litellm_types::{
    llms::openai::ChatMessage,
    responses::streaming::{ResponseTool, ResponsesInput},
};

use crate::base_llm::translation::TranslationError;

pub struct CountTokensRequest {
    pub model: String,
    pub input: ResponsesInput,
    pub tools: Option<Box<[ResponseTool]>>,
    pub instructions: Option<String>,
}

pub enum ChatTool {}

#[derive(Default)]
pub struct OpenAiCountTokensConfig;

impl OpenAiCountTokensConfig {
    pub fn get_openai_count_tokens_endpoint(
        &self,
        _api_base: Option<&str>,
    ) -> Result<url::Url, TranslationError> {
        todo!()
    }

    pub fn transform_request_to_count_tokens(
        &self,
        _model: &str,
        _input: ResponsesInput,
        _tools: Option<&[ChatTool]>,
        _instructions: Option<&str>,
    ) -> Result<CountTokensRequest, TranslationError> {
        todo!()
    }

    pub fn get_required_headers(&self, _api_key: &str) -> Box<[(String, String)]> {
        todo!()
    }

    pub fn validate_request(
        &self,
        _model: &str,
        _input: &ResponsesInput,
    ) -> Result<(), TranslationError> {
        todo!()
    }

    pub fn transform_tools_for_responses_api(
        _tools: &[ChatTool],
    ) -> Result<Box<[ResponseTool]>, TranslationError> {
        todo!()
    }

    pub fn messages_to_responses_input(
        _messages: &[ChatMessage],
    ) -> Result<(ResponsesInput, Option<String>), TranslationError> {
        todo!()
    }
}
