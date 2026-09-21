use crate::base_llm::translation::TranslationError;

#[derive(Default)]
pub struct OpenAiCountTokensHandler;

impl OpenAiCountTokensHandler {
    pub fn count_tokens(&self, _request: serde_json::Value) -> Result<u64, TranslationError> {
        todo!()
    }
}
