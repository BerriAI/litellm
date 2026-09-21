use litellm_http::outbound::RequestSigner;
use std::sync::Arc;

crate::base_llm::responses::transformation::scaffold_responses_translation_types!();

pub struct BedrockMantleResponsesApiConfig {
    pub aws_signer: Arc<dyn RequestSigner>,
    pub use_openai_path: bool,
}

impl BedrockMantleResponsesApiConfig {
    pub fn new(_aws_signer: Arc<dyn RequestSigner>, _use_openai_path: bool) -> Self {
        todo!()
    }
}

crate::base_llm::responses::transformation::scaffold_responses_translation_config!(
    @existing BedrockMantleResponsesApiConfig
);
