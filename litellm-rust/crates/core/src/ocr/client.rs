use litellm_llms::base_llm::ocr::{
    error::Error, handler::OcrClient, transformation::LiteLLMOcrResponse,
};

use crate::ocr::{
    route::{LocalOcrHost, ocr_machine},
    types::LiteLLMOcrRequest,
};

pub async fn perform(
    client: &OcrClient,
    request: LiteLLMOcrRequest,
) -> Result<LiteLLMOcrResponse, Error> {
    litellm_host::run::run(ocr_machine(client.clone()), &LocalOcrHost::new(request)).await
}
