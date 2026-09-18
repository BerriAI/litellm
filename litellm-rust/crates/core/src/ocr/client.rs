use litellm_llms::{
    base_llm::ocr::{error::Error, transformation::LiteLLMOcrResponse},
    custom_httpx::llm_http_handler::OcrClient,
};

use crate::ocr::{
    route::{LocalOcrHost, ocr_machine},
    types::LiteLLMOcrRequest,
};

pub async fn perform(
    client: &OcrClient,
    request: LiteLLMOcrRequest,
) -> Result<LiteLLMOcrResponse, Error> {
    litellm_callbacks::run::run(ocr_machine(client.clone()), &LocalOcrHost::new(request)).await
}

pub async fn ocr(request: LiteLLMOcrRequest) -> Result<LiteLLMOcrResponse, Error> {
    perform(&OcrClient::shared()?, request).await
}
