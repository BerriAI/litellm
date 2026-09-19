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
    litellm_host::run::run(ocr_machine(client.clone()), &LocalOcrHost::new(request)).await
}
