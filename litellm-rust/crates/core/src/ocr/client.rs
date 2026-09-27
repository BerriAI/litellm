use std::sync::Arc;

use litellm_host::hooks::RouteHooks;
use litellm_llms::base_llm::ocr::{
    error::Error, handler::OcrClient, transformation::LiteLLMOcrResponse,
};

use super::{
    handler::perform_ocr_request,
    types::{LiteLLMOcrRequest, OcrDocumentInput, ResolvedOcrRequest},
};

pub async fn perform(
    client: &OcrClient,
    request: LiteLLMOcrRequest,
) -> Result<LiteLLMOcrResponse, Error> {
    perform_with_hooks(client, request, &()).await
}

pub fn perform_with_hooks<'a>(
    client: &'a OcrClient,
    request: LiteLLMOcrRequest,
    hooks: &'a impl RouteHooks<Error>,
) -> futures_util::future::BoxFuture<'a, Result<LiteLLMOcrResponse, Error>> {
    Box::pin(async move {
        litellm_host::lifecycle::observe_unary(hooks.observer(), async {
            let caller_document = matches!(&request.document, OcrDocumentInput::Document(_));
            let prepared = prepare_request_document(request).await?;
            let execute: futures_util::future::BoxFuture<'_, Result<LiteLLMOcrResponse, Error>> =
                Box::pin(perform_ocr_request(
                    client,
                    prepared,
                    hooks,
                    caller_document,
                ));
            execute.await
        })
        .await
    })
}

async fn prepare_request_document(
    request: LiteLLMOcrRequest<OcrDocumentInput>,
) -> Result<ResolvedOcrRequest, Error> {
    if let OcrDocumentInput::Document(_) = &request.document {
        return request.map_document(super::document::prepare_document);
    }
    tokio::task::spawn_blocking(move || request.map_document(super::document::prepare_document))
        .await
        .map_err(|error| Error::DocumentTask(Arc::new(error)))?
}
