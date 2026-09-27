use std::sync::Arc;

use litellm_host::hooks::RouteHooks;
use litellm_llms::base_llm::ocr::{
    error::Error, handler::OcrClient, transformation::LiteLLMOcrResponse,
};

use super::{
    handler::perform_ocr_request,
    types::{LiteLLMOcrRequest, OcrDocumentInput, ResolvedOcrRequest},
};

impl crate::CoreClient {
    pub async fn ocr(&self, request: LiteLLMOcrRequest) -> Result<LiteLLMOcrResponse, Error> {
        self.ocr_with_hooks(request, &()).await
    }

    pub fn ocr_with_hooks<'a>(
        &'a self,
        request: LiteLLMOcrRequest,
        hooks: &'a impl RouteHooks<Error>,
    ) -> futures_util::future::BoxFuture<'a, Result<LiteLLMOcrResponse, Error>> {
        Box::pin(async move {
            litellm_host::lifecycle::observe_unary(hooks.observer(), async {
                let client = self.ocr_client()?;
                execute(&client, request, hooks).await
            })
            .await
        })
    }
}

pub(super) async fn execute(
    client: &OcrClient,
    request: LiteLLMOcrRequest,
    hooks: &impl RouteHooks<Error>,
) -> Result<LiteLLMOcrResponse, Error> {
    let caller_document = matches!(&request.document, OcrDocumentInput::Document(_));
    let prepared = prepare_request_document(request).await?;
    let execute: futures_util::future::BoxFuture<'_, Result<LiteLLMOcrResponse, Error>> = Box::pin(
        perform_ocr_request(client, prepared, hooks, caller_document),
    );
    execute.await
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
