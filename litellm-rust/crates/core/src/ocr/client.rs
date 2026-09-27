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
    pub fn ocr(&self, request: LiteLLMOcrRequest) -> crate::CallBuilder<'_, LiteLLMOcrRequest> {
        crate::CallBuilder::new(self, request)
    }
}

impl<'a, H: RouteHooks<Error>> std::future::IntoFuture
    for crate::CallBuilder<'a, LiteLLMOcrRequest, H>
{
    type Output = Result<LiteLLMOcrResponse, Error>;
    type IntoFuture = futures_util::future::BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            litellm_host::lifecycle::observe_unary(self.hooks.observer(), async {
                let client = self.client.ocr_client();
                execute(client, self.request, self.hooks).await
            })
            .await
        })
    }
}

pub(super) async fn execute(
    client: Result<OcrClient, litellm_http::Error>,
    request: LiteLLMOcrRequest,
    hooks: &impl RouteHooks<Error>,
) -> Result<LiteLLMOcrResponse, Error> {
    let client = client?;
    let caller_document = matches!(&request.document, OcrDocumentInput::Document(_));
    let prepared = prepare_request_document(request).await?;
    let execute: futures_util::future::BoxFuture<'_, Result<LiteLLMOcrResponse, Error>> = Box::pin(
        perform_ocr_request(&client, prepared, hooks, caller_document),
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
