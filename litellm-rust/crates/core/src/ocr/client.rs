use std::sync::Arc;

use litellm_host::hooks::RouteHooks;
use litellm_llms::base_llm::ocr::{
    error::Error, handler::OcrClient, transformation::LiteLLMOcrResponse,
};

use super::{
    handler::perform_ocr_request,
    types::{LiteLLMOcrRequest, OcrDocumentInput, ResolvedOcrRequest},
};

#[derive(Clone)]
pub struct OcrRoute {
    client: OcrClient,
}

impl OcrRoute {
    pub fn new(client: OcrClient) -> Self {
        Self { client }
    }

    pub async fn execute(
        &self,
        request: LiteLLMOcrRequest,
        hooks: &impl RouteHooks<Error>,
    ) -> Result<LiteLLMOcrResponse, Error> {
        litellm_host::lifecycle::observe_unary(hooks.observer(), self.run(request, hooks)).await
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "ocr",
        model = %request.model,
        resolved_model = %request.model,
        provider = <&str>::from(request.config.provider()),
        stream = false,
        outcome
    ))]
    pub(super) async fn run(
        &self,
        request: LiteLLMOcrRequest,
        hooks: &impl RouteHooks<Error>,
    ) -> Result<LiteLLMOcrResponse, Error> {
        crate::diagnostic::unary(async {
            let caller_document = matches!(&request.document, OcrDocumentInput::Document(_));
            let prepared = prepare_request_document(request).await?;
            let execute: futures_util::future::BoxFuture<'_, Result<LiteLLMOcrResponse, Error>> =
                Box::pin(perform_ocr_request(
                    &self.client,
                    prepared,
                    hooks,
                    caller_document,
                ));
            execute.await
        })
        .await
    }
}

#[tracing::instrument(name = "litellm.prepare", level = "debug", skip_all)]
async fn prepare_request_document(
    request: LiteLLMOcrRequest<OcrDocumentInput>,
) -> Result<ResolvedOcrRequest, Error> {
    if let OcrDocumentInput::Document(_) = &request.document {
        return request.map_document(super::document::prepare_document);
    }
    let logger = litellm_tracing::Logger::current();
    let span = tracing::Span::current();
    tokio::task::spawn_blocking(move || {
        logger.scope(|| span.in_scope(|| request.map_document(super::document::prepare_document)))
    })
    .await
    .map_err(|error| Error::DocumentTask(Arc::new(error)))?
}
