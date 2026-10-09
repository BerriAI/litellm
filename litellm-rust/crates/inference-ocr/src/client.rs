use litellm_host::observation::ObservationSender;
use std::sync::Arc;

use litellm_host::interceptors::Interceptors;
use litellm_llms::base_llm::ocr::{error::Error, handler::OcrClient};
use litellm_llms_types::formats::ocr::LiteLLMOcrResponse;

use super::{
    handler::execute,
    provider_config::resolve_provider_config,
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
        interceptors: &impl Interceptors<Error>,
        observers: Option<ObservationSender>,
    ) -> Result<LiteLLMOcrResponse, Error> {
        litellm_host::lifecycle::observe_unary(
            observers.clone(),
            self.run(request, interceptors, observers.as_ref()),
        )
        .await
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "ocr",
        model = %request.model,
        provider,
        resolved_model,
        stream = false,
        outcome
    ))]
    pub(super) async fn run(
        &self,
        request: LiteLLMOcrRequest,
        interceptors: &impl Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<LiteLLMOcrResponse, Error> {
        litellm_inference::diagnostic::unary(async {
            let original_model = request.requested_model.clone();
            let custom_llm_provider = request.custom_llm_provider.clone();
            let caller_document = matches!(&request.document, OcrDocumentInput::Document(_));
            let request = resolve_document(request).await?;
            let (provider, config) =
                resolve_provider_config(&original_model, custom_llm_provider.as_deref())?;
            litellm_inference::diagnostic::provider(
                provider.model,
                <&'static str>::from(provider.provider),
            );
            let secret_names = config.secret_names();
            let secrets = self
                .client
                .secret_source()
                .resolve(&secret_names)
                .await
                .map_err(|error| Error::Secret(Arc::new(error)))?;
            let execute: futures_util::future::BoxFuture<'_, Result<LiteLLMOcrResponse, Error>> =
                Box::pin(execute(
                    &self.client,
                    config,
                    provider,
                    request,
                    secrets,
                    interceptors,
                    caller_document,
                    observers,
                ));
            execute.await
        })
        .await
    }
}

async fn resolve_document(
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
