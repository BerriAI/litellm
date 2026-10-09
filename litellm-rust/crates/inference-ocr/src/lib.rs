pub mod arguments;
pub mod document;
pub(crate) mod handler;
pub mod provider_config;
pub mod route;
pub mod types;
pub mod wire;

use std::sync::Arc;

use litellm_host::interceptors::Interceptors;
use litellm_host::observation::ObservationSender;
use litellm_llms::base_llm::ocr::{error::Error, handler::OcrClient};
use litellm_llms_types::formats::ocr::LiteLLMOcrResponse;

use crate::{
    handler::execute,
    provider_config::resolve_provider_config,
    types::{LiteLLMOcrRequest, OcrDocumentInput},
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
        model = %call.model,
        provider,
        resolved_model,
        stream = false,
        outcome
    ))]
    async fn run(
        &self,
        call: LiteLLMOcrRequest,
        interceptors: &impl Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<LiteLLMOcrResponse, Error> {
        litellm_inference::diagnostic::unary(async {
            let original_model = call.requested_model.clone();
            let custom_llm_provider = call.custom_llm_provider.clone();
            let caller_document = matches!(&call.document, OcrDocumentInput::Document(_));
            let call = document::resolve_document(call).await?;
            let (provider, config) =
                resolve_provider_config(&original_model, custom_llm_provider.as_deref())?;
            litellm_inference::diagnostic::provider(
                provider.model,
                <&'static str>::from(provider.provider),
            );
            let secrets = self
                .client
                .secret_source()
                .resolve(&config.secret_names())
                .await
                .map_err(|error| Error::Secret(Arc::new(error)))?;
            let execute: futures_util::future::BoxFuture<'_, Result<LiteLLMOcrResponse, Error>> =
                Box::pin(execute(
                    &self.client,
                    config,
                    provider,
                    call,
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
