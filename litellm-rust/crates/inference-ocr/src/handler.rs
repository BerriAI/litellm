use futures_util::future::BoxFuture;
use litellm_host::interceptors::{Interceptors, RawResponse, RequestContext, WireRequest};
use litellm_llms::base_llm::ocr::{
    error::Error,
    handler::{CallHooks, OcrClient},
    transformation::PreparedOcrRequest,
};
use litellm_llms_types::formats::ocr::LiteLLMOcrResponse;
use serde_json::Value;

use super::{arguments::is_secret_param, prepare::prepare_request, provider_config::OcrConfigKind};
use crate::types::ResolvedOcrRequest;

pub(crate) async fn perform_ocr_request(
    client: &OcrClient,
    request: ResolvedOcrRequest,
    host: &impl Interceptors<Error>,
    caller_document: bool,
) -> Result<LiteLLMOcrResponse, Error> {
    request.response_format()?;
    let config = request.config;
    let secrets = client
        .secret_source()
        .resolve(&config.secret_names())
        .await
        .map_err(|error| Error::Secret(std::sync::Arc::new(error)))?;
    let request = prepare_request(request, caller_document, client, secrets);
    let interceptors = OcrCallHooks::new(host, &request, config);
    config.ocr(client, &request, &interceptors).await
}

struct OcrCallHooks<'a, H> {
    interceptors: &'a H,
    context: RequestContext,
}

impl<'a, H> OcrCallHooks<'a, H> {
    fn new(interceptors: &'a H, request: &PreparedOcrRequest, config: OcrConfigKind) -> Self {
        Self {
            interceptors,
            context: RequestContext {
                model: request.model.clone(),
                custom_llm_provider: <&str>::from(config.provider()).to_owned(),
                optional_params: Value::Object(request.optional_params.clone().into()),
                secret_fields: request
                    .optional_params
                    .keys()
                    .filter(|name| is_secret_param(name))
                    .cloned()
                    .collect(),
                api_key: request.connection.api_key.clone(),
            },
        }
    }
}

impl<H: Interceptors<Error>> CallHooks<Error> for OcrCallHooks<'_, H> {
    fn before_provider_request(
        &self,
        wire: WireRequest,
    ) -> BoxFuture<'_, Result<WireRequest, Error>> {
        Box::pin(
            self.interceptors
                .before_provider_request(wire, self.context.clone()),
        )
    }

    fn response_received<'a>(&'a self, body: &'a [u8]) -> BoxFuture<'a, Result<(), Error>> {
        let raw = RawResponse {
            body: String::from_utf8_lossy(body).into_owned(),
        };
        Box::pin(self.interceptors.after_provider_response(raw))
    }
}
