use futures_util::future::BoxFuture;
use litellm_host::{
    event::{MachineEvent, RawResponse, RequestContext, WireRequest},
    hooks::RouteHooks,
};
use litellm_llms::base_llm::ocr::{
    error::Error,
    handler::{CallHooks, OcrClient},
    transformation::{LiteLLMOcrResponse, PreparedOcrRequest},
};
use serde_json::Value;

use super::{arguments::is_secret_param, prepare::prepare_request, provider_config::OcrConfigKind};
use crate::ocr::types::ResolvedOcrRequest;

pub(crate) async fn perform_ocr_request(
    client: &OcrClient,
    request: ResolvedOcrRequest,
    host: &impl RouteHooks<Error>,
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
    let hooks = OcrCallHooks::new(host, &request, config);
    config.ocr(client, &request, &hooks).await
}

struct OcrCallHooks<'a, H> {
    hooks: &'a H,
    context: RequestContext,
}

impl<'a, H> OcrCallHooks<'a, H> {
    fn new(hooks: &'a H, request: &PreparedOcrRequest, config: OcrConfigKind) -> Self {
        Self {
            hooks,
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

impl<H: RouteHooks<Error>> CallHooks<Error> for OcrCallHooks<'_, H> {
    fn before_send(&self, wire: WireRequest) -> BoxFuture<'_, Result<WireRequest, Error>> {
        Box::pin(self.hooks.before_send(wire, self.context.clone()))
    }

    fn response_received<'a>(&'a self, body: &'a [u8]) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(self.hooks.emit(MachineEvent::ResponseReceived {
            raw: RawResponse {
                body: String::from_utf8_lossy(body).into_owned(),
            },
        }))
    }
}
