use futures_util::future::BoxFuture;
use litellm_auth::SecretValue;
use litellm_callbacks::event::{MachineEvent, RawResponse, RequestContext, WireRequest};
use litellm_llms::{
    base_llm::ocr::{
        error::Error,
        transformation::{LiteLLMOcrResponse, PreparedOcrRequest},
    },
    custom_httpx::llm_http_handler::{CallHooks, OcrClient},
};
use serde_json::Value;

use super::{
    arguments::is_secret_param, prepare::prepare_request, provider_config::OcrConfigKind,
    route::OcrHost,
};
use crate::ocr::types::ResolvedOcrRequest;

pub(crate) async fn perform_ocr_request(
    client: &OcrClient,
    request: ResolvedOcrRequest,
    host: &OcrHost,
    caller_document: bool,
) -> Result<LiteLLMOcrResponse, Error> {
    request.response_format()?;
    let config = request.config;
    let request = prepare_request(request, caller_document);
    let hooks = OcrCallHooks::new(host.clone(), &request, config);
    config.ocr(client, &request, &hooks).await
}

/// Lets provider code reach the host mid-call, filling in the request context only the
/// route knows.
pub(crate) struct OcrCallHooks {
    host: OcrHost,
    model: String,
    custom_llm_provider: &'static str,
    optional_params: Value,
    secret_fields: Vec<String>,
    api_key: Option<SecretValue>,
}

impl OcrCallHooks {
    pub(crate) fn new(host: OcrHost, request: &PreparedOcrRequest, config: OcrConfigKind) -> Self {
        Self {
            host,
            model: request.model.clone(),
            custom_llm_provider: config.provider().into(),
            optional_params: Value::Object(request.optional_params.clone().into()),
            secret_fields: request
                .optional_params
                .keys()
                .filter(|name| is_secret_param(name))
                .cloned()
                .collect(),
            api_key: request.connection.api_key.clone(),
        }
    }
}

impl CallHooks<Error> for OcrCallHooks {
    fn before_send(&self, wire: WireRequest) -> BoxFuture<'_, Result<WireRequest, Error>> {
        let context = RequestContext {
            model: self.model.clone(),
            custom_llm_provider: self.custom_llm_provider.into(),
            optional_params: self.optional_params.clone(),
            secret_fields: self.secret_fields.clone(),
            api_key: self.api_key.clone(),
        };
        Box::pin(self.host.before_send(wire, context))
    }

    fn response_received<'a>(&'a self, body: &'a [u8]) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(self.host.emit(MachineEvent::ResponseReceived {
            raw: RawResponse {
                body: String::from_utf8_lossy(body).into_owned(),
            },
        }))
    }
}
