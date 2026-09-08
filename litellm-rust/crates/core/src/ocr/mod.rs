mod client;
mod common_utils;
mod handler;
mod hooks;
pub mod prepare;
mod runtime_types;
pub mod transformation;
pub mod types;

use serde_json::Value;

use crate::Error;
use crate::error::json_type_name;
use crate::http_utils::{buffered_post, has_header};

pub use runtime_types::{OcrRequest, OcrRouteRequest};
pub use types::{OcrAdmissionRequest, OcrResponseData, PreparedOcr, PreparedOcrCall};
use types::{OcrDocument, OcrDocumentProjection};

use crate::integrations::custom_guardrail::CustomGuardrailRunner;
use crate::integrations::custom_logger::CustomLoggerRunner;
use crate::lifecycle::{
    CallLifecycle, CallLifecycleContext, Clock, ExecutedCall, TerminalDispatcher,
};
use hooks::OcrRequestPolicy;
use runtime_types::PreparedOcrRequest;

pub trait OcrServices: TerminalDispatcher + Clock {}

impl<T> OcrServices for T where T: TerminalDispatcher + Clock {}

pub struct DefaultOcrServices {
    dispatcher: CustomLoggerRunner,
}

pub struct NoopOcrServices;

impl Default for NoopOcrServices {
    fn default() -> Self {
        Self
    }
}

impl DefaultOcrServices {
    pub fn new(request: &OcrRequest<'_>) -> Self {
        Self {
            dispatcher: CustomLoggerRunner::new(request.callbacks.clone()),
        }
    }
}

impl Clock for DefaultOcrServices {
    fn now(&self) -> f64 {
        crate::lifecycle::SystemClock.now()
    }
}

impl TerminalDispatcher for DefaultOcrServices {
    fn dispatch<'a>(
        &'a self,
        terminal: &'a crate::lifecycle::TerminalRecord,
    ) -> crate::integrations::custom_logger::LogFuture<'a> {
        self.dispatcher.dispatch(terminal)
    }
}

impl Clock for NoopOcrServices {
    fn now(&self) -> f64 {
        crate::lifecycle::SystemClock.now()
    }
}

impl TerminalDispatcher for NoopOcrServices {
    fn dispatch<'a>(
        &'a self,
        _: &'a crate::lifecycle::TerminalRecord,
    ) -> crate::integrations::custom_logger::LogFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

pub async fn ocr<S: OcrServices>(
    services: &S,
    request: OcrRouteRequest<'_>,
    _options: crate::lifecycle::ocr::Options,
    context: CallLifecycleContext,
) -> ExecutedCall<Value, Error> {
    let OcrRouteRequest::Native(request) = request else {
        let OcrRouteRequest::Prepared(request) = request else {
            unreachable!()
        };
        return CallLifecycle
            .run(
                context,
                request,
                &PreparedOcrPolicy,
                services,
                services,
                |request| async move {
                    send_prepared(request.prepared, request.headers, request.body)
                        .await
                        .map(OcrResponseData::into_json)
                },
            )
            .await;
    };
    let policy = OcrRequestPolicy::new(
        CustomGuardrailRunner::new(request.guardrails.clone()),
        request.request_metadata.clone(),
    );
    let provider = crate::routing_utils::provider::get_custom_llm_provider(
        request.model,
        request.custom_llm_provider,
    );
    let config = provider
        .as_ref()
        .ok_or_else(|| Error::InvalidProvider("unable to resolve OCR provider".into()))
        .and_then(|provider| {
            common_utils::ocr_provider_config(provider.custom_llm_provider, provider.model)
                .ok_or_else(|| Error::InvalidProvider("unsupported OCR provider".into()))
        });
    let provider_model = provider.as_ref().map_or(request.model, |value| value.model);
    let provider_name = provider
        .as_ref()
        .map_or(request.custom_llm_provider.unwrap_or(""), |value| {
            value.custom_llm_provider
        });
    let prepared = PreparedOcrRequest {
        config,
        model: provider_model.to_string(),
        custom_llm_provider: provider_name.to_string(),
        litellm_call_id: context.litellm_call_id.clone(),
        document: request.document,
        api_key: request.api_key.map(str::to_string),
        api_base: request.api_base.map(str::to_string),
        extra_headers: request.extra_headers,
        optional_params: request.optional_params,
        timeout: request.timeout,
    };
    CallLifecycle
        .run(context, prepared, &policy, services, services, |request| {
            handler::execute_ocr_provider_call(request, &policy)
        })
        .await
}

struct PreparedOcrPolicy;

impl crate::lifecycle::RequestPolicy<PreparedOcrCall, PreparedOcrCall> for PreparedOcrPolicy {
    type PreCallFuture<'a>
        = std::future::Ready<crate::lifecycle::ActionResult<PreparedOcrCall, Error>>
    where
        Self: 'a;
    type DuringCallFuture<'a>
        = std::future::Ready<crate::lifecycle::ActionResult<PreparedOcrCall, Error>>
    where
        Self: 'a;

    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: PreparedOcrCall,
    ) -> Self::PreCallFuture<'a> {
        std::future::ready(crate::lifecycle::ActionResult::Continue(request))
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: PreparedOcrCall,
    ) -> Self::DuringCallFuture<'a> {
        std::future::ready(crate::lifecycle::ActionResult::Continue(request))
    }
}

pub(crate) async fn send_prepared(
    prepared: PreparedOcr,
    headers: Vec<(String, String)>,
    body: Value,
) -> Result<OcrResponseData, Error> {
    let config = prepare::provider_config(&prepared.custom_llm_provider, &prepared.model)?;
    prepare::validate_capabilities(config)?;
    let object = body.as_object().ok_or_else(|| Error::InvalidType {
        expected: "object",
        actual: json_type_name(&body),
    })?;
    if config.document_projection() != OcrDocumentProjection::Transformed {
        let document = object
            .get("document")
            .ok_or(Error::MissingField("document"))?;
        let document: OcrDocument = serde_json::from_value(document.clone())
            .map_err(|_| Error::InvalidRequest("invalid OCR document".into()))?;
        document.validate(config.requires_data_uri_document())?;
    }
    if object.get("stream").and_then(Value::as_bool) == Some(true) {
        return Err(Error::Unsupported("OCR streaming response handling"));
    }
    if headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("content-encoding") && !value.eq_ignore_ascii_case("identity")
    }) {
        return Err(Error::Unsupported("compressed OCR request"));
    }
    let headers = headers
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .chain(
            [
                ("Content-Type", "application/json"),
                ("Accept-Encoding", "identity"),
            ]
            .into_iter()
            .filter(|(name, _)| !has_header(&headers, name)),
        )
        .map(|(name, value)| (name.as_bytes().to_vec(), value.as_bytes().to_vec()))
        .collect();
    let body = serde_json::to_vec(&body)
        .map_err(|_| Error::InvalidRequest("could not encode OCR request".into()))?;
    let response = buffered_post::send(buffered_post::Request {
        url: prepared.url,
        headers,
        body,
        timeout_seconds: prepared.timeout_seconds,
    })
    .await?;
    if !(200..300).contains(&response.status) {
        return Err(Error::Http {
            status: response.status,
            body: "OCR provider request failed".into(),
        });
    }
    if response.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case(b"content-encoding")
            && value
                .split(|byte| *byte == b',')
                .any(|encoding| !encoding.trim_ascii().eq_ignore_ascii_case(b"identity"))
    }) {
        return Err(Error::Unsupported("compressed OCR response"));
    }
    let response_json = serde_json::from_slice(&response.content)
        .map_err(|_| Error::InvalidResponse("invalid OCR JSON response".into()))?;
    config.transform_ocr_response(&prepared.model, response_json)
}
