pub mod prepare;
pub mod transformation;
pub mod types;

use serde_json::Value;
use std::future::Future;

use crate::Error;
use crate::error::json_type_name;
use crate::http_utils::{buffered_post, has_header};

pub use types::{
    OcrAdmissionRequest, OcrDraft, OcrEndpoint, OcrResponseData, OcrTransportRequest,
    OcrTransportResponse, SettledOcrRequest,
};
use types::{OcrDocument, OcrDocumentProjection};

use crate::lifecycle::{
    CallLifecycle, CallLifecycleContext, Clock, ExecutedCall, TerminalDispatcher,
};

pub trait OcrTransport {
    type SendFuture<'a>: Future<Output = Result<OcrTransportResponse, Error>>
    where
        Self: 'a;

    fn send(&self, request: OcrTransportRequest) -> Self::SendFuture<'_>;
}

pub trait OcrServices: TerminalDispatcher + Clock + OcrTransport {}

impl<T> OcrServices for T where T: TerminalDispatcher + Clock + OcrTransport {}

pub struct DefaultOcrServices;

impl Default for DefaultOcrServices {
    fn default() -> Self {
        Self
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
        _: &'a crate::lifecycle::TerminalRecord,
    ) -> crate::integrations::custom_logger::LogFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

impl OcrTransport for DefaultOcrServices {
    type SendFuture<'a> = impl Future<Output = Result<OcrTransportResponse, Error>> + 'a;

    fn send(&self, request: OcrTransportRequest) -> Self::SendFuture<'_> {
        async move {
            let response = buffered_post::send(buffered_post::Request {
                url: request.url,
                headers: request.headers,
                body: request.body,
                timeout_seconds: request.timeout_seconds,
            })
            .await?;
            Ok(OcrTransportResponse {
                status: response.status,
                headers: response.headers,
                content: response.content,
            })
        }
    }
}

pub async fn ocr<S: OcrServices>(
    services: &S,
    request: SettledOcrRequest,
    _options: crate::lifecycle::ocr::Options,
    context: CallLifecycleContext,
) -> ExecutedCall<Value, Error> {
    CallLifecycle
        .run(
            context,
            request,
            &SettledOcrPolicy,
            services,
            services,
            |request| async move {
                send(services, request)
                    .await
                    .map(OcrResponseData::into_json)
            },
        )
        .await
}

struct SettledOcrPolicy;

impl crate::lifecycle::RequestPolicy<SettledOcrRequest, SettledOcrRequest> for SettledOcrPolicy {
    type PreCallFuture<'a>
        = std::future::Ready<crate::lifecycle::ActionResult<SettledOcrRequest, Error>>
    where
        Self: 'a;
    type DuringCallFuture<'a>
        = std::future::Ready<crate::lifecycle::ActionResult<SettledOcrRequest, Error>>
    where
        Self: 'a;

    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: SettledOcrRequest,
    ) -> Self::PreCallFuture<'a> {
        std::future::ready(crate::lifecycle::ActionResult::Continue(request))
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: SettledOcrRequest,
    ) -> Self::DuringCallFuture<'a> {
        std::future::ready(crate::lifecycle::ActionResult::Continue(request))
    }
}

pub(crate) async fn send<S: OcrTransport>(
    transport: &S,
    request: SettledOcrRequest,
) -> Result<OcrResponseData, Error> {
    let SettledOcrRequest {
        endpoint,
        headers,
        body,
    } = request;
    let config = prepare::provider_config(&endpoint.custom_llm_provider, &endpoint.model)?;
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
    let response = transport
        .send(OcrTransportRequest {
        url: endpoint.url,
        headers,
        body,
        timeout_seconds: endpoint.timeout_seconds,
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
    config.transform_ocr_response(&endpoint.model, response_json)
}
