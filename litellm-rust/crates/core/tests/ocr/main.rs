use litellm_core::ocr::{
    OcrRoute,
    document::prepare_document,
    route::{Ocr, OcrCall, OcrOp},
    types::{LiteLLMOcrRequest, OcrDocumentInput},
    wire::{OcrWireRequest, decode_request},
};
use litellm_host::{
    interceptors::{RequestContext, WireRequest},
    lifecycle::CallEvent,
};
use litellm_llms::base_llm::ocr::{
    error::Error,
    settings::OcrSettings,
    transformation::{LiteLLMOcrResponse, OcrDocument},
};
use serde_json::{Map, Value, json};
use std::sync::Mutex;
use wiremock::{MockServer, ResponseTemplate};

#[path = "../support/mod.rs"]
mod support;
use support::*;

mod aws_textract;
mod azure_ai;
mod azure_document_intelligence;
mod cohere;
mod documents;
mod lifecycle;
mod machine;
mod mistral;
mod reducto;
mod vertex_ai;

const INLINE_PDF: &str = "data:application/pdf;base64,YWJj";

fn object(value: Value) -> Map<String, Value> {
    let Value::Object(map) = value else {
        panic!("expected a json object, got {value}");
    };
    map
}

fn ocr_route() -> OcrRoute {
    ocr_route_with(OcrSettings::default())
}

fn ocr_route_with(settings: OcrSettings) -> OcrRoute {
    build_ocr_route(
        &resources(),
        &http_config(),
        litellm_http::media::UrlPolicy {
            validate: false,
            allowed_hosts: Vec::new(),
        },
        settings,
        no_secrets(),
    )
}

async fn perform(request: LiteLLMOcrRequest) -> Result<LiteLLMOcrResponse, Error> {
    ocr_route().execute(request, &(), None).await
}

async fn perform_with(host: LocalOcrHost) -> Result<LiteLLMOcrResponse, Error> {
    let result = litellm_host_native::in_process::run_hosted(
        ocr_route().machine(host.request()?, None),
        host.runtime(),
    )
    .await
    .map(completed);
    if let Some(observer) = &host.observer {
        for event in host.events.0.lock().unwrap().iter() {
            observer(event);
        }
    }
    result
}

fn wire(model: &str, base: &str, document: Value, options: Value) -> OcrWireRequest {
    OcrWireRequest {
        model: model.into(),
        document,
        api_key: Some(litellm_auth::SecretValue::new("test-key")),
        api_base: Some(base.into()),
        custom_llm_provider: None,
        extra_headers: None,
        optional_params: object(options),
        input_sources: Default::default(),
        timeout_seconds: Some(2.0),
    }
}

/// A request for an inline PDF, authenticated with `test-key`.
fn ocr_request(model: &str, base: &str, options: Value) -> LiteLLMOcrRequest {
    ocr_request_with_document(
        model,
        base,
        json!({"type": "document_url", "document_url": INLINE_PDF}),
        options,
    )
}

fn ocr_request_with_document(
    model: &str,
    base: &str,
    document: Value,
    options: Value,
) -> LiteLLMOcrRequest {
    decode_request(wire(model, base, document, options)).expect("request decodes")
}

fn document(value: Value) -> OcrDocument {
    serde_json::from_value(value).expect("document parses")
}

/// Points the request's resolved document at `source`, keeping its type.
fn with_source(request: LiteLLMOcrRequest, source: &str) -> LiteLLMOcrRequest {
    let resolved = request
        .map_document(prepare_document)
        .expect("document resolves");
    let document = resolved.document.clone().with_source(source.into());
    resolved.with_document(document.into())
}

fn with_headers(request: LiteLLMOcrRequest, headers: &[(&str, &str)]) -> LiteLLMOcrRequest {
    let mut request = request;
    request.transport.extra_headers = headers
        .iter()
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .collect();
    request
}

fn without_api_key(request: LiteLLMOcrRequest) -> LiteLLMOcrRequest {
    let mut request = request;
    request.credentials.api_key = None;
    request
}

fn pages_response() -> ResponseTemplate {
    json_response(json!({"pages": []}))
}

/// An Azure Document Intelligence 202 whose operation lives on `server`.
fn accepted(server: &MockServer, body: Value) -> ResponseTemplate {
    ResponseTemplate::new(202)
        .insert_header("Operation-Location", format!("{}/operation", server.uri()))
        .set_body_json(body)
}

fn completed(
    result: litellm_host::call::HostedCompletion<LiteLLMOcrResponse>,
) -> LiteLLMOcrResponse {
    match result {
        litellm_host::call::HostedCompletion::Complete(response) => response,
        other => panic!("unexpected OCR completion: {other:?}"),
    }
}

type BeforeSend =
    Box<dyn Fn(WireRequest, &RequestContext) -> Result<WireRequest, Error> + Send + Sync>;
type Observer = Box<dyn Fn(&CallEvent) + Send + Sync>;

struct LocalOcrHost {
    request: Mutex<Option<LiteLLMOcrRequest<OcrDocumentInput>>>,
    before_provider_request: Option<BeforeSend>,
    observer: Option<Observer>,
    events: support::CallEvents,
}

impl LocalOcrHost {
    fn new(request: LiteLLMOcrRequest<OcrDocumentInput>) -> Self {
        Self {
            request: Mutex::new(Some(request)),
            before_provider_request: None,
            observer: None,
            events: support::CallEvents::default(),
        }
    }

    fn with_before_send(
        self,
        before_provider_request: impl Fn(WireRequest, &RequestContext) -> Result<WireRequest, Error>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            before_provider_request: Some(Box::new(before_provider_request)),
            ..self
        }
    }

    fn with_observer(self, observer: impl Fn(&CallEvent) + Send + Sync + 'static) -> Self {
        Self {
            observer: Some(Box::new(observer)),
            ..self
        }
    }
}

impl LocalOcrHost {
    pub fn request(&self) -> Result<OcrCall, Error> {
        self.request
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
            .map(|request| OcrCall {
                request,
                caller_token: false,
            })
            .ok_or_else(|| Error::InvalidRequest("OCR request was already projected".into()))
    }
    pub fn runtime(&self) -> litellm_host_native::in_process::Host<'_, Self, Self, ()> {
        litellm_host_native::in_process::Host {
            services: self,
            interceptors: self,
            stream: &(),
            observers: Some(&self.events.0.sender),
        }
    }
}
impl litellm_host_native::services::HostCallHandler<Ocr> for LocalOcrHost {
    async fn handle_host_call(&self, op: OcrOp) -> Result<(), Error> {
        match op {
            OcrOp::AcquireAzureAdToken(_) => {
                Err(Error::Auth(litellm_auth::Error::CredentialAcquisition(
                    "OCR host has no Azure AD token provider".into(),
                )))
            }
        }
    }
}

impl litellm_host::lifecycle::CallObserver for LocalOcrHost {
    fn observe(&self, event: litellm_host::lifecycle::CallEvent) {
        self.events.0.sender.emit(event);
    }
}
impl litellm_host::interceptors::Interceptors<<Ocr as litellm_host::protocol::Protocol>::Error>
    for LocalOcrHost
{
    async fn before_provider_request(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> Result<WireRequest, Error> {
        match &self.before_provider_request {
            Some(before_provider_request) => before_provider_request(wire, &context),
            None => Ok(wire),
        }
    }
    async fn after_provider_response(
        &self,
        raw: litellm_host::interceptors::RawResponse,
    ) -> Result<(), <Ocr as litellm_host::protocol::Protocol>::Error> {
        litellm_host::lifecycle::CallObserver::observe(
            self,
            litellm_host::lifecycle::CallEvent::Execution(
                litellm_host::lifecycle::ExecutionEvent::ProviderResponseReceived { raw },
            ),
        );
        Ok(())
    }
}
