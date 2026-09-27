use litellm_core::ocr::{
    document::prepare_document,
    route::{Ocr, OcrOp, OcrProjection, ocr_machine},
    types::{LiteLLMOcrRequest, OcrDocumentInput},
    wire::{OcrWireRequest, decode_request},
};
use litellm_host::event::{CallEvent, RequestContext, WireRequest};
use litellm_http::Client;
use litellm_llms::base_llm::ocr::{
    error::Error,
    handler::OcrClient,
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

fn ocr_client() -> OcrClient {
    OcrClient::for_test(Client::plain_for_test(), Client::no_redirect_for_test())
}

async fn perform(request: LiteLLMOcrRequest) -> Result<LiteLLMOcrResponse, Error> {
    litellm_core::ocr::client::perform(&ocr_client(), request).await
}

async fn perform_with(host: LocalOcrHost) -> Result<LiteLLMOcrResponse, Error> {
    litellm_host::run::run_hosted(ocr_machine(ocr_client()), &host)
        .await
        .map(completed)
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
    before_send: Option<BeforeSend>,
    observer: Option<Observer>,
}

impl LocalOcrHost {
    fn new(request: LiteLLMOcrRequest<OcrDocumentInput>) -> Self {
        Self {
            request: Mutex::new(Some(request)),
            before_send: None,
            observer: None,
        }
    }

    fn with_before_send(
        self,
        before_send: impl Fn(WireRequest, &RequestContext) -> Result<WireRequest, Error>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            before_send: Some(Box::new(before_send)),
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

impl litellm_host::host::Host<Ocr> for LocalOcrHost {
    async fn project(&self) -> Result<OcrProjection, Error> {
        self.request
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
            .map(|request| OcrProjection {
                request,
                caller_token: false,
            })
            .ok_or_else(|| Error::InvalidRequest("OCR request was already projected".into()))
    }

    async fn custom_op(&self, op: OcrOp) -> Result<(), Error> {
        match op {
            OcrOp::AcquireAzureAdToken(_) => {
                Err(Error::Auth(litellm_auth::Error::AzureTokenAcquisition(
                    "OCR host has no Azure AD token provider".into(),
                )))
            }
        }
    }

    async fn before_send(
        &self,
        wire: WireRequest,
        context: &RequestContext,
    ) -> Result<WireRequest, Error> {
        match &self.before_send {
            Some(before_send) => before_send(wire, context),
            None => Ok(wire),
        }
    }

    async fn emit(&self, event: &CallEvent) -> Result<(), Error> {
        if let Some(observer) = &self.observer {
            observer(event);
        }
        Ok(())
    }
}
