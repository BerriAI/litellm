use litellm_core::ocr::{
    document::prepare_document,
    route::{LocalOcrHost, ocr_machine},
    types::LiteLLMOcrRequest,
    wire::{OcrWireRequest, decode_request},
};
use litellm_llms::base_llm::ocr::{
    error::Error,
    handler::OcrClient,
    transformation::{LiteLLMOcrResponse, OcrDocument},
};
use serde_json::{Map, Value, json};
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
    let document_http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("test document client builds");
    OcrClient::for_test(reqwest::Client::new(), document_http)
}

async fn perform(request: LiteLLMOcrRequest) -> Result<LiteLLMOcrResponse, Error> {
    litellm_core::ocr::client::perform(&ocr_client(), request).await
}

async fn perform_with(host: LocalOcrHost) -> Result<LiteLLMOcrResponse, Error> {
    litellm_host::run::run(ocr_machine(ocr_client()), &host).await
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
