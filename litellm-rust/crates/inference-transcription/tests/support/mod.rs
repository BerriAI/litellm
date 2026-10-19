//! Shared fixtures for route integration tests: a scripted upstream and a recording
//! secret source.

#![allow(dead_code)] // each test binary compiles this module on its own and uses a different subset

use std::sync::{Arc, Mutex};

use litellm_inference_testing::{http_config, no_secrets, provider_http, resources};
use serde_json::Value;
use wiremock::{Mock, MockServer, Request, ResponseTemplate, matchers::any};

/// A port nothing listens on, for calls that must fail before any request is sent.
pub const UNREACHABLE_BASE: &str = "http://127.0.0.1:1";

pub fn audio_transcription_route() -> litellm_inference_transcription::AudioTranscriptionRoute {
    let resources = resources();
    litellm_inference_transcription::AudioTranscriptionRoute::new(
        provider_http(&resources, &http_config()),
        resources.auth,
        no_secrets(),
    )
}

/// Starts an upstream that answers its n-th request with the n-th response and 404s after.
pub async fn upstream(responses: impl IntoIterator<Item = ResponseTemplate>) -> MockServer {
    let server = MockServer::start().await;
    respond_in_order(&server, responses).await;
    server
}

/// Scripts responses on a started server, for responses that need its address.
pub async fn respond_in_order(
    server: &MockServer,
    responses: impl IntoIterator<Item = ResponseTemplate>,
) {
    for response in responses {
        Mock::given(any())
            .respond_with(response)
            .up_to_n_times(1)
            .mount(server)
            .await;
    }
}

pub async fn received(server: &MockServer) -> Vec<Request> {
    server
        .received_requests()
        .await
        .expect("request recording is on")
}

pub async fn only_request(server: &MockServer) -> Request {
    let [request] = <[Request; 1]>::try_from(received(server).await)
        .unwrap_or_else(|requests| panic!("expected one request, got {}", requests.len()));
    request
}

pub fn json_response(body: Value) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(body)
}

pub trait ReceivedRequest {
    fn header(&self, name: &str) -> Option<&str>;
    fn header_values(&self, name: &str) -> Vec<&str>;
    fn json(&self) -> Value;
    fn body_text(&self) -> String;
    /// The path and query, as the request line carried them.
    fn target(&self) -> String;
    fn query(&self, name: &str) -> Option<String>;
}

impl ReceivedRequest for Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|value| value.to_str().ok())
    }

    fn header_values(&self, name: &str) -> Vec<&str> {
        self.headers
            .get_all(name)
            .iter()
            .filter_map(|value| value.to_str().ok())
            .collect()
    }

    fn json(&self) -> Value {
        serde_json::from_slice(&self.body).expect("request body is json")
    }

    fn body_text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    fn target(&self) -> String {
        match self.url.query() {
            Some(query) => format!("{}?{query}", self.url.path()),
            None => self.url.path().to_string(),
        }
    }

    fn query(&self, name: &str) -> Option<String> {
        self.url
            .query_pairs()
            .find_map(|(key, value)| (key == name).then(|| value.into_owned()))
    }
}

#[derive(Clone, Default)]
pub struct TraceCapture(Arc<Mutex<Vec<Value>>>);

impl TraceCapture {
    pub fn logger(&self) -> litellm_tracing::Logger {
        litellm_tracing::Logger::new(self.clone())
    }

    pub fn records(&self) -> Vec<Value> {
        self.0.lock().unwrap().clone()
    }

    pub fn summaries(&self, name: &str) -> Vec<Value> {
        self.records()
            .into_iter()
            .filter(|record| record["span_name"] == name)
            .collect()
    }
}

impl litellm_tracing::Sink for TraceCapture {
    fn enabled(&self, metadata: &litellm_tracing::Metadata<'_>) -> bool {
        metadata.target().starts_with("litellm_inference")
    }

    fn emit(&self, record: &litellm_tracing::Record) {
        self.0
            .lock()
            .unwrap()
            .push(Value::Object(record.fields.clone()));
    }
}

#[rstest::fixture]
pub fn traces() -> TraceCapture {
    TraceCapture::default()
}
