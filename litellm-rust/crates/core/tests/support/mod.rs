//! Shared fixtures for route integration tests: a scripted upstream and a recording
//! secret source.

#![allow(dead_code)] // each test binary compiles this module on its own and uses a different subset

use std::sync::{Arc, Mutex};

use futures_util::future::BoxFuture;
use litellm_http::{
    ClientVariant, HttpClientConfig, HttpClientPool, HttpSettings, Resolution,
    media::PublicDnsResolver,
};
use litellm_secrets::{SecretValue, source::SecretSource};
use serde_json::Value;
use wiremock::{Mock, MockServer, Request, ResponseTemplate, matchers::any};

/// A port nothing listens on, for calls that must fail before any request is sent.
pub const UNREACHABLE_BASE: &str = "http://127.0.0.1:1";

pub fn http_pool() -> HttpClientPool {
    HttpClientPool::new(Arc::new(PublicDnsResolver))
}

pub fn resources() -> litellm_core::resources::CoreResources {
    litellm_core::resources::CoreResources::new(Arc::new(http_pool()))
}

pub fn no_secrets() -> Arc<dyn SecretSource> {
    Arc::new(RecordingSecrets::empty())
}

pub fn provider_http(
    resources: &litellm_core::resources::CoreResources,
    config: &HttpClientConfig,
) -> litellm_http::Client {
    resources
        .pool
        .client(config, ClientVariant::Provider)
        .unwrap()
}

pub fn messages_route(secrets: Arc<dyn SecretSource>) -> litellm_core::messages::MessagesRoute {
    let resources = resources();
    litellm_core::messages::MessagesRoute::new(
        provider_http(&resources, &http_config()),
        resources.auth,
        secrets,
    )
}

pub fn chat_completions_route() -> litellm_core::chat_completions::ChatCompletionsRoute {
    let resources = resources();
    litellm_core::chat_completions::ChatCompletionsRoute::new(
        provider_http(&resources, &http_config()),
        resources.auth,
        no_secrets(),
    )
}

pub fn responses_route(secrets: Arc<dyn SecretSource>) -> litellm_core::responses::ResponsesRoute {
    let resources = resources();
    litellm_core::responses::ResponsesRoute::new(
        provider_http(&resources, &http_config()),
        resources.auth,
        secrets,
    )
}

pub fn audio_transcription_route() -> litellm_core::audio_transcription::AudioTranscriptionRoute {
    let resources = resources();
    litellm_core::audio_transcription::AudioTranscriptionRoute::new(
        provider_http(&resources, &http_config()),
        resources.auth,
        no_secrets(),
    )
}

pub fn build_ocr_route(
    resources: &litellm_core::resources::CoreResources,
    config: &HttpClientConfig,
    url_policy: litellm_http::media::UrlPolicy,
    settings: litellm_llms::base_llm::ocr::settings::OcrSettings,
    secrets: Arc<dyn SecretSource>,
) -> litellm_core::ocr::OcrRoute {
    litellm_core::ocr::OcrRoute::new(
        litellm_llms::base_llm::ocr::handler::OcrClient::new(
            &resources.pool,
            config,
            url_policy,
            resources.auth.clone(),
            settings,
            secrets,
        )
        .unwrap(),
    )
}

pub fn http_config() -> HttpClientConfig {
    Resolution::from(&HttpSettings::default()).config
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

pub fn status_response(status: u16, body: Value) -> ResponseTemplate {
    ResponseTemplate::new(status).set_body_json(body)
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

/// A secret source that answers from a fixed table and records every name it was asked for.
pub struct RecordingSecrets {
    values: Vec<(String, String)>,
    fails: bool,
    requested: Mutex<Vec<String>>,
}

impl RecordingSecrets {
    pub fn new<'a>(values: impl IntoIterator<Item = (&'a str, &'a str)>) -> Self {
        Self {
            values: values
                .into_iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
            fails: false,
            requested: Mutex::new(Vec::new()),
        }
    }

    pub fn empty() -> Self {
        Self::new([])
    }

    pub fn failing() -> Self {
        Self {
            fails: true,
            ..Self::empty()
        }
    }

    pub fn requested(&self) -> Vec<String> {
        self.requested.lock().unwrap().clone()
    }
}

impl SecretSource for RecordingSecrets {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_secrets::Error>> {
        Box::pin(async move {
            self.requested.lock().unwrap().push(name.to_string());
            if self.fails {
                return Err(litellm_secrets::Error::ManagedSecretMissing);
            }
            Ok(self
                .values
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| SecretValue::new(value.clone())))
        })
    }
}

pub struct RecordingCall<P: litellm_host::protocol::Protocol> {
    pub request: Mutex<Option<P::Request>>,
    pub events: Arc<CallEvents>,
    pub chunks: Mutex<Vec<P::Chunk>>,
    pub head: Mutex<Option<P::StreamHead>>,
}

#[derive(Default)]
pub struct CallEvents(pub Mutex<Vec<litellm_host::event::CallEvent>>);

impl litellm_host::lifecycle::CallObserver for CallEvents {
    fn observe(&self, event: litellm_host::event::CallEvent) {
        self.0.lock().unwrap().push(event);
    }
}

impl<P: litellm_host::protocol::Protocol> RecordingCall<P> {
    pub fn new(request: P::Request) -> Self {
        Self {
            request: Mutex::new(Some(request)),
            events: Arc::new(CallEvents::default()),
            chunks: Mutex::new(Vec::new()),
            head: Mutex::new(None),
        }
    }
}

impl<P: litellm_host::protocol::Protocol> litellm_host::hooks::RouteHooks<P::Error>
    for RecordingCall<P>
{
    fn observer(&self) -> Option<Arc<dyn litellm_host::lifecycle::CallObserver>> {
        Some(self.events.clone())
    }

    async fn before_provider_request(
        &self,
        wire: litellm_host::event::WireRequest,
        _: litellm_host::event::RequestContext,
    ) -> Result<litellm_host::event::WireRequest, P::Error> {
        Ok(litellm_host::event::WireRequest {
            headers: wire
                .headers
                .into_iter()
                .chain([("x-hook".into(), "called".into())])
                .collect(),
            ..wire
        })
    }

    async fn on_event(&self, event: litellm_host::event::MachineEvent) -> Result<(), P::Error> {
        self.events
            .0
            .lock()
            .unwrap()
            .push(litellm_host::event::CallEvent::Machine(event));
        Ok(())
    }
}

impl<P> RecordingCall<P>
where
    P: litellm_host::protocol::Protocol<HostCall = std::convert::Infallible>,
    P::Error: From<litellm_host::machine::MachineFault>,
{
    pub fn request(&self) -> Result<P::Request, P::Error> {
        self.request
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| litellm_host::machine::MachineFault::Abandoned.into())
    }
    pub fn runtime(&self) -> litellm_host::in_process::Host<'_, (), Self, Self> {
        litellm_host::in_process::Host {
            services: &(),
            hooks: self,
            stream: self,
            observer: Some(self),
        }
    }
}

impl<P> litellm_host::in_process::StreamConsumer<P> for RecordingCall<P>
where
    P: litellm_host::protocol::Protocol<HostCall = std::convert::Infallible>,
    P::Error: From<litellm_host::machine::MachineFault>,
{
    async fn open_stream(
        &self,
        head: P::StreamHead,
    ) -> Result<litellm_host::protocol::Demand, P::Error> {
        *self.head.lock().unwrap() = Some(head);
        Ok(litellm_host::protocol::Demand::More)
    }
    async fn send_chunk(
        &self,
        chunk: P::Chunk,
    ) -> Result<litellm_host::protocol::Demand, P::Error> {
        self.chunks.lock().unwrap().push(chunk);
        Ok(litellm_host::protocol::Demand::More)
    }
}
impl<P> litellm_host::lifecycle::CallObserver for RecordingCall<P>
where
    P: litellm_host::protocol::Protocol<HostCall = std::convert::Infallible>,
    P::Error: From<litellm_host::machine::MachineFault>,
{
    fn observe(&self, event: litellm_host::event::CallEvent) {
        self.events.0.lock().unwrap().push(event.clone());
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
        metadata.target().starts_with("litellm_core")
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
