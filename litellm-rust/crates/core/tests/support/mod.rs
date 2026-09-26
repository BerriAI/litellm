//! Shared fixtures for route integration tests: a scripted upstream and a recording
//! secret source.

#![allow(dead_code)] // each test binary compiles this module on its own and uses a different subset

use std::sync::{Arc, Mutex};

use futures_util::future::BoxFuture;
use litellm_http::{
    HttpClientConfig, HttpClientPool, HttpSettings, Resolution, media::PublicDnsResolver,
};
use litellm_secrets::{SecretValue, source::SecretSource};
use serde_json::Value;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    task::JoinHandle,
};
use wiremock::{Mock, MockServer, Request, ResponseTemplate, matchers::any};

/// A port nothing listens on, for calls that must fail before any request is sent.
pub const UNREACHABLE_BASE: &str = "http://127.0.0.1:1";

pub fn http_pool() -> HttpClientPool {
    HttpClientPool::new(Arc::new(PublicDnsResolver))
}

pub fn resources() -> litellm_core::resources::CoreResources {
    litellm_core::resources::CoreResources::new(Arc::new(http_pool()))
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

pub async fn truncated_sse_upstream(payload: &'static [u8]) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
    let base = format!("http://{}", listener.local_addr().expect("address"));
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let mut request = [0_u8; 4096];
        assert!(socket.read(&mut request).await.expect("reads request") > 0);
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            payload.len() + 1
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("writes headers");
        socket.write_all(payload).await.expect("writes body");
        socket.shutdown().await.expect("closes early");
    });
    (base, server)
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
