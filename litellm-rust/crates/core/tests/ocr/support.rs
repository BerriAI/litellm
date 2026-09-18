use std::sync::{Arc, Mutex};

use futures_util::future::BoxFuture;
use litellm_callbacks::event::{Passthrough, WireRequest};
use litellm_llms::{
    base_llm::ocr::{error::Error, transformation::LiteLLMOcrResponse},
    custom_httpx::llm_http_handler::{CallHooks, OcrClient},
};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

use crate::ocr::{
    route::{LocalOcrHost, ocr_machine},
    types::LiteLLMOcrRequest,
    wire::{OcrWireRequest, decode_request},
};

/// Stands in for a host with no hooks registered: the wire request goes out unchanged
/// and response events go nowhere.
pub(crate) struct NoHooks;

impl CallHooks<Error> for NoHooks {
    fn before_send(
        &self,
        wire: WireRequest,
        _passthrough_fields: Passthrough,
    ) -> BoxFuture<'_, Result<WireRequest, Error>> {
        Box::pin(async move { Ok(wire) })
    }

    fn response_received<'a>(&'a self, _body: &'a [u8]) -> BoxFuture<'a, Result<(), Error>> {
        Box::pin(async { Ok(()) })
    }
}

pub(crate) fn ocr_client() -> OcrClient {
    let document_http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("test document client builds");
    OcrClient::for_test(reqwest::Client::new(), document_http)
}

pub(crate) async fn perform_ocr(request: LiteLLMOcrRequest) -> Result<LiteLLMOcrResponse, Error> {
    crate::ocr::client::perform(&ocr_client(), request).await
}

pub(crate) async fn perform_ocr_with(host: LocalOcrHost) -> Result<LiteLLMOcrResponse, Error> {
    litellm_callbacks::run::run(ocr_machine(ocr_client()), &host).await
}

pub(crate) fn wire_request(model: &str, base: &str, options: Value) -> LiteLLMOcrRequest {
    wire_request_with_document(
        model,
        base,
        json!({"type":"document_url","document_url":"data:application/pdf;base64,YWJj"}),
        options,
    )
}

pub(crate) fn wire_request_with_document(
    model: &str,
    base: &str,
    document: Value,
    options: Value,
) -> LiteLLMOcrRequest {
    decode_request(OcrWireRequest {
        model: model.into(),
        document,
        api_key: Some("test-key".into()),
        api_base: Some(base.into()),
        custom_llm_provider: None,
        extra_headers: None,
        optional_params: options.as_object().unwrap().clone(),
        input_sources: Default::default(),
        timeout_seconds: Some(2.0),
    })
    .unwrap()
}

pub(crate) fn resolved_request(
    request: LiteLLMOcrRequest,
) -> crate::ocr::types::ResolvedOcrRequest {
    request
        .map_document(crate::ocr::document::prepare_document)
        .unwrap()
}

pub(crate) fn with_source(request: LiteLLMOcrRequest, source: &str) -> LiteLLMOcrRequest {
    let request = resolved_request(request);
    let document = request.document.clone().with_source(source.into());
    request.with_document(document.into())
}

pub(crate) fn request_body(request: &str) -> Value {
    serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
}

pub(crate) const SERVED_DOCUMENT: &[u8] = b"\x89PNG served document";

/// Serves [`SERVED_DOCUMENT`] as `image/png` to every connection until aborted.
pub(crate) async fn document_server() -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buffer = [0u8; 4096];
            let _ = socket.read(&mut buffer).await.unwrap();
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                SERVED_DOCUMENT.len()
            );
            socket.write_all(head.as_bytes()).await.unwrap();
            socket.write_all(SERVED_DOCUMENT).await.unwrap();
        }
    });
    (base, task)
}

pub(crate) struct MockResponse {
    pub status: u16,
    pub headers: Vec<(&'static str, String)>,
    pub body: Value,
}

impl MockResponse {
    pub fn json(body: Value) -> Self {
        Self {
            status: 200,
            headers: vec![],
            body,
        }
    }
}

pub(crate) async fn mock_server(
    responses: Vec<MockResponse>,
) -> (String, Arc<Mutex<Vec<String>>>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = requests.clone();
    let server_base = base.clone();
    let task = tokio::spawn(async move {
        for response in responses {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let mut buffer = [0u8; 4096];
            let header_end = loop {
                let n = socket.read(&mut buffer).await.unwrap();
                assert!(n > 0);
                bytes.extend_from_slice(&buffer[..n]);
                if let Some(index) = bytes.windows(4).position(|s| s == b"\r\n\r\n") {
                    break index + 4;
                }
            };
            let length = String::from_utf8_lossy(&bytes[..header_end])
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            while bytes.len() < header_end + length {
                let n = socket.read(&mut buffer).await.unwrap();
                assert!(n > 0);
                bytes.extend_from_slice(&buffer[..n]);
            }
            seen.lock()
                .unwrap()
                .push(String::from_utf8_lossy(&bytes).into_owned());
            let body = serde_json::to_vec(&response.body).unwrap();
            let headers = response
                .headers
                .into_iter()
                .map(|(name, value)| {
                    format!("{name}: {}\r\n", value.replace("{base}", &server_base))
                })
                .collect::<String>();
            let head = format!(
                "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n",
                response.status,
                body.len(),
                headers
            );
            socket.write_all(head.as_bytes()).await.unwrap();
            socket.write_all(&body).await.unwrap();
        }
    });
    (base, requests, task)
}

pub(crate) fn header<'a>(request: &'a str, name: &str) -> Option<&'a str> {
    request
        .lines()
        .take_while(|line| !line.is_empty())
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name).then(|| value.trim())
        })
}
