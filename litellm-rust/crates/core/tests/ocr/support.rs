use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::ocr::wire::{OcrWireRequest, decode_request};
use crate::ocr::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrClient};

pub(crate) fn ocr_client() -> OcrClient {
    let document_http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("test document client builds");
    OcrClient::for_test(reqwest::Client::new(), document_http)
}

pub(crate) async fn perform_ocr(
    request: LiteLLMOcrRequest,
) -> Result<LiteLLMOcrResponse, crate::Error> {
    ocr_client().perform(request).await
}

pub(crate) fn wire_request(model: &str, base: &str, options: Value) -> LiteLLMOcrRequest {
    decode_request(OcrWireRequest {
        model: model.into(),
        document: json!({"type":"document_url","document_url":"data:application/pdf;base64,YWJj"}),
        api_key: Some("test-key".into()),
        api_base: Some(base.into()),
        custom_llm_provider: None,
        extra_headers: None,
        optional_params: options.as_object().unwrap().clone(),
        timeout_seconds: Some(2.0),
    })
    .unwrap()
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
