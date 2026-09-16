use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::ocr::{
    LiteLLMOcrRequest, LiteLLMOcrResponse, OcrClient, OcrCredentialInputs, OcrDocument,
};

pub(crate) fn ocr_client() -> OcrClient {
    let document_http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("test document client builds");
    OcrClient::for_test(reqwest::Client::new(), document_http)
}

pub(crate) async fn perform_ocr(
    request: LiteLLMOcrRequest,
) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
    ocr_client().perform(request).await
}

pub(crate) fn wire_request(model: &str, base: &str, options: Value) -> LiteLLMOcrRequest {
    let request = LiteLLMOcrRequest::new(
        model.into(),
        OcrDocument::try_from(
            json!({"type":"document_url","document_url":"data:application/pdf;base64,YWJj"}),
        )
        .unwrap(),
        None,
        options.as_object().unwrap().clone().into(),
    )
    .unwrap();
    let transport = request.transport.clone().with_overrides(
        Vec::new(),
        Default::default(),
        Some(std::time::Duration::from_secs(2)),
    );
    request.with_connection_inputs(
        OcrCredentialInputs::new(
            Some("test-key".into()),
            Default::default(),
            Some(base.into()),
            Default::default(),
        ),
        transport,
        Default::default(),
    )
}

pub(crate) struct MockResponse {
    pub(crate) status: u16,
    pub(crate) headers: Vec<(&'static str, String)>,
    pub(crate) body: Value,
}

impl MockResponse {
    pub(crate) fn json(body: Value) -> Self {
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
