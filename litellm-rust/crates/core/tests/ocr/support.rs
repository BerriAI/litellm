use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::ocr::hooks::{OcrDuringCallRequest, OcrHookFuture, OcrHooks};
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
) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
    ocr_client().perform(request).await
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

pub(crate) fn with_hooks(
    request: LiteLLMOcrRequest,
    hooks: Arc<dyn OcrHooks>,
) -> LiteLLMOcrRequest {
    LiteLLMOcrRequest { hooks, ..request }
}

pub(crate) struct RetainedFieldsHost {
    pub original_document: Value,
    pub retained_fields: Arc<Mutex<Vec<String>>>,
}

impl OcrHooks for RetainedFieldsHost {
    fn intercepts_requests(&self) -> bool {
        true
    }

    fn during_call(
        &self,
        mut request: OcrDuringCallRequest,
    ) -> OcrHookFuture<'_, OcrDuringCallRequest> {
        Box::pin(async move {
            *self.retained_fields.lock().unwrap() = request.retained_fields.clone();
            if request
                .retained_fields
                .iter()
                .any(|name| name == "document")
            {
                request.body["document"] = self.original_document.clone();
            }
            Ok(request)
        })
    }
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

    pub fn png(body: Value) -> Self {
        Self {
            headers: vec![("Content-Type", "image/png".into())],
            ..Self::json(body)
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
            let default_content_type = if response
                .headers
                .iter()
                .any(|(name, _)| name.eq_ignore_ascii_case("content-type"))
            {
                ""
            } else {
                "Content-Type: application/json\r\n"
            };
            let headers = response
                .headers
                .into_iter()
                .map(|(name, value)| {
                    format!("{name}: {}\r\n", value.replace("{base}", &server_base))
                })
                .collect::<String>();
            let head = format!(
                "HTTP/1.1 {} OK\r\n{}Content-Length: {}\r\nConnection: close\r\n{}\r\n",
                response.status,
                default_content_type,
                body.len(),
                headers
            );
            socket.write_all(head.as_bytes()).await.unwrap();
            socket.write_all(&body).await.unwrap();
        }
    });
    (base, requests, task)
}
