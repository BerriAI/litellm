use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::transformation::OcrProviderConfig;
use super::types::OcrConnection;
use super::wire::{OcrWireRequest, decode_request, decode_response};
use super::{OcrRequest, perform_ocr as run_ocr};
use crate::ocr::error::OcrError;

pub(crate) fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("test HTTP client builds")
}

pub(crate) async fn perform_ocr(
    request: OcrRequest,
) -> Result<super::OcrResponseData, crate::Error> {
    run_ocr(&http_client(), request).await
}

pub(crate) fn params<C: OcrProviderConfig>(config: &C, value: Value) -> C::MappedParams {
    config
        .map_ocr_params(serde_json::from_value(value).unwrap())
        .unwrap()
}
pub(crate) fn transform<C: OcrProviderConfig>(
    config: &C,
    model: &str,
    value: Value,
    options: Value,
) -> Result<Value, OcrError> {
    let params = config.map_ocr_params(serde_json::from_value(options).unwrap())?;
    let decoded = decode_response(
        &serde_json::to_vec(&value).unwrap(),
        config.preserve_native_response(&params),
    )?;
    let mut response = config.transform_ocr_response(model, decoded.data, &params)?;
    response.provider_native_response = decoded.native;
    Ok(response.into_json())
}
pub(crate) async fn body<C: OcrProviderConfig>(
    config: &C,
    model: &str,
    document: Value,
    options: Value,
) -> Result<Value, OcrError> {
    let params = config.map_ocr_params(serde_json::from_value(options).unwrap())?;
    let http_client = http_client();
    let document = config
        .prepare_document(
            &http_client,
            serde_json::from_value(document).unwrap(),
            &OcrConnection::default(),
            &[],
        )
        .await?;
    Ok(serde_json::to_value(config.transform_ocr_request(model, document, &params)?).unwrap())
}
pub(crate) fn wire_request(model: &str, base: &str, options: Value) -> OcrRequest {
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

#[tokio::test]
async fn performs_mistral_ocr_from_typed_request() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[{"index":0,"markdown":"hello","custom":"preserved"}],"usage_info":{"pages_processed":1}}))]).await;
    let result = perform_ocr(wire_request(
        "mistral/model",
        &base,
        json!({"extract_header":true,"unknown":"ignored"}),
    ))
    .await
    .unwrap();
    server.await.unwrap();
    assert_eq!(result.pages[0].markdown, "hello");
    assert_eq!(result.pages[0].extra_fields["custom"], "preserved");
    let request = seen.lock().unwrap();
    assert!(request[0].starts_with("POST /v1/ocr "));
    let body: Value = serde_json::from_str(request[0].split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(body["extract_header"], true);
    assert!(body.get("unknown").is_none());
}

#[tokio::test]
async fn uses_the_host_injected_http_client() {
    let (base, seen, server) = mock_server(vec![MockResponse::json(
        json!({"pages":[],"usage_info":{"pages_processed":0}}),
    )])
    .await;
    let mut default_headers = reqwest::header::HeaderMap::new();
    default_headers.insert(
        "x-transport-owner",
        reqwest::header::HeaderValue::from_static("python-sdk"),
    );
    let http_client = reqwest::Client::builder()
        .default_headers(default_headers)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("test HTTP client builds");

    run_ocr(
        &http_client,
        wire_request("mistral/model", &base, json!({})),
    )
    .await
    .expect("OCR request succeeds");

    server.await.expect("mock server completes");
    assert!(seen.lock().unwrap()[0].contains("x-transport-owner: python-sdk"));
}

#[tokio::test]
async fn invalid_pages_fail_before_network_and_invoke_failure_hook() {
    use crate::call_lifecycle::{CallLifecycleContext, CallLifecycleTiming};
    use crate::ocr::hooks::{OcrHooks, OcrLogFuture};
    struct Failures(Arc<Mutex<Vec<String>>>);
    impl OcrHooks for Failures {
        fn failure<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            error: &'a crate::Error,
            _timing: &'a CallLifecycleTiming,
        ) -> OcrLogFuture<'a> {
            Box::pin(async move {
                self.0.lock().unwrap().push(error.to_string());
            })
        }
    }
    let failures = Arc::new(Mutex::new(Vec::new()));
    let mut request = wire_request(
        "azure_ai/doc-intelligence/prebuilt-read",
        "http://127.0.0.1:1",
        json!({"pages":[-1]}),
    );
    request.hooks = Arc::new(Failures(failures.clone()));
    assert_eq!(
        perform_ocr(request).await.unwrap_err().kind(),
        crate::error::ErrorKind::InvalidRequest
    );
    assert_eq!(failures.lock().unwrap().len(), 1);
}

#[test]
fn typed_response_rejects_malformed_pages_with_field_path() {
    let error = decode_response::<crate::providers::mistral::ocr::types::MistralOcrResponse>(
        br#"{"pages":[{"index":0,"markdown":42}]}"#,
        false,
    )
    .unwrap_err();
    assert!(error.to_string().contains("pages[0].markdown"));
    assert!(!error.to_string().contains("42"));
}
