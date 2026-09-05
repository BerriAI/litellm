use litellm_core::Error;
use litellm_core::call_lifecycle::CallLifecycle;
use serde_json::Value;

mod common_utils;
mod handler;
mod hooks;
mod prepare;
mod types;

pub use types::OcrRequest;

use handler::execute_ocr_provider_call;
use prepare::{PreparedOcrCall, prepare_ocr_call};

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn ocr(request: OcrRequest<'_>) -> Result<Value, Error> {
    let PreparedOcrCall { request, hooks } = prepare_ocr_call(request);
    CallLifecycle::default()
        .run_request(request, &hooks, |request| {
            execute_ocr_provider_call(request, &hooks)
        })
        .await
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, json};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    use super::{OcrRequest, ocr};
    use crate::integrations::types::RequestMetadata;

    async fn read_http_request(socket: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        let header_end = loop {
            let n = socket.read(&mut buffer).await.expect("reads request");
            if n == 0 {
                break request.len();
            }
            request.extend_from_slice(&buffer[..n]);
            if let Some(position) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break position + 4;
            }
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        while request.len().saturating_sub(header_end) < content_length {
            let n = socket.read(&mut buffer).await.expect("reads body");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..n]);
        }
        String::from_utf8(request).expect("request is utf8")
    }

    fn base_ocr_request(model: &str) -> OcrRequest<'_> {
        OcrRequest {
            model,
            document: json!({
                "type": "document_url",
                "document_url": "https://example.com/doc.pdf"
            }),
            api_key: Some("sk-test"),
            api_base: None,
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: Map::new(),
            timeout: None,
            callbacks: Vec::new(),
            guardrails: Vec::new(),
            request_metadata: RequestMetadata::default(),
            litellm_call_id: None,
        }
    }

    #[tokio::test]
    async fn reducto_file_upload_then_parse_maps_response() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let address = listener.local_addr().expect("listener has local address");
        let server = tokio::spawn(async move {
            let (mut upload_socket, _) = listener.accept().await.expect("accepts upload request");
            let upload_request = read_http_request(&mut upload_socket).await;
            let upload_body = r#"{"file_id":"reducto://uploaded.pdf"}"#;
            let upload_response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                upload_body.len(),
                upload_body
            );
            upload_socket
                .write_all(upload_response.as_bytes())
                .await
                .expect("writes upload response");

            let (mut parse_socket, _) = listener.accept().await.expect("accepts parse request");
            let parse_request = read_http_request(&mut parse_socket).await;
            let parse_body = r#"{"job_id":"job_123","usage":{"num_pages":3,"credits":3},"result":{"chunks":[{"content":"Page 1 block A","blocks":[{"content":"Page 1 block A","bbox":{"page":1},"kind":"text"}]},{"content":"Page 2 block A","blocks":[{"content":"Page 2 block A","bbox":{"page":2},"kind":"table"}]},{"content":"Page 1 block B","blocks":[{"content":"Page 1 block B","bbox":{"page":1},"kind":"text"}]},{"content":"Page 3 block A","blocks":[{"content":"Page 3 block A","bbox":{"page":3},"kind":"figure"}]}]}}"#;
            let parse_response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                parse_body.len(),
                parse_body
            );
            parse_socket
                .write_all(parse_response.as_bytes())
                .await
                .expect("writes parse response");
            (upload_request, parse_request)
        });
        let api_base = format!("http://{address}");
        let mut request = base_ocr_request("reducto/parse-v3");
        request.api_base = Some(&api_base);
        request.api_key = None;
        request.extra_headers = Some(Map::from_iter([
            ("Authorization".to_string(), json!("Bearer test-key")),
            ("x-trace-id".to_string(), json!("trace-1")),
        ]));
        request.document = json!({
            "type": "document_url",
            "document_url": "data:application/pdf;base64,JVBERi0xLjQ="
        });
        request.optional_params = Map::from_iter([
            (
                "formatting".to_string(),
                json!({"table_output_format": "html"}),
            ),
            ("retrieval".to_string(), json!({"chunk_mode": "section"})),
            ("settings".to_string(), json!({"ocr_system": "standard"})),
        ]);

        let response = ocr(request).await.expect("Reducto OCR succeeds");

        assert_eq!(response["pages"].as_array().map(Vec::len), Some(3));
        assert_eq!(
            response["pages"][0]["markdown"],
            "Page 1 block A\n\nPage 1 block B"
        );
        assert_eq!(response["pages"][1]["markdown"], "Page 2 block A");
        assert_eq!(response["pages"][2]["markdown"], "Page 3 block A");
        assert_eq!(response["usage_info"]["pages_processed"], 3);
        assert_eq!(response["usage_info"]["credits"], 3);
        assert_eq!(response["provider_native_response"]["job_id"], "job_123");
        let (upload_request, parse_request) = server.await.expect("server task completes");
        assert!(
            upload_request
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key")
        );
        assert!(upload_request.contains("application/pdf"));
        assert!(upload_request.contains("%PDF-1.4"));
        assert!(upload_request.contains("x-trace-id: trace-1"));
        assert!(
            parse_request
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key")
        );
        assert!(parse_request.contains(r#""input":"reducto://uploaded.pdf""#));
        assert!(parse_request.contains(r#""table_output_format":"html""#));
        assert!(parse_request.contains(r#""chunk_mode":"section""#));
        assert!(parse_request.contains(r#""ocr_system":"standard""#));
    }
}
