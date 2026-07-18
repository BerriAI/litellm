use litellm_core::error::CoreError;
use litellm_core::router::Router;
use serde_json::{Map, Value};

pub enum MessagesResponse {
    Json(Value),
    Stream(reqwest::Response),
}

pub async fn run(router: &Router, mut body: Value) -> Result<MessagesResponse, CoreError> {
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.trim().is_empty())
        .ok_or_else(|| CoreError::InvalidRequest("missing 'model' in request body".to_string()))?;
    let deployment = router.get_available_deployment(model).ok_or_else(|| {
        CoreError::InvalidRequest(format!("no deployment registered for model '{model}'"))
    })?;
    let params = &deployment.litellm_params;
    let provider_model = params.model.clone();
    let is_stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let object = body.as_object_mut().ok_or_else(|| {
        CoreError::InvalidRequest("Anthropic messages request must be a JSON object".to_string())
    })?;
    object.insert("model".to_string(), Value::String(provider_model.clone()));
    let extra_headers = is_stream.then(|| {
        Map::from_iter([(
            crate::constants::MESSAGES_STREAM_ACCEPT_HEADER.to_string(),
            Value::String(crate::constants::MESSAGES_STREAM_ACCEPT_VALUE.to_string()),
        )])
    });

    let request = crate::messages::MessagesRequest {
        model: &provider_model,
        body,
        api_key: params.api_key.as_deref(),
        api_base: params.api_base.as_deref(),
        custom_llm_provider: None,
        extra_headers,
        timeout: None,
    };
    if is_stream {
        crate::messages::stream_messages(request)
            .await
            .map(MessagesResponse::Stream)
    } else {
        crate::messages::messages(request)
            .await
            .map(MessagesResponse::Json)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;
    use litellm_core::router::{Deployment, LiteLLMParams};

    fn router(api_base: String) -> Router {
        Router::new(vec![Deployment {
            model_name: "rust-model".to_string(),
            litellm_params: LiteLLMParams {
                model: "azure_ai/claude-loadtest".to_string(),
                api_key: Some("sk-upstream".to_string()),
                api_base: Some(api_base),
            },
        }])
    }

    async fn accept_request(listener: TcpListener, response: String) -> String {
        let (mut socket, _) = listener.accept().await.expect("accepts request");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        loop {
            let count = socket.read(&mut buffer).await.expect("reads request");
            if count == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..count]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        let header_end = request
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|position| position + 4)
            .expect("request headers");
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
            let count = socket.read(&mut buffer).await.expect("reads body");
            if count == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..count]);
        }
        socket
            .write_all(response.as_bytes())
            .await
            .expect("writes response");
        String::from_utf8(request).expect("request is utf8")
    }

    #[tokio::test]
    async fn runs_non_stream_request_against_selected_deployment() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let address = listener.local_addr().expect("address");
        let body = br#"{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-loadtest","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            String::from_utf8_lossy(body)
        );
        let server = tokio::spawn(accept_request(listener, response));

        let result = run(
            &router(format!("http://{address}")),
            json!({
                "model": "rust-model",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "ping"}]
            }),
        )
        .await
        .expect("request succeeds");
        let MessagesResponse::Json(result) = result else {
            panic!("expected JSON response");
        };
        assert_eq!(result["id"], "msg_1");
        let request = server.await.expect("server completes");
        assert!(request.contains("POST /anthropic/v1/messages "));
        assert!(request.contains("claude-loadtest"), "{request}");
    }

    #[tokio::test]
    async fn passes_stream_bytes_through_unchanged() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("binds");
        let address = listener.local_addr().expect("address");
        let body = b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        );
        let response = Arc::new([response.as_bytes(), body].concat());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accepts request");
            let mut request = [0_u8; 4096];
            let count = socket.read(&mut request).await.expect("reads request");
            socket.write_all(&response).await.expect("writes response");
            String::from_utf8_lossy(&request[..count]).into_owned()
        });

        let result = run(
            &router(format!("http://{address}")),
            json!({
                "model": "rust-model",
                "stream": true,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "ping"}]
            }),
        )
        .await
        .expect("request succeeds");
        let MessagesResponse::Stream(response) = result else {
            panic!("expected stream response");
        };
        assert_eq!(
            response.bytes().await.expect("reads stream").as_ref(),
            body.as_slice()
        );
        let request = server.await.expect("server completes");
        assert!(request.contains("\"stream\":true"));
        assert!(request
            .to_ascii_lowercase()
            .contains("accept: text/event-stream"));
    }
}
