mod support;

use std::time::Duration;

use axum::body::to_bytes;
use futures_util::StreamExt;
use rstest::rstest;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

const COMPLETION: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-6-astra","choices":[{"index":0,"message":{"role":"assistant","content":"hi","refusal":null},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4},"system_fingerprint":"fp"}"#;
const STREAM: &str = "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\ndata: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\ndata: [DONE]\n\n";

fn request(stream: bool) -> Value {
    json!({
        "model": "public/model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
        "stream_options": stream.then(|| json!({"include_usage": true})),
        "caching": false,
        "metadata": {"tag": "x"}
    })
}

#[rstest]
#[case::json(false, COMPLETION, "application/json")]
#[case::sse(true, STREAM, "text/event-stream")]
#[tokio::test]
async fn openai_chat_passes_the_provider_response_through(
    #[case] stream: bool,
    #[case] upstream_body: &'static str,
    #[case] content_type: &str,
    #[values("/v1/chat/completions", "/engines/public%2Fmodel/chat/completions")] route: &str,
) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({
            "model": "gpt-6-astra",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
            "stream_options": stream.then(|| json!({"include_usage": true})),
        })))
        .respond_with(ResponseTemplate::new(200).set_body_raw(upstream_body, content_type))
        .expect(1)
        .mount(&upstream)
        .await;

    let response = support::post(
        support::app("openai/gpt-6-astra", &format!("{}/v1", upstream.uri())),
        route,
        request(stream),
    )
    .await;

    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], content_type);
    assert_eq!(
        to_bytes(response.into_body(), 1 << 20).await.unwrap(),
        upstream_body
    );
}

#[rstest]
#[case::json(false)]
#[case::sse(true)]
#[tokio::test]
async fn an_openai_error_keeps_its_status_and_message(#[case] stream: bool) {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).set_body_raw(
            r#"{"error":{"message":"Rate limit reached","type":"requests","code":"rate_limit_exceeded"}}"#,
            "application/json",
        ))
        .mount(&upstream)
        .await;

    let response = support::post(
        support::app("openai/gpt-6-astra", &upstream.uri()),
        "/v1/chat/completions",
        request(stream),
    )
    .await;

    assert_eq!(response.status(), 429);
    let body = support::json(response).await;
    assert_eq!(body["error"]["message"], "Rate limit reached");
    assert_eq!(body["error"]["type"], "rate_limit_error");
}

/// Sends the SSE head and one event, then holds the connection open until told to send the
/// rest, or drops it short of the end when `truncate` is set.
async fn paused_upstream(release: tokio::sync::oneshot::Receiver<()>, truncate: bool) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = vec![0; 64 * 1024];
        let _ = socket.read(&mut request).await;
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n{:x}\r\n{FIRST_EVENT}\r\n",
                    FIRST_EVENT.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
        let _ = release.await;
        if truncate {
            return;
        }
        let _ = socket
            .write_all(format!("{:x}\r\n{DONE}\r\n0\r\n\r\n", DONE.len()).as_bytes())
            .await;
    });
    base
}

const FIRST_EVENT: &str = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n";
const DONE: &str = "data: [DONE]\n\n";

#[rstest]
#[tokio::test]
async fn the_gateway_forwards_each_event_while_the_provider_is_still_streaming() {
    let (release, released) = tokio::sync::oneshot::channel();
    let base = paused_upstream(released, false).await;
    let response = support::post(
        support::app("openai/gpt-6-astra", &base),
        "/v1/chat/completions",
        request(true),
    )
    .await;
    assert_eq!(response.status(), 200);
    let mut body = response.into_body().into_data_stream();

    let first = tokio::time::timeout(Duration::from_secs(2), body.next())
        .await
        .expect("the first event reaches the client before the provider finishes")
        .unwrap()
        .unwrap();
    assert_eq!(first, FIRST_EVENT);
    release.send(()).unwrap();
    assert_eq!(body.next().await.unwrap().unwrap(), DONE);
    assert!(body.next().await.is_none());
}

#[rstest]
#[tokio::test]
async fn a_stream_cut_by_the_provider_ends_with_an_openai_error_event() {
    let (release, released) = tokio::sync::oneshot::channel();
    let base = paused_upstream(released, true).await;
    let response = support::post(
        support::app("openai/gpt-6-astra", &base),
        "/v1/chat/completions",
        request(true),
    )
    .await;
    release.send(()).unwrap();

    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    let error = text
        .strip_prefix(FIRST_EVENT)
        .and_then(|rest| rest.strip_prefix("data: "))
        .and_then(|rest| rest.strip_suffix("\n\n"))
        .unwrap_or_else(|| panic!("expected the first event then an error event, got {text:?}"));
    let error: Value = serde_json::from_str(error).unwrap();
    assert_eq!(error["error"]["code"], 500);
}

#[rstest]
#[tokio::test]
async fn openai_like_chat_streaming_is_still_declined() {
    let upstream = MockServer::start().await;
    let response = support::post(
        support::app("openai_like/gpt-6-astra", &upstream.uri()),
        "/v1/chat/completions",
        json!({"model": "public/model", "messages": [{"role": "user", "content": "hi"}], "stream": true}),
    )
    .await;
    assert_eq!(response.status(), 501);
    assert!(upstream.received_requests().await.unwrap().is_empty());
}
