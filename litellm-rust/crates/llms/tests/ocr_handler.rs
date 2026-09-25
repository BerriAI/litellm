use std::time::Duration;

use litellm_llms::base_llm::ocr::{error::Error, handler::read_response_bytes};
use rstest::rstest;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

/// Answers one request with raw `response` bytes and then holds the connection open, so a
/// read that waits for the rest of an oversized body hangs instead of passing.
async fn read_bounded(response: String, limit: usize) -> Result<bytes::Bytes, Error> {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0; 4096];
        assert!(socket.read(&mut request).await.unwrap() > 0);
        socket.write_all(response.as_bytes()).await.unwrap();
        std::future::pending::<()>().await;
    });
    let response = reqwest::Client::new()
        .get(format!("http://{address}"))
        .send()
        .await
        .unwrap();
    let result =
        tokio::time::timeout(Duration::from_secs(2), read_response_bytes(response, limit)).await;
    server.abort();
    result.expect("bounded reads must finish without waiting for the rest of an oversized body")
}

#[rstest]
#[case::declared("HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\nabcdefgh")]
#[case::chunked(
    "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nabcd\r\n4\r\nefgh\r\n0\r\n\r\n"
)]
#[tokio::test]
async fn a_body_of_exactly_the_limit_is_read(#[case] response: &str) {
    assert_eq!(read_bounded(response.into(), 8).await.unwrap(), "abcdefgh");
}

#[rstest]
#[case::declared("HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\n")]
#[case::chunked("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nabcd\r\n5\r\nefghi\r\n")]
#[tokio::test]
async fn a_body_over_the_limit_is_rejected(#[case] response: &str) {
    assert!(matches!(
        read_bounded(response.into(), 8).await,
        Err(Error::TooLarge { limit: 8 })
    ));
}

#[rstest]
#[case::declared("Content-Length: 1000000")]
#[case::chunked("Transfer-Encoding: chunked")]
#[tokio::test]
async fn an_oversized_error_keeps_its_status_and_a_bounded_body_without_draining(
    #[case] headers: &str,
) {
    let prefix = "x".repeat(4096);
    let body = match headers.starts_with("Transfer") {
        true => format!("{:x}\r\n{prefix}\r\n", prefix.len()),
        false => prefix.clone(),
    };

    let error = read_bounded(
        format!("HTTP/1.1 429 Too Many Requests\r\n{headers}\r\n\r\n{body}"),
        prefix.len(),
    )
    .await
    .unwrap_err();

    let Error::Transport(litellm_http::transport::Error::Http { status, body }) = error else {
        panic!("unexpected error: {error}");
    };
    assert_eq!(status, 429);
    assert_eq!(body, prefix);
}
