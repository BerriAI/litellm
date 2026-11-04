mod support;

use std::{collections::BTreeSet, net::SocketAddr};

use axum::http::StatusCode;
use rstest::rstest;
use support::{Gateway, Head, Raw, raw_upstream, within};
use tokio::sync::mpsc;

struct Captured {
    head: Head,
    body: Vec<u8>,
}

/// A keep-alive upstream that decodes framing like h11 and reports every request it reads.
async fn capturing() -> (SocketAddr, mpsc::UnboundedReceiver<Captured>) {
    let (sender, receiver) = mpsc::unbounded_channel();
    let address = raw_upstream(move |mut python: Raw| {
        let sender = sender.clone();
        async move {
            while let Some(head) = python.request_head().await {
                let body = match head.header("transfer-encoding") {
                    Some(_) => python.chunked_body().await,
                    None => python.body(content_length(&head)).await,
                };
                sender.send(Captured { head, body }).unwrap();
                python
                    .send("HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok")
                    .await;
            }
        }
    })
    .await;
    (address, receiver)
}

/// The length h11 reads from every Content-Length field, which may repeat one value.
fn content_length(head: &Head) -> usize {
    let lengths: BTreeSet<usize> = head
        .headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .flat_map(|(_, value)| value.split(|byte| *byte == b','))
        .map(|length| {
            std::str::from_utf8(length.trim_ascii())
                .unwrap()
                .parse()
                .unwrap()
        })
        .collect();
    assert!(
        lengths.len() < 2,
        "python got conflicting lengths {lengths:?}"
    );
    lengths.first().copied().unwrap_or(0)
}

/// RFC 9112 section 6.3: Transfer-Encoding overrides Content-Length. Forwarding both would let
/// Python split the stream at a different place than Rust did. Section 6.1 then closes the
/// connection after the response, so a request smuggled behind it never runs.
#[rstest]
#[tokio::test]
async fn a_request_framed_two_ways_reaches_python_framed_by_transfer_encoding_and_closes() {
    let (upstream, mut received) = capturing().await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("POST /v1/files HTTP/1.1\r\nhost: x\r\ncontent-length: 3\r\ntransfer-encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\nGET /next HTTP/1.1\r\nhost: x\r\n\r\n")
        .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    raw.body(2).await;
    within(raw.until_closed()).await;

    let upload = received.recv().await.unwrap();
    assert_eq!(upload.head.header("content-length"), None);
    assert_eq!(
        upload.head.header("transfer-encoding"),
        Some(&b"chunked"[..])
    );
    assert_eq!(upload.body, b"hello");
    assert!(
        received.try_recv().is_err(),
        "python ran the smuggled request"
    );
}

#[rstest]
#[case::conflicting_content_lengths("content-length: 4\r\ncontent-length: 40\r\n")]
#[case::content_length_list("content-length: 4, 40\r\n")]
#[case::signed_content_length("content-length: +4\r\n")]
#[case::chunked_not_last("transfer-encoding: chunked, identity\r\ncontent-length: 4\r\n")]
#[case::unknown_coding("transfer-encoding: xchunked\r\ncontent-length: 4\r\n")]
#[case::space_before_colon("transfer-encoding : chunked\r\ncontent-length: 4\r\n")]
#[case::folded_transfer_encoding("transfer-encoding:\r\n chunked\r\ncontent-length: 4\r\n")]
#[case::bare_cr("x-a: b\rx-b: c\r\ncontent-length: 4\r\n")]
#[case::nul_byte("x-a: b\x00c\r\ncontent-length: 4\r\n")]
#[tokio::test]
async fn ambiguous_framing_never_reaches_python(#[case] headers: &str) {
    let (upstream, mut received) = capturing().await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(format!(
        "POST /v1/files HTTP/1.1\r\nhost: x\r\n{headers}\r\nbodyGET /smuggled HTTP/1.1\r\nhost: x\r\n\r\n"
    ))
    .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::BAD_REQUEST
    );
    within(raw.closed()).await;
    assert!(received.try_recv().is_err(), "python received a request");
}

#[rstest]
#[case::bare_lf("4\nbody\n0\n\n")]
#[case::invalid_size("zz\r\nbody\r\n0\r\n\r\n")]
#[case::size_overflow("10000000000000004\r\nbody\r\n0\r\n\r\n")]
#[tokio::test]
async fn a_malformed_request_body_is_a_400_not_a_502(#[case] chunks: &str) {
    let upstream = raw_upstream(|mut python| async move { python.closed().await }).await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(format!(
        "POST /v1/files HTTP/1.1\r\nhost: x\r\ntransfer-encoding: chunked\r\n\r\n{chunks}"
    ))
    .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::BAD_REQUEST
    );
}

/// hyper decodes only the chunked layer, so any other coding would reach Python still coded but
/// framed as a plain body. uvicorn turns every h11 parse error into a 400, ignoring the 501 h11
/// suggests for codings.
#[rstest]
#[case::listed_before_chunked("transfer-encoding: gzip, chunked\r\n")]
#[case::split_across_fields("transfer-encoding: gzip\r\ntransfer-encoding: chunked\r\n")]
#[tokio::test]
async fn a_transfer_coding_other_than_chunked_is_the_400_uvicorn_sends_python_never_sees(
    #[case] headers: &str,
) {
    let (upstream, mut received) = capturing().await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(format!(
        "POST /v1/files HTTP/1.1\r\nhost: x\r\n{headers}\r\n4\r\nbody\r\n0\r\n\r\n"
    ))
    .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::BAD_REQUEST
    );
    assert!(
        received.try_recv().is_err(),
        "python received a still-coded body"
    );
}

async fn assert_python_reads_the_body_and_the_next_request(headers: &str) {
    let (upstream, mut received) = capturing().await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(format!(
        "POST /v1/files HTTP/1.1\r\nhost: x\r\n{headers}\r\npayloadGET /next HTTP/1.1\r\nhost: x\r\n\r\n"
    ))
    .await;
    for _ in 0..2 {
        assert_eq!(
            within(raw.response_head()).await.unwrap().status,
            StatusCode::OK
        );
        raw.body(2).await;
    }
    assert_eq!(received.recv().await.unwrap().body, b"payload");
    assert_eq!(received.recv().await.unwrap().head.path, "/next");
}

#[rstest]
#[case::repeated_fields("content-length: 7\r\ncontent-length: 7\r\n")]
#[case::leading_zeros("content-length: 007\r\n")]
#[tokio::test]
async fn a_content_length_h11_reads_one_way_reaches_python_framed_the_same(#[case] headers: &str) {
    assert_python_reads_the_body_and_the_next_request(headers).await;
}

/// h11 accepts a Content-Length list that repeats one value, as RFC 9110 section 8.6 allows.
#[rstest]
#[ignore = "divergence: hyper's server parses each Content-Length field as one number and 400s a list"]
#[tokio::test]
async fn a_content_length_list_of_one_value_reaches_python() {
    assert_python_reads_the_body_and_the_next_request("content-length: 7, 7\r\n").await;
}

#[rstest]
#[case::leading_zeros("0004\r\nbody\r\n0\r\n\r\n", "body")]
#[case::uppercase_hex("A\r\nbody\r\nbody\r\n0\r\n\r\n", "body\r\nbody")]
#[case::whitespace_after_size("4 \r\nbody\r\n0\r\n\r\n", "body")]
#[case::extension_and_trailer("4;a=\"b;c\"\r\nbody\r\n0\r\nx-sum: 1\r\n\r\n", "body")]
#[tokio::test]
async fn chunk_framing_h11_accepts_reaches_python_decoded(
    #[case] chunks: &str,
    #[case] body: &str,
) {
    let (upstream, mut received) = capturing().await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send(format!(
        "POST /v1/files HTTP/1.1\r\nhost: x\r\ntransfer-encoding: chunked\r\n\r\n{chunks}GET /next HTTP/1.1\r\nhost: x\r\n\r\n"
    ))
    .await;
    for _ in 0..2 {
        assert_eq!(
            within(raw.response_head()).await.unwrap().status,
            StatusCode::OK
        );
        raw.body(2).await;
    }
    assert_eq!(received.recv().await.unwrap().body, body.as_bytes());
    assert_eq!(received.recv().await.unwrap().head.path, "/next");
}

/// h11 reads a head whose lines end in a bare LF, so Python gets the same request through Rust.
#[rstest]
#[tokio::test]
async fn a_head_with_bare_lf_line_endings_reaches_python() {
    let (upstream, mut received) = capturing().await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("POST /v1/files HTTP/1.1\nhost: x\nx-a: b\ncontent-length: 4\n\nbody")
        .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    let upload = received.recv().await.unwrap();
    assert_eq!(upload.head.header("x-a"), Some(&b"b"[..]));
    assert_eq!(upload.body, b"body");
}

/// h11 decodes a chunked body from an HTTP/1.0 client, and RFC 9112 section 6.1 only asks the
/// server to close the connection afterwards.
#[rstest]
#[ignore = "divergence: hyper rejects Transfer-Encoding on HTTP/1.0 with a 400 that uvicorn never sends"]
#[tokio::test]
async fn an_http10_chunked_upload_reaches_python() {
    let (upstream, mut received) = capturing().await;
    let gateway = Gateway::start(upstream).await;
    let mut raw = gateway.connect().await;
    raw.send("POST /v1/files HTTP/1.0\r\nhost: x\r\ntransfer-encoding: chunked\r\n\r\n4\r\nbody\r\n0\r\n\r\n")
        .await;
    assert_eq!(
        within(raw.response_head()).await.unwrap().status,
        StatusCode::OK
    );
    assert_eq!(received.recv().await.unwrap().body, b"body");
}
